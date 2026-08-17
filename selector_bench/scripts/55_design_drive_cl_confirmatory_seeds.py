#!/usr/bin/env python3
"""Design confirmatory seed count from audit-only crossed pilot matrices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from selector_bench.continual.seed_design import build_seed_design_payload, sha256


def parse_seed_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed IDs must be comma-separated integers") from exc
    if len(result) < 2 or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("seed IDs must contain at least two unique integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-matrix", type=Path, required=True)
    parser.add_argument("--candidate-seed-ids", type=parse_seed_ids, required=True)
    parser.add_argument("--minimum-relevant-effect", type=float, required=True)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--family-alpha", type=float, default=0.05)
    parser.add_argument("--simulation-repetitions", type=int, default=10000)
    parser.add_argument("--simulation-seed", type=int, default=0)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--generator-module-source", type=Path)
    parser.add_argument("--generator-entrypoint-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pilot_matrix.is_symlink() or not args.pilot_matrix.is_file():
        parser.error(f"missing or symlinked pilot matrix: {args.pilot_matrix}")
    if args.minimum_relevant_effect <= 0.0:
        parser.error("minimum relevant effect must be positive")
    if not 0.0 < args.target_power < 1.0 or not 0.0 < args.family_alpha < 1.0:
        parser.error("power and alpha must lie strictly between zero and one")
    if args.simulation_repetitions < 10000:
        parser.error("confirmatory power/calibration requires at least 10000 simulations")
    if not (args.repository / ".git").exists():
        parser.error("seed design repository is not a Git worktree")
    return args


def repository_path(repository: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside --repository") from exc


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    module_source = (
        args.generator_module_source
        or Path(__file__).resolve().parents[1]
        / "selector_bench"
        / "continual"
        / "seed_design.py"
    ).resolve()
    entrypoint_source = (args.generator_entrypoint_source or Path(__file__)).resolve()
    imported_module = (
        Path(__file__).resolve().parents[1]
        / "selector_bench"
        / "continual"
        / "seed_design.py"
    )
    if sha256(module_source) != sha256(imported_module):
        raise SystemExit("generator module source does not match the imported implementation")
    if sha256(entrypoint_source) != sha256(Path(__file__)):
        raise SystemExit("generator entrypoint source does not match the executing CLI")
    payload = build_seed_design_payload(
        args.pilot_matrix.resolve(),
        pilot_reference=os.path.relpath(args.pilot_matrix.resolve(), args.output.resolve().parent),
        candidate_seed_ids=args.candidate_seed_ids,
        minimum_relevant_effect=args.minimum_relevant_effect,
        target_power=args.target_power,
        family_alpha=args.family_alpha,
        simulation_repetitions=args.simulation_repetitions,
        simulation_seed=args.simulation_seed,
        generator_module_repository_path=repository_path(
            repository, module_source, "generator module"
        ),
        generator_module_sha256=sha256(module_source),
        generator_entrypoint_repository_path=repository_path(
            repository, entrypoint_source, "generator entrypoint"
        ),
        generator_entrypoint_sha256=sha256(entrypoint_source),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate confirmatory seeds against one complete frozen Holm family."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from selector_bench.continual.seed_design import (
    PRODUCTION_POWER_SIMULATION_MINIMUM,
    SYNTHETIC_FIXTURE_SIMULATION_MINIMUM,
    build_seed_design_payload,
    finite_integer,
    finite_real,
    sha256,
)
from selector_bench.continual.statistics import StatisticsError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-spec", type=Path, required=True)
    parser.add_argument("--pilot-matrix", type=Path, required=True)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--simulation-repetitions", type=int, default=10000)
    parser.add_argument("--simulation-seed", type=int, default=0)
    parser.add_argument(
        "--synthetic-cpu-fixture",
        action="store_true",
        help="Permit the explicit navsim-fixture low-repetition CPU test contract.",
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--generator-module-source", type=Path)
    parser.add_argument("--generator-entrypoint-source", type=Path)
    parser.add_argument("--inference-module-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for label, path in (
        ("family specification", args.family_spec),
        ("pilot matrix", args.pilot_matrix),
    ):
        if path.is_symlink() or not path.is_file():
            parser.error(f"missing or symlinked {label}: {path}")
    try:
        args.target_power = finite_real(args.target_power, "target power")
        args.simulation_repetitions = finite_integer(
            args.simulation_repetitions, "simulation repetitions"
        )
        args.simulation_seed = finite_integer(args.simulation_seed, "simulation seed")
        if not 0.0 < args.target_power < 1.0:
            raise StatisticsError("target power must lie strictly between zero and one")
        minimum_repetitions = (
            SYNTHETIC_FIXTURE_SIMULATION_MINIMUM
            if args.synthetic_cpu_fixture
            else PRODUCTION_POWER_SIMULATION_MINIMUM
        )
        if args.simulation_repetitions < minimum_repetitions:
            raise StatisticsError(
                "power/FWER outer simulations are below the selected scope minimum"
            )
    except StatisticsError as exc:
        parser.error(str(exc))
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
    inference_source = (
        args.inference_module_source
        or Path(__file__).resolve().parents[1]
        / "selector_bench"
        / "continual"
        / "statistics.py"
    ).resolve()
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
    imported_inference = (
        Path(__file__).resolve().parents[1]
        / "selector_bench"
        / "continual"
        / "statistics.py"
    )
    if sha256(inference_source) != sha256(imported_inference):
        raise SystemExit("inference module source does not match the imported implementation")
    payload = build_seed_design_payload(
        args.family_spec.resolve(),
        args.pilot_matrix.resolve(),
        repository=repository,
        family_spec_repository_path=repository_path(
            repository, args.family_spec.resolve(), "family specification"
        ),
        pilot_reference=os.path.relpath(args.pilot_matrix.resolve(), args.output.resolve().parent),
        simulation_scope=(
            "synthetic_cpu_fixture"
            if args.synthetic_cpu_fixture
            else "confirmatory"
        ),
        target_power=args.target_power,
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
        inference_module_repository_path=repository_path(
            repository, inference_source, "inference module"
        ),
        inference_module_sha256=sha256(inference_source),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

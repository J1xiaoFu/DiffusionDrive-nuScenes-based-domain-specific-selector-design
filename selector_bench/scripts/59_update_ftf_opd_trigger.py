#!/usr/bin/env python3
"""Update the FTF-OPD functional trigger from paired sentinel evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_snapshot(path: Path) -> Any:
    from selector_bench.continual.functional_trigger import (
        FunctionalTriggerError,
        SentinelSnapshot,
    )

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"session_id", "planning_risk", "perception_risk"}
    if not rows or not required.issubset(rows[0]):
        raise FunctionalTriggerError(f"{path} lacks required columns {sorted(required)}")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        session_id = str(row.get("session_id", ""))
        if not session_id or session_id in indexed:
            raise FunctionalTriggerError(f"{path} contains duplicate/empty session IDs")
        indexed[session_id] = row
    ordered = [indexed[key] for key in sorted(indexed)]

    def values(name: str) -> tuple[float, ...] | None:
        present = [str(row.get(name, "")).strip() != "" for row in ordered]
        if not any(present):
            return None
        if not all(present):
            raise FunctionalTriggerError(f"optional {name} must be complete when present")
        return tuple(float(row[name]) for row in ordered)

    return SentinelSnapshot(
        session_ids=tuple(row["session_id"] for row in ordered),
        planning_risk=tuple(float(row["planning_risk"]) for row in ordered),
        perception_risk=tuple(float(row["perception_risk"]) for row in ordered),
        collision_risk=values("collision_risk"),
        student_mode_risk=values("student_mode_risk"),
        teacher_mode_risk=values("teacher_mode_risk"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--current-csv", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--planning-mde", type=float, default=0.03)
    parser.add_argument("--perception-mde", type=float, default=0.03)
    parser.add_argument("--collision-mde", type=float, default=0.01)
    parser.add_argument("--mode-mde", type=float, default=0.03)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from selector_bench.continual.functional_trigger import (
        FunctionalTriggerConfig,
        FunctionalTriggerController,
    )

    for path in (args.baseline_csv, args.current_csv):
        if not path.is_file():
            raise SystemExit(f"sentinel evaluation is missing: {path}")
    if args.state.exists():
        controller = FunctionalTriggerController.from_state_dict(
            json.loads(args.state.read_text())
        )
    else:
        controller = FunctionalTriggerController(
            FunctionalTriggerConfig(
                planning_mde=args.planning_mde,
                perception_mde=args.perception_mde,
                collision_mde=args.collision_mde,
                mode_mde=args.mode_mde,
                confidence=args.confidence,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    baseline = load_snapshot(args.baseline_csv)
    current = load_snapshot(args.current_csv)
    decision = controller.observe(baseline, current)
    payload = decision.to_jsonable()
    payload["sentinel_session_count"] = len(current.session_ids)
    payload["scientific_semantics"] = (
        "paired session-level functional degradation; sentinel data never enter gradients"
    )
    atomic_json(args.decision_output, payload)
    atomic_json(args.state, controller.state_dict())
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

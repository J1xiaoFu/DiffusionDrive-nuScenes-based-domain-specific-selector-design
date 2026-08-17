#!/usr/bin/env python3
"""Create a CPU-only pre-run registration for one claim-eligible Drive-CL run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

from selector_bench.continual.claim_protocol import (
    RUN_BUDGET_FIELDS,
    RUN_REGISTRATION_SCHEMA,
    load_json,
    sha256,
    validate_method_arm,
    validate_method_registry,
)
from selector_bench.continual.statistics import StatisticsError


METHOD_BY_ARM = {
    "sequential": "sequential",
    "planner_only": "planner_only",
    "ewc": "ewc",
    "agem": "agem",
    "aler_drive": "aler_drive",
    "step_matched_replay": "step_matched_replay",
    "full_exposure_replay": "full_exposure_replay",
    "lwf": "lwf",
    "fixed_opd": "drive_opd_fixed",
    "ema099_opd": "drive_opd_ema099",
    "perception_only": "drive_perception_only",
    "planning_only": "drive_planning_only",
}


def git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode != 0:
        raise StatisticsError(f"Git registration command failed: {process.stdout}")
    return process.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--method-registry", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-budget", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(METHOD_BY_ARM), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--end-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.protocol_manifest,
        args.experiment_config,
        args.method_registry,
        args.environment_lock,
        args.source_checkpoint,
        args.target_budget,
    ):
        if path.is_symlink() or not path.is_file():
            parser.error(f"missing or symlinked registration input: {path}")
    if args.output.exists():
        parser.error(f"refusing to overwrite run registration: {args.output}")
    if args.end_epoch < args.start_epoch:
        parser.error("end epoch must not precede start epoch")
    return args


def main() -> None:
    args = parse_args()
    source_repository = args.source_repository.resolve()
    if git(source_repository, "status", "--porcelain"):
        raise StatisticsError("run registration requires a clean source worktree")
    source_commit = git(source_repository, "rev-parse", "HEAD")
    source_tree = git(source_repository, "rev-parse", "HEAD^{tree}")
    protocol = load_json(args.protocol_manifest, "registration protocol")
    experiment = load_json(args.experiment_config, "registration experiment config")
    if experiment.get("protocol_content_sha256") != protocol.get("content_sha256"):
        raise StatisticsError("registration config/protocol content identity mismatch")
    registry = validate_method_registry(args.method_registry, sha256(args.method_registry))
    method_id = METHOD_BY_ARM[args.arm]
    validate_method_arm(method_id, args.arm, registry=registry)
    budget = load_json(args.target_budget, "run target budget")
    if set(budget) != set(RUN_BUDGET_FIELDS) or any(
        not isinstance(budget[field], int) or budget[field] < 0
        for field in RUN_BUDGET_FIELDS
    ):
        raise StatisticsError("run target budget is malformed")
    payload = {
        "schema": RUN_REGISTRATION_SCHEMA,
        "run_id": uuid.uuid4().hex,
        "method_id": method_id,
        "arm": args.arm,
        "seed": args.seed,
        "stage_index": args.stage_index,
        "dataset_identity": protocol["dataset_identity"],
        "source_code": {
            "repository": str(source_repository),
            "commit": source_commit,
            "tree": source_tree,
            "dirty": False,
        },
        "protocol_manifest_sha256": sha256(args.protocol_manifest),
        "protocol_content_sha256": protocol["content_sha256"],
        "experiment_config_sha256": sha256(args.experiment_config),
        "method_registry_sha256": sha256(args.method_registry),
        "environment_lock_sha256": sha256(args.environment_lock),
        "source_checkpoint_sha256": sha256(args.source_checkpoint),
        "stage_start_state_sha256": sha256(args.source_checkpoint),
        "start_epoch": args.start_epoch,
        "end_epoch": args.end_epoch,
        "max_steps": 0,
        "target_budget": budget,
        "target_budget_source": str(args.target_budget.resolve()),
        "target_budget_source_sha256": sha256(args.target_budget),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

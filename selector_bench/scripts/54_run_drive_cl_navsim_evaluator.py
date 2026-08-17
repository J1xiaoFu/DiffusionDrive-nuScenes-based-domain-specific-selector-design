#!/usr/bin/env python3
"""Run a frozen evaluator and seal checkpoint-to-CSV lineage in one receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selector_bench.continual.claim_protocol import (
    EVALUATOR_RUN_SCHEMA,
    SEALED_ACCESS_SCHEMA,
    SEALED_LEDGER_SCHEMA,
    git,
    load_json,
    load_training_run_contract,
    require_equal,
    resolve,
    sha256,
    validate_metric_registry,
    verify_frozen_file,
)
from selector_bench.continual.evaluation_contract import load_evaluation_cell_contract
from selector_bench.continual.statistics import StatisticsError
from selector_bench.continual.statistics import PDM_DEFAULT_METRICS, read_pdm_rows


COMMAND_SCHEMA = "selector_bench.drive_cl_evaluator_command.v1"
PLACEHOLDERS = (
    "{checkpoint}",
    "{token_file}",
    "{output_csv}",
    "{evaluator_entrypoint}",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--evaluation-cell-receipt", type=Path, required=True)
    parser.add_argument("--evaluator-repository", type=Path, required=True)
    parser.add_argument("--evaluator-source-commit", required=True)
    parser.add_argument("--evaluator-entrypoint", type=Path, required=True)
    parser.add_argument("--command-spec", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--metric-registry", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--deterministic-seed", type=int, default=0)
    parser.add_argument("--access-repository", type=Path)
    parser.add_argument("--access-receipt", type=Path)
    parser.add_argument("--access-commit")
    args = parser.parse_args()
    inputs = (
        args.training_protocol,
        args.training_result,
        args.evaluation_cell_receipt,
        args.evaluator_entrypoint,
        args.command_spec,
        args.environment_lock,
        args.metric_registry,
    )
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            parser.error(f"missing or symlinked input: {path}")
    if not (args.evaluator_repository / ".git").exists():
        parser.error("evaluator repository is not a Git worktree")
    for path in (args.output_csv, args.output_receipt, args.stdout_log, args.stderr_log):
        if path.exists():
            parser.error(f"refusing to reuse pre-existing evaluator output: {path}")
    return args


def verify_sealed_access(
    args: argparse.Namespace,
    *,
    evaluation_receipt_sha256: str,
    protocol_content_sha256: str,
    evaluation_domain: str,
    evaluator_started_at: datetime,
) -> dict[str, Any] | None:
    supplied = (args.access_repository, args.access_receipt, args.access_commit)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise StatisticsError("sealed access requires repository, receipt and full commit")
    access_repository = args.access_repository.resolve()
    access_receipt = args.access_receipt.resolve()
    access_commit = str(args.access_commit)
    access_path = verify_frozen_file(access_repository, access_receipt, access_commit)
    payload = load_json(access_receipt, "sealed-test access receipt")
    receipt_keys = {
        "schema",
        "status",
        "family_id",
        "family_spec_repository_path",
        "family_spec_sha256",
        "family_freeze_commit",
        "access_ledger_repository_path",
        "access_ledger_sha256",
        "access_event_id",
        "access_event_index",
        "authorized_at_utc",
        "evaluation_cell_receipt_sha256",
        "protocol_content_sha256",
        "evaluation_domain",
    }
    if set(payload) != receipt_keys:
        raise StatisticsError("sealed access receipt fields are not canonical")
    require_equal(payload.get("schema"), SEALED_ACCESS_SCHEMA, "sealed access schema")
    require_equal(payload.get("status"), "authorized", "sealed access status")
    event_id = payload.get("access_event_id")
    if not isinstance(event_id, str) or len(event_id) != 32:
        raise StatisticsError("sealed access ledger lacks a unique event ID")
    authorized_at = strict_utc(payload.get("authorized_at_utc"), "authorization")
    require_equal(
        payload.get("evaluation_cell_receipt_sha256"),
        evaluation_receipt_sha256,
        "sealed access evaluation receipt SHA256",
    )
    require_equal(
        payload.get("protocol_content_sha256"),
        protocol_content_sha256,
        "sealed access protocol content SHA256",
    )
    require_equal(
        payload.get("evaluation_domain"), evaluation_domain, "sealed access domain"
    )
    family_path = resolve(
        access_repository,
        payload.get("family_spec_repository_path"),
        "sealed access family specification",
    )
    family_commit = str(payload.get("family_freeze_commit", ""))
    verify_frozen_file(access_repository, family_path, family_commit)
    require_equal(
        payload.get("family_spec_sha256"), sha256(family_path), "sealed access family SHA256"
    )
    family = load_json(family_path, "sealed access family specification")
    family_id = family.get("family_id")
    require_equal(payload.get("family_id"), family_id, "sealed access family ID")
    ledger_path = resolve(
        access_repository,
        family.get("sealed_test_access_ledger_repository_path"),
        "canonical sealed-test access ledger",
    )
    require_equal(
        payload.get("access_ledger_repository_path"),
        ledger_path.relative_to(access_repository).as_posix(),
        "sealed access canonical ledger path",
    )
    ledger_relative = ledger_path.relative_to(access_repository).as_posix()
    empty_bytes = git(access_repository, "show", f"{family_commit}:{ledger_relative}")
    require_equal(
        family.get("sealed_test_access_ledger_empty_sha256"),
        hashlib.sha256(empty_bytes).hexdigest(),
        "family empty access-ledger SHA256",
    )
    try:
        empty_ledger = json.loads(empty_bytes)
    except json.JSONDecodeError as exc:
        raise StatisticsError("family commit contains an invalid empty access ledger") from exc
    if (
        not isinstance(empty_ledger, dict)
        or set(empty_ledger) != {"schema", "family_id", "events"}
        or empty_ledger.get("schema") != SEALED_LEDGER_SCHEMA
        or empty_ledger.get("family_id") != family_id
        or empty_ledger.get("events") != []
    ):
        raise StatisticsError("family commit does not contain one canonical empty ledger")
    git(access_repository, "merge-base", "--is-ancestor", family_commit, access_commit)
    ancestry = [
        value
        for value in git(
            access_repository,
            "rev-list",
            "--reverse",
            "--ancestry-path",
            f"{family_commit}..{access_commit}",
        )
        .decode()
        .splitlines()
        if value
    ]
    if not ancestry or ancestry[-1] != access_commit:
        raise StatisticsError("sealed access commit is not a linear descendant of family freeze")
    previous = family_commit
    for commit in ancestry:
        parents = git(access_repository, "show", "-s", "--format=%P", commit).decode().split()
        if parents != [previous]:
            raise StatisticsError("sealed access authorization history may not merge or fork")
        ledger_bytes = git(access_repository, "show", f"{commit}:{ledger_relative}")
        if commit != access_commit and ledger_bytes != empty_bytes:
            raise StatisticsError(
                "canonical access ledger changed before the authorized access commit"
            )
        previous = commit
    verify_frozen_file(access_repository, ledger_path, access_commit)
    ledger = load_json(ledger_path, "sealed-test access ledger")
    if set(ledger) != {"schema", "family_id", "events"}:
        raise StatisticsError("sealed-test access ledger fields are not canonical")
    require_equal(ledger.get("schema"), SEALED_LEDGER_SCHEMA, "access ledger schema")
    require_equal(ledger.get("family_id"), family_id, "access ledger family ID")
    events = ledger.get("events")
    if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], dict):
        raise StatisticsError("first final-test access requires exactly one canonical ledger event")
    event = events[0]
    event_keys = {
        "access_event_id",
        "authorized_at_utc",
        "family_id",
        "family_spec_sha256",
        "evaluation_cell_receipt_sha256",
        "protocol_content_sha256",
        "evaluation_domain",
    }
    if set(event) != event_keys:
        raise StatisticsError("sealed-test access event fields are not canonical")
    require_equal(event.get("access_event_id"), event_id, "access ledger event ID")
    require_equal(event.get("family_id"), family_id, "access event family ID")
    require_equal(
        event.get("family_spec_sha256"), sha256(family_path), "access event family SHA256"
    )
    require_equal(
        event.get("evaluation_cell_receipt_sha256"),
        evaluation_receipt_sha256,
        "access event evaluation receipt SHA256",
    )
    require_equal(
        event.get("protocol_content_sha256"),
        protocol_content_sha256,
        "access event protocol content SHA256",
    )
    require_equal(event.get("evaluation_domain"), evaluation_domain, "access event domain")
    require_equal(event.get("authorized_at_utc"), payload.get("authorized_at_utc"), "access event time")
    require_equal(payload.get("access_event_index"), 0, "derived access event index")
    require_equal(
        payload.get("access_ledger_sha256"), sha256(ledger_path), "sealed access ledger SHA256"
    )
    family_time = git_commit_utc(access_repository, family_commit)
    access_time = git_commit_utc(access_repository, access_commit)
    if not family_time <= authorized_at <= access_time <= evaluator_started_at:
        raise StatisticsError(
            "sealed access chronology must satisfy family <= authorization <= access <= evaluator"
        )
    return {
        "status": "verified",
        "access_repository": str(access_repository),
        "access_receipt": str(access_receipt),
        "access_receipt_sha256": sha256(access_receipt),
        "access_commit": access_commit,
        "access_repository_path": access_path,
        "family_spec": str(family_path),
        "family_spec_sha256": sha256(family_path),
        "family_freeze_commit": family_commit,
        "family_id": family_id,
        "access_ledger": str(ledger_path),
        "access_ledger_sha256": sha256(ledger_path),
        "access_event_index": 0,
        "access_event_id": event_id,
        "authorized_at_utc": authorized_at.isoformat(),
        "family_commit_time_utc": family_time.isoformat(),
        "access_commit_time_utc": access_time.isoformat(),
        "evaluator_started_at_utc": evaluator_started_at.isoformat(),
    }


def strict_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise StatisticsError(f"sealed access {label} time must be a UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StatisticsError(f"sealed access {label} time is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise StatisticsError(f"sealed access {label} time is not UTC")
    if parsed.isoformat() != value:
        raise StatisticsError(f"sealed access {label} time is not canonical ISO-8601")
    return parsed


def git_commit_utc(repository: Path, commit: str) -> datetime:
    value = git(repository, "show", "-s", "--format=%cI", commit).decode().strip()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StatisticsError("Git commit time is malformed") from exc
    if parsed.tzinfo is None:
        raise StatisticsError("Git commit time lacks an offset")
    return parsed.astimezone(timezone.utc)


def render_command(spec: dict[str, Any], replacements: dict[str, str]) -> list[str]:
    require_equal(spec.get("schema"), COMMAND_SCHEMA, "evaluator command schema")
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(value, str) or not value for value in argv
    ):
        raise StatisticsError("evaluator command argv must contain non-empty strings")
    joined = "\0".join(argv)
    for placeholder in PLACEHOLDERS:
        if joined.count(placeholder) != 1:
            raise StatisticsError(
                f"evaluator command must contain {placeholder} exactly once"
            )
    rendered = []
    for value in argv:
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        rendered.append(value)
    if any("{" in value or "}" in value for value in rendered):
        raise StatisticsError("evaluator command contains an unknown placeholder")
    return rendered


def main() -> None:
    args = parse_args()
    evaluator_started_at = datetime.now(timezone.utc)
    training = load_training_run_contract(args.training_protocol, args.training_result)
    evaluation = load_evaluation_cell_contract(args.evaluation_cell_receipt)
    split = str(evaluation.receipt["split"])
    sealed_access = verify_sealed_access(
        args,
        evaluation_receipt_sha256=sha256(evaluation.receipt_path),
        protocol_content_sha256=str(evaluation.protocol["content_sha256"]),
        evaluation_domain=evaluation.evaluation_domain,
        evaluator_started_at=evaluator_started_at,
    )
    if split == "test" and sealed_access is None:
        raise StatisticsError("test evaluation requires a frozen sealed-access receipt")
    if split == "audit" and sealed_access is not None:
        raise StatisticsError("audit evaluation may not use a sealed-test access receipt")

    evaluator_repository = args.evaluator_repository.resolve()
    entrypoint_relative = verify_frozen_file(
        evaluator_repository, args.evaluator_entrypoint, args.evaluator_source_commit
    )
    command_relative = verify_frozen_file(
        evaluator_repository, args.command_spec, args.evaluator_source_commit
    )
    command_spec = load_json(args.command_spec, "evaluator command specification")
    validate_metric_registry(args.metric_registry, sha256(args.metric_registry))
    command = render_command(
        command_spec,
        {
            "{checkpoint}": str(training.endpoint_path),
            "{token_file}": str(evaluation.token_file_path),
            "{output_csv}": str(args.output_csv.resolve()),
            "{evaluator_entrypoint}": str(args.evaluator_entrypoint.resolve()),
        },
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    args.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PDM_DETERMINISTIC_GLOBAL_SEED"] = str(args.deterministic_seed)
    started = datetime.now(timezone.utc).isoformat()
    with args.stdout_log.open("wb") as stdout, args.stderr_log.open("wb") as stderr:
        process = subprocess.run(
            command,
            cwd=str(evaluator_repository),
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    completed = datetime.now(timezone.utc).isoformat()
    if process.returncode != 0:
        raise StatisticsError(
            f"frozen evaluator failed with return code {process.returncode}; receipt not emitted"
        )
    if args.output_csv.is_symlink() or not args.output_csv.is_file():
        raise StatisticsError("frozen evaluator did not create a regular output CSV")
    evaluated_rows = read_pdm_rows(
        args.output_csv,
        expected_tokens=set(evaluation.tokens),
        require_all_valid=True,
        required_metrics=PDM_DEFAULT_METRICS,
    )
    metric_schema_sha256 = hashlib.sha256(
        json.dumps(list(PDM_DEFAULT_METRICS), separators=(",", ":")).encode()
    ).hexdigest()
    evaluator_tree = git(
        evaluator_repository,
        "rev-parse",
        f"{args.evaluator_source_commit}^{{tree}}",
    ).decode().strip()
    payload = {
        "schema": EVALUATOR_RUN_SCHEMA,
        "status": "completed",
        "return_code": process.returncode,
        "run_id": training.protocol["run_id"],
        "training_arm": training.protocol["arm"],
        "training_seed": training.protocol["seed"],
        "checkpoint_stage_index": training.protocol["stage_index"],
        "dataset_identity": training.protocol["dataset_identity"],
        "training_protocol": str(training.protocol_path),
        "training_protocol_sha256": training.protocol_sha256,
        "training_result": str(training.result_path),
        "training_result_sha256": training.result_sha256,
        "checkpoint": str(training.endpoint_path),
        "checkpoint_sha256": training.endpoint_sha256,
        "evaluation_cell_receipt": str(evaluation.receipt_path),
        "evaluation_cell_receipt_sha256": sha256(evaluation.receipt_path),
        "evaluation_stage_index": evaluation.receipt["stage_index"],
        "evaluation_stage_name": evaluation.receipt["stage_name"],
        "evaluation_domain": evaluation.evaluation_domain,
        "split": split,
        "token_file": str(evaluation.token_file_path),
        "token_file_sha256": sha256(evaluation.token_file_path),
        "token_count": len(evaluation.tokens),
        "scene_set_manifest": str(evaluation.receipt_path),
        "scene_set_manifest_sha256": sha256(evaluation.receipt_path),
        "csv": str(args.output_csv.resolve()),
        "csv_sha256": sha256(args.output_csv),
        "csv_row_count": len(evaluated_rows),
        "invalid_row_count": 0,
        "failure_count": 0,
        "metric_schema_sha256": metric_schema_sha256,
        "metric_registry": str(args.metric_registry.resolve()),
        "metric_registry_sha256": sha256(args.metric_registry),
        "domain_registry": str(evaluation.protocol_path),
        "domain_registry_sha256": sha256(evaluation.protocol_path),
        "evaluator_repository": str(evaluator_repository),
        "evaluator_source_commit": args.evaluator_source_commit,
        "evaluator_source_tree": evaluator_tree,
        "evaluator_entrypoint": str(args.evaluator_entrypoint.resolve()),
        "evaluator_entrypoint_sha256": sha256(args.evaluator_entrypoint),
        "evaluator_entrypoint_repository_path": entrypoint_relative,
        "command_spec": str(args.command_spec.resolve()),
        "command_spec_sha256": sha256(args.command_spec),
        "command_spec_repository_path": command_relative,
        "executed_argv": command,
        "environment_lock": str(args.environment_lock.resolve()),
        "environment_lock_sha256": sha256(args.environment_lock),
        "deterministic_seed_env": "PDM_DETERMINISTIC_GLOBAL_SEED",
        "deterministic_seed": args.deterministic_seed,
        "stdout_log": str(args.stdout_log.resolve()),
        "stdout_log_sha256": sha256(args.stdout_log),
        "stderr_log": str(args.stderr_log.resolve()),
        "stderr_log_sha256": sha256(args.stderr_log),
        "started_at_utc": started,
        "completed_at_utc": completed,
        "sealed_test_access": sealed_access,
    }
    atomic_json(args.output_receipt, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

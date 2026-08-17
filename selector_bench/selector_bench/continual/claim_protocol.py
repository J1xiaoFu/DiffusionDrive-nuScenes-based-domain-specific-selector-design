"""Fail-closed identities shared by Drive-CL claim-producing tools."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from selector_bench.continual.statistics import PDM_DEFAULT_METRICS, StatisticsError


TRAINING_PROTOCOL_SCHEMA = "selector_bench.drive_cl_diffusiondrive_run.v3"
TRAINING_RESULT_SCHEMA = "selector_bench.drive_cl_diffusiondrive_result.v3"
RUN_REGISTRATION_SCHEMA = "selector_bench.drive_cl_run_registration.v1"
EVALUATOR_RUN_SCHEMA = "selector_bench.drive_cl_evaluator_run.v2"
CELL_SUMMARY_SCHEMA = "selector_bench.drive_cl_pdm_cell_summary.v5"
CROSSED_SPEC_SCHEMA = "selector_bench.drive_cl_crossed_comparison_spec.v2"
CROSSED_RECEIPT_SCHEMA = "selector_bench.drive_cl_crossed_pdm_comparison.v4"
GLOBAL_FAMILY_SCHEMA = "selector_bench.drive_cl_global_holm_family.v4"
GLOBAL_RESULT_SCHEMA = "selector_bench.drive_cl_global_holm_result.v4"
SEALED_ACCESS_SCHEMA = "selector_bench.drive_cl_sealed_test_access.v1"
SEED_DESIGN_SCHEMA = "selector_bench.drive_cl_seed_design.v1"
EVIDENCE_INVENTORY_SCHEMA = "selector_bench.drive_cl_evidence_inventory.v1"
METHOD_REGISTRY_SCHEMA = "selector_bench.drive_cl_method_registry.v1"
METRIC_REGISTRY_SCHEMA = "selector_bench.drive_cl_metric_registry.v1"
CLAIM_RECEIPT_SUFFIX = ".crossed.json"


# Paper-facing method names are intentionally registered against one executable
# training arm.  A table label may not silently relabel another code path.
REGISTERED_METHOD_ARMS: Mapping[str, str] = {
    "sequential": "sequential",
    "planner_only": "planner_only",
    "ewc": "ewc",
    "agem": "agem",
    "aler_drive": "aler_drive",
    "step_matched_replay": "step_matched_replay",
    "full_exposure_replay": "full_exposure_replay",
    "lwf": "lwf",
    "drive_opd_fixed": "fixed_opd",
    "drive_opd_ema099": "ema099_opd",
    "drive_perception_only": "perception_only",
    "drive_planning_only": "planning_only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StatisticsError(f"missing or symlinked {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StatisticsError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise StatisticsError(f"{label} must be a JSON object: {path}")
    return value


def resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StatisticsError(f"{label} must be a non-empty path")
    raw = Path(value)
    path = (root / raw).absolute() if not raw.is_absolute() else raw.absolute()
    resolved = path.resolve()
    if path != resolved:
        raise StatisticsError(f"{label} may not traverse a symlink: {path}")
    return resolved


def require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StatisticsError(f"{label} is outside registered root {root}: {path}") from exc


def git(repository: Path, *arguments: str, allow_failure: bool = False) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0 and not allow_failure:
        detail = process.stderr.decode(errors="replace").strip()
        raise StatisticsError(
            f"Git verification failed: {' '.join(arguments)}: {detail}"
        )
    return process.stdout


def verify_frozen_file(
    repository: Path,
    path: Path,
    freeze_commit: str,
    *,
    require_ancestor_of_head: bool = True,
) -> str:
    """Verify that ``path`` is exactly one blob at a full immutable commit."""

    repository = repository.resolve()
    path = path.resolve()
    if not (repository / ".git").exists():
        raise StatisticsError(f"registered repository is not a Git worktree: {repository}")
    try:
        relative = path.relative_to(repository)
    except ValueError as exc:
        raise StatisticsError("frozen file must be inside the registered repository") from exc
    resolved_commit = git(
        repository, "rev-parse", f"{freeze_commit}^{{commit}}"
    ).decode().strip()
    if len(resolved_commit) != 40 or resolved_commit != freeze_commit:
        raise StatisticsError("freeze commit must be a full 40-character commit SHA")
    frozen = git(repository, "show", f"{resolved_commit}:{relative.as_posix()}")
    if frozen != path.read_bytes():
        raise StatisticsError("registered file differs from the immutable freeze commit")
    if require_ancestor_of_head:
        git(repository, "merge-base", "--is-ancestor", resolved_commit, "HEAD")
    return relative.as_posix()


def canonical_evaluation_domain(protocol: Mapping[str, Any], stage: Mapping[str, Any]) -> str:
    protocol_id = protocol.get("protocol_id")
    stage_index = stage.get("stage_index")
    stage_name = stage.get("name")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise StatisticsError("protocol_id must be a non-empty string")
    if not isinstance(stage_index, int) or not isinstance(stage_name, str) or not stage_name:
        raise StatisticsError("evaluation stage identity is malformed")
    dataset = protocol.get("dataset_identity")
    if not isinstance(dataset, Mapping):
        raise StatisticsError("protocol lacks a registered dataset identity")
    dataset_name = dataset.get("dataset")
    dataset_version = dataset.get("dataset_version")
    if (
        not isinstance(dataset_name, str)
        or not dataset_name
        or not isinstance(dataset_version, str)
        or not dataset_version
    ):
        raise StatisticsError("protocol dataset identity is malformed")
    return (
        f"{dataset_name}/{dataset_version}::{protocol_id}::"
        f"stage_{stage_index}::{stage_name}"
    )


def validate_method_registry(
    registry_path: str | Path,
    expected_sha256: object,
) -> dict[str, str]:
    path = Path(registry_path).resolve()
    require_equal(expected_sha256, sha256(path), "method registry SHA256")
    payload = load_json(path, "method registry")
    require_equal(payload.get("schema"), METHOD_REGISTRY_SCHEMA, "method registry schema")
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise StatisticsError("method registry must contain registered methods")
    observed: dict[str, str] = {}
    for item in methods:
        if not isinstance(item, dict) or set(item) != {"method_id", "training_arm"}:
            raise StatisticsError("method registry entry has invalid semantic fields")
        method_id = item.get("method_id")
        training_arm = item.get("training_arm")
        if (
            not isinstance(method_id, str)
            or not method_id
            or not isinstance(training_arm, str)
            or not training_arm
            or method_id in observed
            or training_arm in observed.values()
        ):
            raise StatisticsError("method registry identities must be non-empty and one-to-one")
        observed[method_id] = training_arm
    require_equal(observed, dict(REGISTERED_METHOD_ARMS), "method registry contents")
    return observed


def validate_metric_registry(
    registry_path: str | Path,
    expected_sha256: object,
) -> tuple[str, ...]:
    path = Path(registry_path).resolve()
    require_equal(expected_sha256, sha256(path), "metric registry SHA256")
    payload = load_json(path, "metric registry")
    require_equal(payload.get("schema"), METRIC_REGISTRY_SCHEMA, "metric registry schema")
    require_equal(tuple(payload.get("metrics", [])), PDM_DEFAULT_METRICS, "metric registry")
    return PDM_DEFAULT_METRICS


def validate_method_arm(
    method_id: object,
    training_arm: object,
    *,
    registry: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    if not isinstance(method_id, str) or not method_id:
        raise StatisticsError("method_id must be a non-empty registered name")
    if not isinstance(training_arm, str) or not training_arm:
        raise StatisticsError("training_arm must be a non-empty registered arm")
    expected = (registry or REGISTERED_METHOD_ARMS).get(method_id)
    if expected is None:
        raise StatisticsError(f"unregistered method_id: {method_id}")
    if training_arm != expected:
        raise StatisticsError(
            f"method {method_id} is registered to arm {expected}, not {training_arm}"
        )
    return method_id, training_arm


def validate_seed_design(
    family: dict[str, Any],
    *,
    family_root: Path,
    repository: Path,
    freeze_commit: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Verify an audit-only executable seed power/calibration receipt."""

    design_path = resolve(
        family_root, family.get("seed_design_receipt"), "seed-design receipt"
    )
    verify_frozen_file(repository, design_path, freeze_commit)
    design = load_json(design_path, "seed-design receipt")
    require_equal(design.get("schema"), SEED_DESIGN_SCHEMA, "seed-design schema")
    require_equal(design.get("status"), "passed", "seed-design status")
    require_equal(
        tuple(design.get("designed_seed_ids", [])), expected_seeds, "designed seed IDs"
    )
    require_equal(design.get("design_split"), "audit", "seed-design split")
    require_equal(
        family.get("seed_design_receipt_sha256"),
        sha256(design_path),
        "family seed-design receipt SHA256",
    )
    try:
        target_power = float(design.get("target_power"))
        achieved_power = float(design.get("estimated_power"))
        minimum_effect = float(design.get("minimum_relevant_effect"))
        repetitions = int(design.get("power_simulation_repetitions"))
    except (TypeError, ValueError) as exc:
        raise StatisticsError("seed-design numerical contract is malformed") from exc
    if target_power < 0.8 or achieved_power < target_power or minimum_effect <= 0.0:
        raise StatisticsError("seed design does not meet its preregistered power/effect target")
    family_alpha = float(family.get("alpha"))
    tail_resolution_minimum = math.ceil(
        1.0 - math.log2(family_alpha / len(PDM_DEFAULT_METRICS))
    )
    if len(expected_seeds) < tail_resolution_minimum:
        raise StatisticsError(
            "confirmatory seed set cannot resolve the two-sided seven-metric family tail: "
            f"need at least {tail_resolution_minimum}, got {len(expected_seeds)}"
        )
    require_equal(
        design.get("minimum_seed_count_for_two_sided_family_tail_resolution"),
        tail_resolution_minimum,
        "seed-design discrete tail-resolution minimum",
    )
    if repetitions < 10_000 or design.get("calibration_status") != "passed":
        raise StatisticsError("seed design lacks the required calibration audit")
    require_equal(
        design.get("calibration_method"),
        "heldout_null_crossed_resampling_max_statistic_over_seven_metrics",
        "seed-design calibration method",
    )
    require_equal(
        design.get("power_method"),
        "audit_only_crossed_resampling_at_minimum_relevant_effect",
        "seed-design power method",
    )
    if not isinstance(design.get("pilot_matrix_sha256"), str) or len(
        design["pilot_matrix_sha256"]
    ) != 64:
        raise StatisticsError("seed design lacks a content-addressed pilot matrix")
    pilot_path = resolve(family_root, design.get("pilot_matrix"), "seed-design pilot matrix")
    verify_frozen_file(repository, pilot_path, freeze_commit)
    require_equal(
        design.get("pilot_matrix_sha256"), sha256(pilot_path), "seed-design pilot matrix SHA256"
    )
    pilot = load_json(pilot_path, "seed-design pilot matrix")
    require_equal(
        pilot.get("schema"),
        "selector_bench.drive_cl_audit_pilot_matrix.v1",
        "seed-design pilot schema",
    )
    require_equal(pilot.get("split"), "audit", "seed-design pilot split")
    require_equal(tuple(pilot.get("metrics", [])), PDM_DEFAULT_METRICS, "seed-design pilot metrics")
    require_equal(
        design.get("family_alpha"), family_alpha, "seed-design family alpha"
    )
    require_equal(
        design.get("family_metric_count"),
        len(PDM_DEFAULT_METRICS),
        "seed-design family metric count",
    )
    if not isinstance(design.get("simulation_seed"), int):
        raise StatisticsError("seed design lacks an integer simulation seed")
    candidate_audits = design.get("candidate_seed_audits")
    if not isinstance(candidate_audits, list) or not candidate_audits:
        raise StatisticsError("seed design lacks executable candidate-seed audit rows")
    selected_rows = [
        row
        for row in candidate_audits
        if isinstance(row, dict) and row.get("seed_count") == len(expected_seeds)
    ]
    if len(selected_rows) != 1:
        raise StatisticsError("seed design does not contain one selected-seed audit row")
    selected = selected_rows[0]
    if float(selected.get("minimum_metric_power", -1.0)) != achieved_power:
        raise StatisticsError("seed-design power does not match its selected audit row")
    if float(selected.get("heldout_null_fwer", math.inf)) > float(
        selected.get("null_fwer_tolerance", -math.inf)
    ):
        raise StatisticsError("seed-design held-out null calibration failed")
    metric_power = selected.get("metric_power")
    if not isinstance(metric_power, dict) or set(metric_power) != set(PDM_DEFAULT_METRICS):
        raise StatisticsError("seed-design selected row lacks all seven metric powers")
    if min(float(value) for value in metric_power.values()) != achieved_power:
        raise StatisticsError("seed-design minimum metric power is not replayable")
    if (
        int(selected.get("calibration_simulations", 0))
        + int(selected.get("evaluation_simulations", 0))
        != repetitions
    ):
        raise StatisticsError("seed-design calibration/evaluation repetitions mismatch")
    sources = design.get("audit_source_receipt_sha256")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(value, str) or len(value) != 64 for value in sources
    ):
        raise StatisticsError("seed design lacks content-addressed audit-only pilot evidence")
    require_equal(
        sources, pilot.get("source_receipt_sha256"), "seed-design pilot source hashes"
    )
    source_paths = pilot.get("source_receipts")
    if not isinstance(source_paths, list) or len(source_paths) != len(sources):
        raise StatisticsError("seed-design pilot lacks source receipt paths")
    for source_value, expected_hash in zip(source_paths, sources):
        source_path = resolve(pilot_path.parent, source_value, "seed-design audit source receipt")
        verify_frozen_file(repository, source_path, freeze_commit)
        require_equal(expected_hash, sha256(source_path), "seed-design audit source SHA256")
    return {
        "seed_design_receipt": str(design_path),
        "seed_design_receipt_sha256": sha256(design_path),
        "target_power": target_power,
        "estimated_power": achieved_power,
        "minimum_relevant_effect": minimum_effect,
        "power_simulation_repetitions": repetitions,
        "heldout_null_fwer": float(selected["heldout_null_fwer"]),
        "null_fwer_tolerance": float(selected["null_fwer_tolerance"]),
        "minimum_seed_count_for_two_sided_family_tail_resolution": tail_resolution_minimum,
    }


def validate_family_registries_and_cells(
    family: dict[str, Any],
    *,
    family_path: Path,
    repository: Path,
    freeze_commit: str,
    expected_protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Validate immutable dataset/method/metric/domain and cross-cell registrations."""

    family_root = family_path.parent
    method_path = resolve(family_root, family.get("method_registry"), "family method registry")
    metric_path = resolve(family_root, family.get("metric_registry"), "family metric registry")
    domain_path = resolve(family_root, family.get("domain_registry"), "family domain registry")
    for path in (method_path, metric_path, domain_path):
        verify_frozen_file(repository, path, freeze_commit)
    methods = validate_method_registry(method_path, family.get("method_registry_sha256"))
    validate_metric_registry(metric_path, family.get("metric_registry_sha256"))
    require_equal(
        family.get("domain_registry_sha256"),
        sha256(domain_path),
        "family domain registry SHA256",
    )
    protocol = load_json(domain_path, "family domain registry")
    require_equal(
        family.get("dataset_identity"),
        protocol.get("dataset_identity"),
        "family dataset/domain identity",
    )
    if expected_protocol_path is not None:
        require_equal(domain_path, expected_protocol_path.resolve(), "family domain registry path")

    expected_comparisons = family.get("expected_comparisons")
    if not isinstance(expected_comparisons, list) or not expected_comparisons:
        raise StatisticsError("family must explicitly register expected comparisons")
    comparison_keys = {
        "comparison_id",
        "comparison_spec_repository_path",
        "comparison_spec_sha256",
        "comparison_receipt_repository_path",
        "checkpoint_stage_index",
        "evaluation_stage_index",
        "evaluation_domain",
        "split",
        "baseline_method",
        "baseline_training_arm",
        "candidate_method",
        "candidate_training_arm",
    }
    comparison_by_id: dict[str, dict[str, Any]] = {}
    registered_cells: set[tuple[object, ...]] = set()
    for item in expected_comparisons:
        if not isinstance(item, dict) or set(item) != comparison_keys:
            raise StatisticsError("family expected-comparison fields are malformed")
        comparison_id = item.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id or comparison_id in comparison_by_id:
            raise StatisticsError("family comparison IDs must be non-empty and unique")
        validate_method_arm(
            item.get("baseline_method"), item.get("baseline_training_arm"), registry=methods
        )
        validate_method_arm(
            item.get("candidate_method"), item.get("candidate_training_arm"), registry=methods
        )
        spec_path = resolve(repository, item.get("comparison_spec_repository_path"), "family comparison spec")
        verify_frozen_file(repository, spec_path, freeze_commit)
        require_equal(item.get("comparison_spec_sha256"), sha256(spec_path), "family comparison spec SHA256")
        receipt_path = resolve(
            repository,
            item.get("comparison_receipt_repository_path"),
            "family comparison receipt",
        )
        require_within(receipt_path, repository, "family comparison receipt")
        checkpoint_stage = next(
            (
                stage
                for stage in protocol.get("stages", [])
                if isinstance(stage, dict)
                and stage.get("stage_index") == item.get("checkpoint_stage_index")
            ),
            None,
        )
        evaluation_stage = next(
            (
                stage
                for stage in protocol.get("stages", [])
                if isinstance(stage, dict)
                and stage.get("stage_index") == item.get("evaluation_stage_index")
            ),
            None,
        )
        if not isinstance(checkpoint_stage, dict) or not isinstance(evaluation_stage, dict):
            raise StatisticsError("family comparison references an unknown protocol stage")
        require_equal(
            item.get("evaluation_domain"),
            canonical_evaluation_domain(protocol, evaluation_stage),
            "family comparison evaluation domain",
        )
        split = item.get("split")
        if split not in {"audit", "test"} or split not in evaluation_stage.get("splits", {}):
            raise StatisticsError("family comparison references an unknown evaluation split")
        cell = (
            item.get("checkpoint_stage_index"),
            item.get("evaluation_stage_index"),
            item.get("evaluation_domain"),
            item.get("split"),
        )
        if cell in registered_cells:
            raise StatisticsError("family reuses a checkpoint-stage/evaluation-domain cell")
        registered_cells.add(cell)
        comparison_by_id[comparison_id] = item

    required_cells = family.get("required_cross_cells")
    cell_keys = {
        "checkpoint_stage_index",
        "evaluation_stage_index",
        "evaluation_domain",
        "split",
    }
    if not isinstance(required_cells, list) or not required_cells:
        raise StatisticsError("family must explicitly register required cross cells")
    required_coordinates: set[tuple[object, ...]] = set()
    for cell in required_cells:
        if not isinstance(cell, dict) or set(cell) != cell_keys:
            raise StatisticsError("family required-cross-cell fields are malformed")
        coordinate = tuple(cell[field] for field in (
            "checkpoint_stage_index",
            "evaluation_stage_index",
            "evaluation_domain",
            "split",
        ))
        if coordinate in required_coordinates:
            raise StatisticsError("family has a duplicate required cross cell")
        required_coordinates.add(coordinate)
    if not required_coordinates.issubset(registered_cells):
        raise StatisticsError("family is missing a required checkpoint-stage/evaluation-domain cell")
    if not any(checkpoint != evaluation for checkpoint, evaluation, _, _ in required_coordinates):
        raise StatisticsError("family must preregister at least one off-diagonal continual-learning cell")
    hypotheses = family.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise StatisticsError("family must preregister claim hypotheses")
    metrics_by_comparison: dict[str, list[str]] = {
        comparison_id: [] for comparison_id in comparison_by_id
    }
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            raise StatisticsError("family hypothesis is malformed")
        comparison_id = hypothesis.get("comparison_id")
        if comparison_id not in metrics_by_comparison:
            raise StatisticsError("family hypothesis references an unexpected comparison")
        metrics_by_comparison[str(comparison_id)].append(str(hypothesis.get("metric")))
    for comparison_id, metrics in metrics_by_comparison.items():
        if len(metrics) != len(PDM_DEFAULT_METRICS) or set(metrics) != set(PDM_DEFAULT_METRICS):
            raise StatisticsError(
                f"family comparison {comparison_id} must register all seven metrics exactly once"
            )
    return {
        "method_registry": str(method_path),
        "method_registry_sha256": sha256(method_path),
        "metric_registry": str(metric_path),
        "metric_registry_sha256": sha256(metric_path),
        "domain_registry": str(domain_path),
        "domain_registry_sha256": sha256(domain_path),
        "dataset_identity": protocol.get("dataset_identity"),
        "expected_comparisons": comparison_by_id,
        "required_cross_cells": required_coordinates,
    }


def comparison_inventory_record(
    receipt: dict[str, Any], receipt_path: Path, repository: Path
) -> dict[str, Any]:
    """Return and rehash the complete claim-bearing evidence bundle for one comparison."""

    result: dict[str, Any] = {
        "comparison_id": receipt.get("comparison_id"),
        "comparison_receipt_repository_path": receipt_path.resolve()
        .relative_to(repository.resolve())
        .as_posix(),
        "comparison_receipt_sha256": sha256(receipt_path),
        "comparison_spec_repository_path": Path(receipt.get("comparison_spec", ""))
        .resolve()
        .relative_to(repository.resolve())
        .as_posix(),
        "comparison_spec_sha256": receipt.get("comparison_spec_sha256"),
        "baseline": {},
        "candidate": {},
    }
    for side in ("baseline", "candidate"):
        method = receipt.get(side)
        if not isinstance(method, dict):
            raise StatisticsError(f"comparison receipt lacks {side} evidence")
        side_record = {
            "method_id": method.get("method_id"),
            "training_arm": method.get("training_arm"),
            "runs": [],
        }
        runs = method.get("runs")
        if not isinstance(runs, list) or not runs:
            raise StatisticsError(f"comparison receipt lacks {side} runs")
        for run in sorted(runs, key=lambda value: int(value.get("training_seed", -1))):
            if not isinstance(run, dict):
                raise StatisticsError("comparison receipt has malformed run evidence")
            record: dict[str, Any] = {
                "training_seed": run.get("training_seed"),
                "run_id": run.get("run_id"),
                "optimizer_state_sha256": run.get("optimizer_state_sha256"),
            }
            for prefix in (
                "run_registration",
                "training_protocol",
                "training_result",
                "cell_summary",
                "evaluator_receipt",
                "source_checkpoint",
                "rng_record",
                "optimizer_state_receipt",
                "checkpoint",
                "csv",
            ):
                path = resolve(receipt_path.parent, run.get(f"{prefix}_path"), f"inventory {prefix}")
                digest = sha256(path)
                require_equal(run.get(f"{prefix}_sha256"), digest, f"inventory {prefix} SHA256")
                try:
                    registered_path = path.relative_to(repository).as_posix()
                except ValueError:
                    registered_path = str(path)
                record[f"{prefix}_path"] = registered_path
                record[f"{prefix}_sha256"] = digest
            record["run_registration_commit"] = run.get("run_registration_commit")
            side_record["runs"].append(record)
        result[side] = side_record
    return result


def stable_run_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that a completed result must copy exactly from its run protocol."""

    fields = (
        "run_id",
        "method_id",
        "arm",
        "seed",
        "stage_index",
        "dataset_identity",
        "source_code",
        "protocol_manifest_sha256",
        "protocol_content_sha256",
        "experiment_config_sha256",
        "method_registry_sha256",
        "environment_lock_sha256",
        "rng_record_sha256",
        "run_registration_sha256",
        "run_registration_commit",
        "source_checkpoint_sha256",
        "stage_start_state_sha256",
        "start_epoch",
        "end_epoch",
        "max_steps",
        "target_budget",
    )
    return {field: protocol.get(field) for field in fields}


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise StatisticsError(
            f"{label} mismatch: expected={expected!r} actual={actual!r}"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


RUN_BUDGET_FIELDS = (
    "epochs",
    "optimizer_updates",
    "current_unique_identities",
    "old_unique_identities",
    "current_presentations",
    "old_presentations",
    "forward_calls",
    "backward_calls",
    "student_queries",
    "teacher_queries",
)


def _validate_budget(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(RUN_BUDGET_FIELDS):
        raise StatisticsError(f"{label} must contain the complete frozen budget")
    result: dict[str, int] = {}
    for field in RUN_BUDGET_FIELDS:
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise StatisticsError(f"{label} has an invalid {field}")
        result[field] = item
    if result["epochs"] <= 0 or result["optimizer_updates"] <= 0:
        raise StatisticsError(f"{label} must contain positive epochs and optimizer updates")
    return result


@dataclass(frozen=True)
class TrainingRunContract:
    protocol_path: Path
    protocol: dict[str, Any]
    protocol_sha256: str
    result_path: Path
    result: dict[str, Any]
    result_sha256: str
    endpoint_path: Path
    endpoint_sha256: str


def load_training_run_contract(
    protocol_path: str | Path, result_path: str | Path
) -> TrainingRunContract:
    protocol_path = Path(protocol_path).resolve()
    result_path = Path(result_path).resolve()
    protocol = load_json(protocol_path, "training protocol")
    result = load_json(result_path, "training result")
    require_equal(protocol.get("schema"), TRAINING_PROTOCOL_SCHEMA, "training protocol schema")
    require_equal(result.get("schema"), TRAINING_RESULT_SCHEMA, "training result schema")
    run_id = protocol.get("run_id")
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise StatisticsError("training run_id must be 32 lowercase hexadecimal characters")
    registered_protocol_path = resolve(
        result_path.parent, result.get("training_protocol"), "result training protocol"
    )
    require_equal(registered_protocol_path, protocol_path, "result training protocol path")
    protocol_hash = sha256(protocol_path)
    require_equal(
        result.get("training_protocol_sha256"),
        protocol_hash,
        "result training protocol SHA256",
    )
    for field, expected in stable_run_identity(protocol).items():
        require_equal(result.get(field), expected, f"result {field}")
    registration_repository = resolve(
        protocol_path.parent,
        protocol.get("run_registration_repository"),
        "run-registration repository",
    )
    registration_path = resolve(
        protocol_path.parent, protocol.get("run_registration"), "run registration"
    )
    registration_commit = protocol.get("run_registration_commit")
    if not isinstance(registration_commit, str):
        raise StatisticsError("run registration lacks a full freeze commit")
    verify_frozen_file(registration_repository, registration_path, registration_commit)
    require_equal(
        protocol.get("run_registration_sha256"),
        sha256(registration_path),
        "run-registration SHA256",
    )
    registration = load_json(registration_path, "run registration")
    require_equal(
        registration.get("schema"), RUN_REGISTRATION_SCHEMA, "run-registration schema"
    )
    registration_fields = (
        "run_id",
        "method_id",
        "arm",
        "seed",
        "stage_index",
        "dataset_identity",
        "source_code",
        "protocol_manifest_sha256",
        "protocol_content_sha256",
        "experiment_config_sha256",
        "method_registry_sha256",
        "environment_lock_sha256",
        "source_checkpoint_sha256",
        "stage_start_state_sha256",
        "start_epoch",
        "end_epoch",
        "max_steps",
        "target_budget",
    )
    for field in registration_fields:
        require_equal(protocol.get(field), registration.get(field), f"registered run {field}")
    for label, field in (
        ("training protocol manifest", "protocol_manifest"),
        ("training experiment config", "experiment_config"),
        ("training method registry", "method_registry"),
        ("training environment lock", "environment_lock"),
        ("training RNG record", "rng_record"),
    ):
        artifact = resolve(protocol_path.parent, protocol.get(field), label)
        if not artifact.is_file():
            raise StatisticsError(f"missing {label}: {artifact}")
        require_equal(
            protocol.get(f"{field}_sha256"), sha256(artifact), f"{label} SHA256"
        )
    method_registry_path = resolve(
        protocol_path.parent, protocol.get("method_registry"), "training method registry"
    )
    method_registry = validate_method_registry(
        method_registry_path, protocol.get("method_registry_sha256")
    )
    validate_method_arm(
        protocol.get("method_id"), protocol.get("arm"), registry=method_registry
    )
    dataset = protocol.get("dataset_identity")
    if not isinstance(dataset, dict) or set(dataset) != {
        "dataset",
        "dataset_version",
        "dataset_root_metadata_sha256",
    }:
        raise StatisticsError("training dataset identity is malformed")
    for field in ("dataset", "dataset_version"):
        if not isinstance(dataset.get(field), str) or not dataset[field]:
            raise StatisticsError(f"training dataset identity lacks {field}")
    if not _is_sha256(dataset.get("dataset_root_metadata_sha256")):
        raise StatisticsError("training dataset root metadata SHA256 is malformed")
    manifest_path = resolve(
        protocol_path.parent, protocol.get("protocol_manifest"), "training protocol manifest"
    )
    manifest = load_json(manifest_path, "training protocol manifest")
    require_equal(
        manifest.get("dataset_identity"), dataset, "training dataset/protocol identity"
    )
    source_code = protocol.get("source_code")
    if not isinstance(source_code, dict) or set(source_code) != {
        "repository",
        "commit",
        "tree",
        "dirty",
    }:
        raise StatisticsError("training source-code identity is malformed")
    if (
        not isinstance(source_code.get("repository"), str)
        or not source_code["repository"]
        or not _is_git_sha(source_code.get("commit"))
        or not _is_git_sha(source_code.get("tree"))
        or source_code.get("dirty") is not False
    ):
        raise StatisticsError("claim-eligible training requires a clean full source commit/tree")
    source_repository = resolve(
        protocol_path.parent, source_code.get("repository"), "training source repository"
    )
    if not (source_repository / ".git").exists():
        raise StatisticsError("training source repository is not a Git worktree")
    resolved_source_commit = git(
        source_repository, "rev-parse", f"{source_code['commit']}^{{commit}}"
    ).decode().strip()
    require_equal(
        resolved_source_commit, source_code["commit"], "training source commit"
    )
    resolved_source_tree = git(
        source_repository, "rev-parse", f"{source_code['commit']}^{{tree}}"
    ).decode().strip()
    require_equal(resolved_source_tree, source_code["tree"], "training source tree")
    rng_path = resolve(protocol_path.parent, protocol.get("rng_record"), "training RNG record")
    rng = load_json(rng_path, "training RNG record")
    require_equal(
        rng.get("schema"), "selector_bench.drive_cl_rng_record.v1", "RNG-record schema"
    )
    require_equal(rng.get("run_id"), run_id, "RNG run ID")
    require_equal(rng.get("seed"), protocol.get("seed"), "RNG seed")
    for field in ("python_seed", "numpy_seed", "torch_seed"):
        require_equal(rng.get(field), protocol.get("seed"), f"RNG {field}")
    if not _is_sha256(protocol.get("rng_record_sha256")):
        raise StatisticsError("training RNG-record SHA256 is malformed")
    start_epoch = protocol.get("start_epoch")
    end_epoch = protocol.get("end_epoch")
    if not isinstance(start_epoch, int) or not isinstance(end_epoch, int) or end_epoch < start_epoch:
        raise StatisticsError("training protocol has an invalid epoch completion target")
    require_equal(result.get("completed_epoch"), end_epoch, "completed epoch target")
    require_equal(result.get("stopped_early"), False, "result completion")
    require_equal(protocol.get("max_steps"), 0, "claim-eligible max_steps")
    target_budget = _validate_budget(protocol.get("target_budget"), "target budget")
    observed_budget = _validate_budget(result.get("observed_budget"), "observed budget")
    require_equal(observed_budget, target_budget, "completed run budget")
    require_equal(
        target_budget["epochs"],
        end_epoch - start_epoch + 1,
        "registered epoch budget",
    )
    source_value = protocol.get("source_checkpoint")
    source_digest = protocol.get("source_checkpoint_sha256")
    stage_index = protocol.get("stage_index")
    if stage_index != 1 or source_value is not None or source_digest is not None:
        source_path = resolve(
            protocol_path.parent, source_value, "training source checkpoint"
        )
        if not source_path.is_file():
            raise StatisticsError(f"missing training source checkpoint: {source_path}")
        require_equal(
            source_digest,
            sha256(source_path),
            "training source checkpoint SHA256",
        )
        require_equal(
            protocol.get("stage_start_state_sha256"),
            source_digest,
            "training stage-start state SHA256",
        )
    global_step = result.get("global_step")
    invocation_steps = result.get("steps_this_invocation")
    if (
        not isinstance(global_step, int)
        or not isinstance(invocation_steps, int)
        or global_step <= 0
        or invocation_steps <= 0
        or global_step < invocation_steps
    ):
        raise StatisticsError("completed training result has invalid optimizer-step counts")
    require_equal(
        invocation_steps,
        observed_budget["optimizer_updates"],
        "optimizer update count",
    )
    endpoint_path = resolve(result_path.parent, result.get("endpoint"), "training endpoint")
    if not endpoint_path.is_file():
        raise StatisticsError(f"missing training endpoint: {endpoint_path}")
    endpoint_hash = sha256(endpoint_path)
    require_equal(result.get("endpoint_sha256"), endpoint_hash, "training endpoint SHA256")
    require_equal(
        result.get("parent_checkpoint_sha256"),
        protocol.get("source_checkpoint_sha256"),
        "result parent checkpoint identity",
    )
    if not _is_sha256(result.get("optimizer_state_sha256")):
        raise StatisticsError("result optimizer-state SHA256 is malformed")
    optimizer_receipt_path = resolve(
        result_path.parent,
        result.get("optimizer_state_receipt"),
        "optimizer-state receipt",
    )
    require_equal(
        result.get("optimizer_state_receipt_sha256"),
        sha256(optimizer_receipt_path),
        "optimizer-state receipt SHA256",
    )
    optimizer_receipt = load_json(optimizer_receipt_path, "optimizer-state receipt")
    require_equal(
        optimizer_receipt.get("schema"),
        "selector_bench.drive_cl_optimizer_state_receipt.v1",
        "optimizer-state receipt schema",
    )
    require_equal(optimizer_receipt.get("run_id"), run_id, "optimizer-state run ID")
    require_equal(
        optimizer_receipt.get("optimizer_state_sha256"),
        result.get("optimizer_state_sha256"),
        "optimizer-state content SHA256",
    )
    require_equal(
        optimizer_receipt.get("endpoint_sha256"),
        endpoint_hash,
        "optimizer-state endpoint SHA256",
    )
    resources = result.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
        "wall_seconds",
        "peak_vram_bytes",
        "persistent_bytes",
        "transient_bytes",
        "host",
        "execution_device",
        "gpu_uuid",
    }:
        raise StatisticsError("training resource receipt is malformed")
    if (
        float(resources.get("wall_seconds", -1.0)) < 0.0
        or int(resources.get("peak_vram_bytes", -1)) < 0
        or int(resources.get("persistent_bytes", -1)) < 0
        or int(resources.get("transient_bytes", -1)) < 0
        or not isinstance(resources.get("host"), str)
        or resources.get("execution_device") not in {"cpu_fixture", "cuda"}
    ):
        raise StatisticsError("training resource receipt contains invalid values")
    if resources["execution_device"] == "cuda" and not resources.get("gpu_uuid"):
        raise StatisticsError("CUDA training receipt lacks a GPU UUID")
    return TrainingRunContract(
        protocol_path=protocol_path,
        protocol=protocol,
        protocol_sha256=protocol_hash,
        result_path=result_path,
        result=result,
        result_sha256=sha256(result_path),
        endpoint_path=endpoint_path,
        endpoint_sha256=endpoint_hash,
    )


@dataclass(frozen=True)
class EvaluatorRunContract:
    receipt_path: Path
    receipt: dict[str, Any]
    receipt_sha256: str
    csv_path: Path
    csv_sha256: str


def load_evaluator_run_contract(
    receipt_path: str | Path,
    *,
    training: TrainingRunContract,
    evaluation_cell_receipt: Path,
    evaluation_cell_receipt_sha256: str,
    evaluation_stage_index: int,
    evaluation_stage_name: str,
    evaluation_domain: str,
    split: str,
    token_file_path: Path,
    token_file_sha256: str,
) -> EvaluatorRunContract:
    receipt_path = Path(receipt_path).resolve()
    receipt = load_json(receipt_path, "evaluator-run receipt")
    require_equal(receipt.get("schema"), EVALUATOR_RUN_SCHEMA, "evaluator-run schema")
    require_equal(receipt.get("status"), "completed", "evaluator-run status")
    require_equal(receipt.get("return_code"), 0, "evaluator return code")
    bindings = {
        "run_id": training.protocol["run_id"],
        "training_arm": training.protocol["arm"],
        "training_seed": training.protocol["seed"],
        "checkpoint_stage_index": training.protocol["stage_index"],
        "dataset_identity": training.protocol["dataset_identity"],
        "training_protocol_sha256": training.protocol_sha256,
        "training_result_sha256": training.result_sha256,
        "checkpoint_sha256": training.endpoint_sha256,
        "evaluation_cell_receipt_sha256": evaluation_cell_receipt_sha256,
        "token_file_sha256": token_file_sha256,
        "evaluation_stage_index": evaluation_stage_index,
        "evaluation_stage_name": evaluation_stage_name,
        "evaluation_domain": evaluation_domain,
        "split": split,
    }
    for field, expected in bindings.items():
        require_equal(receipt.get(field), expected, f"evaluator {field}")
    path_bindings = (
        ("training_protocol", training.protocol_path),
        ("training_result", training.result_path),
        ("checkpoint", training.endpoint_path),
        ("evaluation_cell_receipt", evaluation_cell_receipt),
        ("token_file", token_file_path),
    )
    for field, expected in path_bindings:
        actual = resolve(receipt_path.parent, receipt.get(field), f"evaluator {field}")
        require_equal(actual, expected, f"evaluator {field} path")
    csv_path = resolve(receipt_path.parent, receipt.get("csv"), "evaluator CSV")
    if not csv_path.is_file():
        raise StatisticsError(f"missing evaluator CSV: {csv_path}")
    csv_hash = sha256(csv_path)
    require_equal(receipt.get("csv_sha256"), csv_hash, "evaluator CSV SHA256")
    metric_registry_path = resolve(
        receipt_path.parent, receipt.get("metric_registry"), "evaluator metric registry"
    )
    validate_metric_registry(
        metric_registry_path, receipt.get("metric_registry_sha256")
    )
    domain_registry_path = resolve(
        receipt_path.parent, receipt.get("domain_registry"), "evaluator domain registry"
    )
    evaluation_payload = load_json(evaluation_cell_receipt, "evaluation cell receipt")
    registered_domain_registry = resolve(
        evaluation_cell_receipt.parent,
        evaluation_payload.get("protocol_manifest"),
        "evaluation domain registry",
    )
    require_equal(
        domain_registry_path,
        registered_domain_registry,
        "evaluator domain registry path",
    )
    require_equal(
        receipt.get("domain_registry_sha256"),
        sha256(domain_registry_path),
        "evaluator domain registry SHA256",
    )
    metric_schema_sha256 = hashlib.sha256(
        json.dumps(list(PDM_DEFAULT_METRICS), separators=(",", ":")).encode()
    ).hexdigest()
    require_equal(
        receipt.get("metric_schema_sha256"), metric_schema_sha256, "evaluator metric schema"
    )
    if receipt.get("deterministic_seed_env") != "PDM_DETERMINISTIC_GLOBAL_SEED":
        raise StatisticsError("evaluator deterministic seed environment mismatch")
    if not isinstance(receipt.get("deterministic_seed"), int):
        raise StatisticsError("evaluator deterministic seed must be an integer")
    require_equal(
        receipt.get("deterministic_seed"),
        training.protocol["seed"],
        "evaluator deterministic seed/training seed",
    )
    command = receipt.get("executed_argv")
    if not isinstance(command, list) or not command or any(
        not isinstance(value, str) or not value for value in command
    ):
        raise StatisticsError("evaluator receipt lacks an exact argv vector")
    source_commit = receipt.get("evaluator_source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise StatisticsError("evaluator source commit must be a full SHA")
    source_tree = receipt.get("evaluator_source_tree")
    if not _is_git_sha(source_tree):
        raise StatisticsError("evaluator source tree must be a full SHA")
    artifact_paths: dict[str, Path] = {}
    for field in ("command_spec", "evaluator_entrypoint", "environment_lock", "stdout_log", "stderr_log"):
        artifact = resolve(receipt_path.parent, receipt.get(field), f"evaluator {field}")
        if not artifact.is_file():
            raise StatisticsError(f"missing evaluator {field}: {artifact}")
        require_equal(receipt.get(f"{field}_sha256"), sha256(artifact), f"evaluator {field} SHA256")
        artifact_paths[field] = artifact
    evaluator_repository = resolve(
        receipt_path.parent, receipt.get("evaluator_repository"), "evaluator repository"
    )
    verify_frozen_file(
        evaluator_repository, artifact_paths["command_spec"], source_commit
    )
    verify_frozen_file(
        evaluator_repository, artifact_paths["evaluator_entrypoint"], source_commit
    )
    require_equal(
        git(evaluator_repository, "rev-parse", f"{source_commit}^{{tree}}").decode().strip(),
        source_tree,
        "evaluator source tree",
    )
    joined_argv = "\0".join(command)
    for label, required_path in (
        ("checkpoint", training.endpoint_path),
        ("token file", token_file_path),
        ("CSV", csv_path),
        ("evaluator entrypoint", artifact_paths["evaluator_entrypoint"]),
    ):
        if joined_argv.count(str(required_path)) != 1:
            raise StatisticsError(
                f"evaluator argv must bind the exact {label} path exactly once"
            )
    sealed_access = receipt.get("sealed_test_access")
    if split == "test":
        if not isinstance(sealed_access, dict) or sealed_access.get("status") != "verified":
            raise StatisticsError("sealed test evaluation lacks a verified access receipt")
    elif sealed_access is not None:
        raise StatisticsError("audit evaluation may not claim sealed-test access")
    token_count = sum(1 for line in token_file_path.read_text().splitlines() if line)
    require_equal(receipt.get("token_count"), token_count, "evaluator token count")
    require_equal(receipt.get("csv_row_count"), token_count, "evaluator CSV row count")
    require_equal(receipt.get("invalid_row_count"), 0, "evaluator invalid-row count")
    require_equal(receipt.get("failure_count"), 0, "evaluator failure count")
    scene_manifest = resolve(
        receipt_path.parent, receipt.get("scene_set_manifest"), "evaluator scene-set manifest"
    )
    require_equal(scene_manifest, evaluation_cell_receipt, "evaluator scene-set manifest path")
    require_equal(
        receipt.get("scene_set_manifest_sha256"),
        evaluation_cell_receipt_sha256,
        "evaluator scene-set manifest SHA256",
    )
    return EvaluatorRunContract(
        receipt_path=receipt_path,
        receipt=receipt,
        receipt_sha256=sha256(receipt_path),
        csv_path=csv_path,
        csv_sha256=csv_hash,
    )

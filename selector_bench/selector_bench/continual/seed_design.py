"""Deterministic family-calibrated crossed seed design shared by CLI and claim gate."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from selector_bench.continual.statistics import (
    PDM_DEFAULT_METRICS,
    StatisticsError,
    crossed_matrix_bootstrap_inference,
    crossed_matrix_bootstrap_shift_pvalues_batch,
    holm_rejection_matrix,
)


PILOT_SCHEMA = "selector_bench.drive_cl_joint_audit_pilot_matrix.v2"
SEED_DESIGN_SCHEMA = "selector_bench.drive_cl_seed_design.v4"
INVOCATION_SCHEMA = "selector_bench.drive_cl_seed_design_invocation.v3"
PRODUCTION_POWER_SIMULATION_MINIMUM = 1_000
SYNTHETIC_FIXTURE_SIMULATION_MINIMUM = 4
# Deliberately process-local and one-entry only. Disk receipts/caches are never
# consulted because content addressing does not authenticate who computed them.
_IN_PROCESS_REPLAY_MEMO: dict[str, bytes] = {}


def _memo_lookup(key: str) -> bytes | None:
    cached = _IN_PROCESS_REPLAY_MEMO.get(key)
    if cached is None:
        _IN_PROCESS_REPLAY_MEMO.clear()
    return cached


def _memo_store(key: str, payload: bytes) -> None:
    _IN_PROCESS_REPLAY_MEMO.clear()
    _IN_PROCESS_REPLAY_MEMO[key] = payload

COMPARISON_IDENTITY_FIELDS = (
    "comparison_id",
    "baseline_method",
    "baseline_training_arm",
    "candidate_method",
    "candidate_training_arm",
)
RAW_PVALUE_METHOD = "null_centered_studentized_crossed_seed_session_bootstrap"
RESAMPLING_CONVENTION = (
    "independent_crossed_training_seed_and_evaluation_session_with_metric_seed_offset"
)
HYPOTHESIS_IDENTITY_FIELDS = (
    "id",
    "comparison_id",
    "metric",
    "baseline_method",
    "baseline_training_arm",
    "candidate_method",
    "candidate_training_arm",
)


def finite_real(value: object, label: str) -> float:
    """Return one finite real number without accepting bool/string coercions."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise StatisticsError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise StatisticsError(f"{label} must be a finite real number")
    return result


def finite_integer(value: object, label: str) -> int:
    """Return one integer while rejecting booleans and lossy numeric coercion."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StatisticsError(f"{label} must be an integer")
    return int(value)


def require_finite_array(value: object, label: str) -> np.ndarray:
    """Parse a numeric array and reject booleans, strings and non-finite cells."""

    raw = np.asarray(value, dtype=object)
    for cell in raw.flat:
        finite_real(cell, f"{label} cell")
    matrix = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise StatisticsError(f"{label} contains non-finite values")
    return matrix


def require_finite_tree(value: object, label: str) -> None:
    """Reject non-finite numeric leaves in a replay/input/output payload."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, Real):
        finite_real(value, label)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _seed_design_memo_key(material: Mapping[str, Any]) -> str:
    """Immutable key for post-computation in-process replay reuse."""

    require_finite_tree(material, "seed-design replay-memo material")
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def seed_design_replay_identity(
    *,
    normalized_invocation_payload: Mapping[str, Any],
    pilot_reference: str,
    generator_module_repository_path: str,
    generator_module_sha256: str,
    generator_entrypoint_repository_path: str,
    generator_entrypoint_sha256: str,
    inference_module_repository_path: str,
    inference_module_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Bind every input/program identity used by safe in-process memoization."""

    material = {
        "schema": "selector_bench.seed_design_replay_memo_key.v1",
        "seed_design_schema": SEED_DESIGN_SCHEMA,
        "numpy_version": np.__version__,
        "normalized_invocation": dict(normalized_invocation_payload),
        "pilot_reference": pilot_reference,
        "generator_module_repository_path": generator_module_repository_path,
        "generator_module_sha256": _sha256_value(
            generator_module_sha256, "generator module SHA256"
        ),
        "generator_entrypoint_repository_path": generator_entrypoint_repository_path,
        "generator_entrypoint_sha256": _sha256_value(
            generator_entrypoint_sha256, "generator entrypoint SHA256"
        ),
        "inference_module_repository_path": inference_module_repository_path,
        "inference_module_sha256": _sha256_value(
            inference_module_sha256, "inference module SHA256"
        ),
    }
    return _seed_design_memo_key(material), material


def _sha256_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StatisticsError(f"{label} must be a lowercase SHA256")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StatisticsError(f"missing or symlinked {label}: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StatisticsError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise StatisticsError(f"{label} must be a JSON object")
    require_finite_tree(payload, label)
    return payload


def family_hypothesis_contract(
    family: Mapping[str, Any], *, repository: Path
) -> dict[str, Any]:
    """Return the exact ordered comparison/hypothesis family used by Holm."""

    family_id = family.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise StatisticsError("seed-design family ID must be non-empty")
    dataset_identity = family.get("dataset_identity")
    if not isinstance(dataset_identity, Mapping):
        raise StatisticsError("seed-design family dataset identity must be an object")
    required_dataset_fields = {
        "dataset",
        "dataset_version",
        "dataset_root_metadata_sha256",
    }
    if set(dataset_identity) != required_dataset_fields or any(
        not isinstance(dataset_identity.get(field), str)
        or not dataset_identity.get(field)
        for field in required_dataset_fields
    ):
        raise StatisticsError("seed-design family dataset identity is malformed")
    _sha256_value(
        dataset_identity["dataset_root_metadata_sha256"],
        "seed-design dataset metadata SHA256",
    )
    alpha = finite_real(family.get("alpha"), "seed-design family alpha")
    if not 0.0 < alpha < 1.0:
        raise StatisticsError("seed-design family alpha must lie in (0, 1)")
    expected_seed_values = family.get("expected_seed_ids")
    if not isinstance(expected_seed_values, list):
        raise StatisticsError("seed-design family expected seeds must be a list")
    expected_seed_ids = [
        finite_integer(value, "seed-design family expected seed ID")
        for value in expected_seed_values
    ]
    if len(expected_seed_ids) < 2 or len(expected_seed_ids) != len(
        set(expected_seed_ids)
    ):
        raise StatisticsError("seed-design family needs at least two unique seeds")

    raw_comparisons = family.get("expected_comparisons")
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) < 2:
        raise StatisticsError("seed-design family needs at least two comparisons")
    comparisons: list[dict[str, str]] = []
    comparison_by_id: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_comparisons):
        if not isinstance(raw, Mapping):
            raise StatisticsError(f"seed-design comparison {index} is malformed")
        comparison = {field: raw.get(field) for field in COMPARISON_IDENTITY_FIELDS}
        if any(not isinstance(value, str) or not value for value in comparison.values()):
            raise StatisticsError(f"seed-design comparison {index} identity is malformed")
        comparison_id = comparison["comparison_id"]
        if comparison_id in comparison_by_id:
            raise StatisticsError("seed-design family comparison IDs must be unique")
        spec_repository_path = raw.get("comparison_spec_repository_path")
        spec_digest = raw.get("comparison_spec_sha256")
        if not isinstance(spec_repository_path, str) or not spec_repository_path:
            raise StatisticsError("seed-design comparison lacks its frozen spec path")
        _sha256_value(spec_digest, "seed-design comparison spec SHA256")
        spec_path = (repository.resolve() / spec_repository_path).resolve()
        try:
            spec_path.relative_to(repository.resolve())
        except ValueError as exc:
            raise StatisticsError("seed-design comparison spec escapes repository") from exc
        spec = _load_json(spec_path, "seed-design comparison specification")
        if sha256(spec_path) != spec_digest:
            raise StatisticsError("seed-design comparison spec SHA256 mismatch")
        if (
            spec.get("schema") != "selector_bench.drive_cl_crossed_comparison_spec.v2"
            or spec.get("comparison_id") != comparison_id
            or spec.get("analysis_scope") != "confirmatory"
            or spec.get("metric_family") != list(PDM_DEFAULT_METRICS)
            or spec.get("expected_seed_ids") != expected_seed_ids
        ):
            raise StatisticsError("seed-design comparison specification contract mismatch")
        for side, method_field, arm_field in (
            ("baseline", "baseline_method", "baseline_training_arm"),
            ("candidate", "candidate_method", "candidate_training_arm"),
        ):
            method = spec.get(side)
            if (
                not isinstance(method, Mapping)
                or method.get("method_id") != comparison[method_field]
                or method.get("training_arm") != comparison[arm_field]
            ):
                raise StatisticsError(
                    "seed-design comparison spec method identity mismatch"
                )
        repetitions = finite_integer(
            spec.get("bootstrap_repetitions"),
            f"seed-design comparison {comparison_id} bootstrap repetitions",
        )
        bootstrap_seed = finite_integer(
            spec.get("bootstrap_seed"),
            f"seed-design comparison {comparison_id} bootstrap seed",
        )
        if repetitions <= 0:
            raise StatisticsError(
                "seed-design comparison bootstrap repetitions must be positive"
            )
        comparison.update(
            {
                "comparison_spec_repository_path": spec_repository_path,
                "comparison_spec_sha256": spec_digest,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": bootstrap_seed,
                "raw_pvalue_method": RAW_PVALUE_METHOD,
                "resampling_convention": RESAMPLING_CONVENTION,
                "metric_seed_convention": (
                    "bootstrap_seed_plus_index_in_frozen_PDM_DEFAULT_METRICS"
                ),
                "bootstrap_pvalue_floor": 1.0 / (repetitions + 1.0),
            }
        )
        comparison_by_id[comparison_id] = comparison
        comparisons.append(comparison)

    raw_hypotheses = family.get("hypotheses")
    if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
        raise StatisticsError("seed-design family lacks ordered hypotheses")
    hypotheses: list[dict[str, str]] = []
    hypothesis_ids: list[str] = []
    metrics_by_comparison = {comparison_id: [] for comparison_id in comparison_by_id}
    for index, raw in enumerate(raw_hypotheses):
        if not isinstance(raw, Mapping):
            raise StatisticsError(f"seed-design hypothesis {index} is malformed")
        hypothesis = {field: raw.get(field) for field in HYPOTHESIS_IDENTITY_FIELDS}
        if any(not isinstance(value, str) or not value for value in hypothesis.values()):
            raise StatisticsError(f"seed-design hypothesis {index} identity is malformed")
        hypothesis_id = hypothesis["id"]
        comparison_id = hypothesis["comparison_id"]
        metric = hypothesis["metric"]
        if hypothesis_id in hypothesis_ids:
            raise StatisticsError("seed-design hypothesis IDs must be unique")
        comparison = comparison_by_id.get(comparison_id)
        if comparison is None:
            raise StatisticsError("seed-design hypothesis references an unknown comparison")
        for field in COMPARISON_IDENTITY_FIELDS[1:]:
            if hypothesis[field] != comparison[field]:
                raise StatisticsError("seed-design hypothesis method identity is inconsistent")
        if metric not in PDM_DEFAULT_METRICS:
            raise StatisticsError("seed-design hypothesis references an unknown metric")
        hypothesis_ids.append(hypothesis_id)
        hypotheses.append(hypothesis)
        metrics_by_comparison[comparison_id].append(metric)
    for comparison_id, metrics in metrics_by_comparison.items():
        if len(metrics) != len(PDM_DEFAULT_METRICS) or set(metrics) != set(
            PDM_DEFAULT_METRICS
        ):
            raise StatisticsError(
                f"seed-design comparison {comparison_id} must contain all seven metrics"
            )
    return {
        "family_id": family_id,
        "family_alpha": alpha,
        "dataset_identity": dict(dataset_identity),
        "expected_seed_ids": expected_seed_ids,
        "ordered_comparison_contracts": comparisons,
        "ordered_hypothesis_contracts": hypotheses,
        "ordered_hypothesis_ids": hypothesis_ids,
        "total_hypothesis_count": len(hypotheses),
    }


def load_joint_pilot(
    path: Path,
    *,
    family_contract: Mapping[str, Any],
    family_spec_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    """Load one joint crossed pilot array for the complete frozen family."""

    payload = _load_json(path, "joint seed-design pilot")
    expected_keys = {
        "schema",
        "split",
        "family_id",
        "family_spec_sha256",
        "ordered_hypothesis_ids",
        "seed_ids",
        "session_ids",
        "source_receipts",
        "source_receipt_sha256",
        "hypotheses",
    }
    if set(payload) != expected_keys or payload.get("schema") != PILOT_SCHEMA:
        raise StatisticsError("joint seed-design pilot fields/schema are not canonical")
    if payload.get("split") != "audit":
        raise StatisticsError("seed design may use audit-only pilot evidence")
    if payload.get("family_id") != family_contract["family_id"]:
        raise StatisticsError("joint pilot family ID mismatch")
    if payload.get("family_spec_sha256") != family_spec_sha256:
        raise StatisticsError("joint pilot family SHA256 mismatch")
    if payload.get("ordered_hypothesis_ids") != family_contract["ordered_hypothesis_ids"]:
        raise StatisticsError("joint pilot ordered hypothesis IDs mismatch")

    seed_values = payload.get("seed_ids")
    session_ids = payload.get("session_ids")
    if not isinstance(seed_values, list) or not isinstance(session_ids, list):
        raise StatisticsError("joint pilot seed/session identities must be lists")
    seed_ids = [finite_integer(value, "joint pilot seed ID") for value in seed_values]
    if len(seed_ids) < 2 or len(seed_ids) != len(set(seed_ids)):
        raise StatisticsError("joint pilot needs at least two unique seeds")
    if (
        len(session_ids) < 2
        or len(session_ids) != len(set(session_ids))
        or any(not isinstance(value, str) or not value for value in session_ids)
    ):
        raise StatisticsError("joint pilot needs at least two unique session identities")

    raw_hypotheses = payload.get("hypotheses")
    expected_hypotheses = family_contract["ordered_hypothesis_contracts"]
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != len(
        expected_hypotheses
    ):
        raise StatisticsError("joint pilot must contain every family hypothesis exactly once")
    shape = (len(seed_ids), len(session_ids))
    matrices: dict[str, np.ndarray] = {}
    effects: dict[str, float] = {}
    pilot_keys = set(HYPOTHESIS_IDENTITY_FIELDS) | {
        "minimum_relevant_effect",
        "candidate_minus_baseline",
    }
    for index, (raw, expected) in enumerate(zip(raw_hypotheses, expected_hypotheses)):
        if not isinstance(raw, dict) or set(raw) != pilot_keys:
            raise StatisticsError(f"joint pilot hypothesis {index} fields are malformed")
        observed_contract = {field: raw.get(field) for field in HYPOTHESIS_IDENTITY_FIELDS}
        if observed_contract != expected:
            raise StatisticsError(f"joint pilot hypothesis {index} identity mismatch")
        hypothesis_id = expected["id"]
        effect = finite_real(
            raw.get("minimum_relevant_effect"),
            f"joint pilot minimum effect for {hypothesis_id}",
        )
        if effect <= 0.0:
            raise StatisticsError("joint pilot minimum effects must be positive")
        matrix = require_finite_array(
            raw.get("candidate_minus_baseline"),
            f"joint pilot matrix for {hypothesis_id}",
        )
        if matrix.shape != shape:
            raise StatisticsError(f"joint pilot matrix for {hypothesis_id} has wrong shape")
        matrices[hypothesis_id] = matrix
        effects[hypothesis_id] = effect

    sources = payload.get("source_receipt_sha256")
    source_paths = payload.get("source_receipts")
    if (
        not isinstance(sources, list)
        or not sources
        or not isinstance(source_paths, list)
        or len(source_paths) != len(sources)
    ):
        raise StatisticsError("joint pilot lacks content-addressed source receipts")
    for index, (source_path, source_hash) in enumerate(zip(source_paths, sources)):
        if not isinstance(source_path, str) or not source_path:
            raise StatisticsError(f"joint pilot source path {index} is malformed")
        _sha256_value(source_hash, f"joint pilot source hash {index}")
    return matrices, effects, payload


def validate_seed_design_numeric_semantics(
    payload: Mapping[str, Any], label: str
) -> None:
    """Reject bool/non-finite values in every semantic numeric design field."""

    if payload.get("simulation_scope") not in {
        "confirmatory",
        "synthetic_cpu_fixture",
    }:
        raise StatisticsError(f"{label}.simulation_scope is invalid")
    _sha256_value(payload.get("replay_memo_key"), f"{label}.replay_memo_key")
    if payload.get("replay_memo_policy") != (
        "in_process_post_computation_content_addressed_v1"
    ):
        raise StatisticsError(f"{label}.replay_memo_policy is invalid")

    for field in ("target_power", "estimated_power", "family_alpha"):
        finite_real(payload.get(field), f"{label}.{field}")
    for field in (
        "total_hypothesis_count",
        "power_simulation_repetitions",
        "simulation_seed",
    ):
        finite_integer(payload.get(field), f"{label}.{field}")
    for index, value in enumerate(payload.get("designed_seed_ids", [])):
        finite_integer(value, f"{label}.designed_seed_ids[{index}]")
    for collection_name, comparisons in (
        ("ordered_comparison_contracts", payload.get("ordered_comparison_contracts")),
        (
            "normalized_invocation.ordered_comparison_contracts",
            (
                payload.get("normalized_invocation", {}).get(
                    "ordered_comparison_contracts"
                )
                if isinstance(payload.get("normalized_invocation"), Mapping)
                else None
            ),
        ),
    ):
        if not isinstance(comparisons, list):
            raise StatisticsError(f"{label}.{collection_name} must be a list")
        for index, comparison in enumerate(comparisons):
            if not isinstance(comparison, Mapping):
                raise StatisticsError(
                    f"{label}.{collection_name}[{index}] must be an object"
                )
            repetitions = finite_integer(
                comparison.get("bootstrap_repetitions"),
                f"{label}.{collection_name}[{index}].bootstrap_repetitions",
            )
            finite_integer(
                comparison.get("bootstrap_seed"),
                f"{label}.{collection_name}[{index}].bootstrap_seed",
            )
            floor = finite_real(
                comparison.get("bootstrap_pvalue_floor"),
                f"{label}.{collection_name}[{index}].bootstrap_pvalue_floor",
            )
            if repetitions <= 0 or floor != 1.0 / (repetitions + 1.0):
                raise StatisticsError(
                    f"{label}.{collection_name}[{index}] p-value resolution mismatch"
                )
    effects = payload.get("minimum_relevant_effects")
    if not isinstance(effects, Mapping):
        raise StatisticsError(f"{label}.minimum_relevant_effects must be an object")
    for hypothesis_id, value in effects.items():
        if finite_real(value, f"{label}.minimum_relevant_effects.{hypothesis_id}") <= 0.0:
            raise StatisticsError("minimum relevant effects must be positive")

    invocation = payload.get("normalized_invocation")
    if not isinstance(invocation, Mapping):
        raise StatisticsError(f"{label}.normalized_invocation must be an object")
    if invocation.get("simulation_scope") != payload.get("simulation_scope"):
        raise StatisticsError(f"{label}.normalized_invocation simulation scope mismatch")
    for field in ("target_power", "family_alpha"):
        finite_real(invocation.get(field), f"{label}.normalized_invocation.{field}")
    for field in (
        "total_hypothesis_count",
        "simulation_repetitions",
        "simulation_seed",
    ):
        finite_integer(invocation.get(field), f"{label}.normalized_invocation.{field}")
    for index, value in enumerate(invocation.get("expected_seed_ids", [])):
        finite_integer(value, f"{label}.normalized_invocation.expected_seed_ids[{index}]")

    audit = payload.get("family_seed_audit")
    if not isinstance(audit, Mapping):
        raise StatisticsError(f"{label}.family_seed_audit must be an object")
    for field in (
        "seed_count",
        "session_count",
        "total_hypothesis_count",
        "minimum_conditional_bootstrap_repetitions",
        "maximum_conditional_bootstrap_repetitions",
        "heldout_evaluation_repetitions",
        "total_conditional_bootstrap_resamples",
    ):
        finite_integer(audit.get(field), f"{label}.family_seed_audit.{field}")
    for field in (
        "heldout_null_fwer",
        "null_fwer_tolerance",
        "minimum_hypothesis_power",
        "bootstrap_pvalue_floor",
        "first_holm_threshold",
    ):
        finite_real(audit.get(field), f"{label}.family_seed_audit.{field}")
    if audit.get("bootstrap_resolution_pass") is not True:
        raise StatisticsError(f"{label}.family_seed_audit bootstrap resolution must pass")
    if audit.get("bootstrap_resolution_derivation") != (
        "maximum_per_hypothesis_plus_one_floor_le_first_global_holm_threshold"
    ):
        raise StatisticsError(f"{label}.family_seed_audit resolution derivation mismatch")
    if audit.get("rejected_sign_tail_rule") != (
        "two_to_the_one_minus_n_is_not_the_conditional_bootstrap_pvalue_floor"
    ):
        raise StatisticsError(f"{label}.family_seed_audit sign-tail rejection mismatch")
    thresholds = audit.get("holm_step_thresholds")
    if not isinstance(thresholds, list):
        raise StatisticsError(f"{label}.family_seed_audit Holm thresholds are malformed")
    for index, value in enumerate(thresholds):
        finite_real(value, f"{label}.family_seed_audit.holm_step_thresholds[{index}]")
    powers = audit.get("hypothesis_power")
    if not isinstance(powers, Mapping):
        raise StatisticsError(f"{label}.family_seed_audit.hypothesis_power is malformed")
    for hypothesis_id, values in powers.items():
        if not isinstance(values, Mapping) or set(values) != {
            "positive_effect",
            "negative_effect",
            "two_sided_minimum",
        }:
            raise StatisticsError(f"power fields for {hypothesis_id} are malformed")
        for field, value in values.items():
            finite_real(value, f"{label}.family_seed_audit.{hypothesis_id}.{field}")
    family_power = audit.get("family_power")
    if not isinstance(family_power, Mapping) or set(family_power) != {
        "any_rejection_positive_effect",
        "any_rejection_negative_effect",
        "any_rejection_two_sided_minimum",
        "all_rejections_positive_effect",
        "all_rejections_negative_effect",
        "all_rejections_two_sided_minimum",
    }:
        raise StatisticsError(f"{label}.family_seed_audit family power is malformed")
    for field, value in family_power.items():
        finite_real(value, f"{label}.family_seed_audit.family_power.{field}")
    kernel = audit.get("production_kernel_reference_pvalues")
    if not isinstance(kernel, Mapping):
        raise StatisticsError(f"{label}.family_seed_audit kernel checks are malformed")
    for hypothesis_id, values in kernel.items():
        if not isinstance(values, Mapping) or set(values) != {
            "bootstrap_repetitions",
            "bootstrap_seed",
            "pilot_observed_pvalue",
        }:
            raise StatisticsError(
                f"{label}.family_seed_audit.kernel.{hypothesis_id} is malformed"
            )
        finite_integer(
            values.get("bootstrap_repetitions"),
            f"{label}.family_seed_audit.kernel.{hypothesis_id}.repetitions",
        )
        finite_integer(
            values.get("bootstrap_seed"),
            f"{label}.family_seed_audit.kernel.{hypothesis_id}.seed",
        )
        finite_real(
            values.get("pilot_observed_pvalue"),
            f"{label}.family_seed_audit.kernel.{hypothesis_id}.pvalue",
        )


def simulate_family_design(
    matrices: Mapping[str, np.ndarray],
    effects: Mapping[str, float],
    *,
    ordered_hypothesis_ids: list[str],
    ordered_hypothesis_bootstrap_repetitions: list[int],
    ordered_hypothesis_kernel_seeds: list[int],
    seed_count: int,
    alpha: float,
    outer_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Replay production conditional p-values for every simulated family draw."""

    seed_count = finite_integer(seed_count, "seed count")
    outer_repetitions = finite_integer(
        outer_repetitions, "power simulation repetitions"
    )
    alpha = finite_real(alpha, "family alpha")
    if seed_count < 2 or outer_repetitions < 1 or not 0.0 < alpha < 1.0:
        raise StatisticsError("family seed simulation inputs are outside their domain")
    if list(matrices) != ordered_hypothesis_ids or list(effects) != ordered_hypothesis_ids:
        raise StatisticsError("seed simulation matrices/effects do not match hypothesis order")
    if (
        len(ordered_hypothesis_bootstrap_repetitions)
        != len(ordered_hypothesis_ids)
        or len(ordered_hypothesis_kernel_seeds) != len(ordered_hypothesis_ids)
    ):
        raise StatisticsError("seed simulation downstream kernel seeds are malformed")
    bootstrap_repetitions = [
        finite_integer(value, "downstream hypothesis bootstrap repetitions")
        for value in ordered_hypothesis_bootstrap_repetitions
    ]
    if any(value <= 0 for value in bootstrap_repetitions):
        raise StatisticsError("downstream bootstrap repetitions must be positive")
    kernel_seeds = [
        finite_integer(value, "downstream hypothesis bootstrap seed")
        for value in ordered_hypothesis_kernel_seeds
    ]
    centered = {
        hypothesis_id: matrices[hypothesis_id] - float(matrices[hypothesis_id].mean())
        for hypothesis_id in ordered_hypothesis_ids
    }
    shapes = {matrix.shape for matrix in centered.values()}
    if len(shapes) != 1:
        raise StatisticsError("joint seed simulation matrices must have one crossed shape")
    pilot_seed_count, session_count = next(iter(shapes))
    seed_indices = rng.integers(
        0, pilot_seed_count, size=(outer_repetitions, seed_count)
    )
    session_indices = rng.integers(
        0, session_count, size=(outer_repetitions, session_count)
    )
    family_size = len(ordered_hypothesis_ids)
    first_holm_threshold = alpha / family_size
    null_pvalues = np.empty((outer_repetitions, family_size), dtype=np.float64)
    positive_pvalues = np.empty_like(null_pvalues)
    negative_pvalues = np.empty_like(null_pvalues)
    production_kernel_reference_pvalues: dict[str, dict[str, float | int]] = {}
    for index, (
        hypothesis_id,
        hypothesis_repetitions,
        hypothesis_seed,
    ) in enumerate(
        zip(
            ordered_hypothesis_ids,
            bootstrap_repetitions,
            kernel_seeds,
        )
    ):
        sampled = centered[hypothesis_id][
            seed_indices[:, :, None], session_indices[:, None, :]
        ]
        effect = finite_real(effects[hypothesis_id], f"effect for {hypothesis_id}")
        conditional = crossed_matrix_bootstrap_shift_pvalues_batch(
            sampled,
            constant_shifts=(0.0, effect, -effect),
            repetitions=hypothesis_repetitions,
            seed=hypothesis_seed,
        )
        pvalues = conditional["pvalues"]
        null_pvalues[:, index] = pvalues[:, 0]
        positive_pvalues[:, index] = pvalues[:, 1]
        negative_pvalues[:, index] = pvalues[:, 2]
        production_kernel_reference_pvalues[hypothesis_id] = {
            "bootstrap_repetitions": hypothesis_repetitions,
            "bootstrap_seed": hypothesis_seed,
            "pilot_observed_pvalue": float(
                crossed_matrix_bootstrap_inference(
                    matrices[hypothesis_id],
                    repetitions=hypothesis_repetitions,
                    seed=hypothesis_seed,
                )["null_centered_studentized_pvalue"]
            ),
        }
    if not all(
        np.all(np.isfinite(value))
        for value in (null_pvalues, positive_pvalues, negative_pvalues)
    ):
        raise StatisticsError("family seed simulation produced non-finite p-values")
    null_rejected = holm_rejection_matrix(null_pvalues, alpha=alpha)
    positive_rejected = holm_rejection_matrix(positive_pvalues, alpha=alpha)
    negative_rejected = holm_rejection_matrix(negative_pvalues, alpha=alpha)
    heldout_null_fwer = float(np.mean(np.any(null_rejected, axis=1)))
    hypothesis_power: dict[str, dict[str, float]] = {}
    for index, hypothesis_id in enumerate(ordered_hypothesis_ids):
        positive_power = float(np.mean(positive_rejected[:, index]))
        negative_power = float(np.mean(negative_rejected[:, index]))
        hypothesis_power[hypothesis_id] = {
            "positive_effect": positive_power,
            "negative_effect": negative_power,
            "two_sided_minimum": min(positive_power, negative_power),
        }
    minimum_power = min(
        values["two_sided_minimum"] for values in hypothesis_power.values()
    )
    monte_carlo_tolerance = 3.0 * math.sqrt(
        max(alpha * (1.0 - alpha) / outer_repetitions, 0.0)
    )
    bootstrap_pvalue_floor = max(
        1.0 / (value + 1.0) for value in bootstrap_repetitions
    )
    holm_step_thresholds = [
        alpha / (family_size - rank) for rank in range(family_size)
    ]
    positive_any = float(np.mean(np.any(positive_rejected, axis=1)))
    negative_any = float(np.mean(np.any(negative_rejected, axis=1)))
    positive_all = float(np.mean(np.all(positive_rejected, axis=1)))
    negative_all = float(np.mean(np.all(negative_rejected, axis=1)))
    result = {
        "seed_count": seed_count,
        "session_count": session_count,
        "total_hypothesis_count": len(ordered_hypothesis_ids),
        "ordered_hypothesis_ids": ordered_hypothesis_ids,
        "pvalue_method": RAW_PVALUE_METHOD,
        "multiple_testing_method": "global_holm_step_down",
        "power_calibration_contract": (
            "joint_audit_pilot_outer_resampling_with_per_observed_"
            "production_conditional_crossed_bootstrap_pvalues"
        ),
        "production_kernel_reference_pvalues": (
            production_kernel_reference_pvalues
        ),
        "bootstrap_pvalue_floor": bootstrap_pvalue_floor,
        "bootstrap_resolution_derivation": (
            "maximum_per_hypothesis_plus_one_floor_le_first_global_holm_threshold"
        ),
        "rejected_sign_tail_rule": (
            "two_to_the_one_minus_n_is_not_the_conditional_bootstrap_pvalue_floor"
        ),
        "first_holm_threshold": first_holm_threshold,
        "bootstrap_resolution_pass": bootstrap_pvalue_floor <= first_holm_threshold,
        "holm_step_thresholds": holm_step_thresholds,
        "heldout_null_fwer": heldout_null_fwer,
        "null_fwer_tolerance": alpha + monte_carlo_tolerance,
        "hypothesis_power": hypothesis_power,
        "minimum_hypothesis_power": minimum_power,
        "family_power": {
            "any_rejection_positive_effect": positive_any,
            "any_rejection_negative_effect": negative_any,
            "any_rejection_two_sided_minimum": min(positive_any, negative_any),
            "all_rejections_positive_effect": positive_all,
            "all_rejections_negative_effect": negative_all,
            "all_rejections_two_sided_minimum": min(positive_all, negative_all),
        },
        "minimum_conditional_bootstrap_repetitions": min(bootstrap_repetitions),
        "maximum_conditional_bootstrap_repetitions": max(bootstrap_repetitions),
        "heldout_evaluation_repetitions": outer_repetitions,
        "total_conditional_bootstrap_resamples": int(
            outer_repetitions * sum(bootstrap_repetitions)
        ),
    }
    require_finite_tree(result, "family seed simulation output")
    return result


def normalized_invocation(
    *,
    family_spec_repository_path: str,
    family_spec_sha256: str,
    pilot_matrix_sha256: str,
    family_contract: Mapping[str, Any],
    simulation_scope: str,
    target_power: float,
    simulation_repetitions: int,
    simulation_seed: int,
) -> dict[str, Any]:
    payload = {
        "schema": INVOCATION_SCHEMA,
        "family_spec_repository_path": family_spec_repository_path,
        "family_spec_sha256": _sha256_value(
            family_spec_sha256, "normalized family SHA256"
        ),
        "pilot_matrix_sha256": _sha256_value(
            pilot_matrix_sha256, "normalized pilot SHA256"
        ),
        "family_id": family_contract["family_id"],
        "dataset_identity": dict(family_contract["dataset_identity"]),
        "family_alpha": finite_real(
            family_contract["family_alpha"], "normalized family alpha"
        ),
        "expected_seed_ids": list(family_contract["expected_seed_ids"]),
        "ordered_comparison_contracts": list(
            family_contract["ordered_comparison_contracts"]
        ),
        "ordered_hypothesis_contracts": list(
            family_contract["ordered_hypothesis_contracts"]
        ),
        "ordered_hypothesis_ids": list(family_contract["ordered_hypothesis_ids"]),
        "total_hypothesis_count": finite_integer(
            family_contract["total_hypothesis_count"],
            "normalized hypothesis count",
        ),
        "target_power": finite_real(target_power, "normalized target power"),
        "simulation_scope": simulation_scope,
        "simulation_repetitions": finite_integer(
            simulation_repetitions, "normalized simulation repetitions"
        ),
        "simulation_seed": finite_integer(
            simulation_seed, "normalized simulation seed"
        ),
    }
    require_finite_tree(payload, "normalized seed-design invocation")
    return payload


def build_seed_design_payload(
    family_spec: Path,
    pilot_matrix: Path,
    *,
    repository: Path,
    family_spec_repository_path: str,
    pilot_reference: str,
    simulation_scope: str,
    target_power: float,
    simulation_repetitions: int,
    simulation_seed: int,
    generator_module_repository_path: str,
    generator_module_sha256: str,
    generator_entrypoint_repository_path: str,
    generator_entrypoint_sha256: str,
    inference_module_repository_path: str,
    inference_module_sha256: str,
) -> dict[str, Any]:
    """Build one executable receipt for the exact frozen family and joint pilot."""

    simulation_repetitions = finite_integer(
        simulation_repetitions, "simulation repetitions"
    )
    simulation_seed = finite_integer(simulation_seed, "simulation seed")
    target_power = finite_real(target_power, "target power")
    if simulation_scope not in {"confirmatory", "synthetic_cpu_fixture"}:
        raise StatisticsError("seed-design simulation scope is invalid")
    if not 0.0 < target_power < 1.0:
        raise StatisticsError("target power must lie strictly between zero and one")
    family = _load_json(family_spec, "seed-design family specification")
    family_digest = sha256(family_spec)
    family_contract = family_hypothesis_contract(family, repository=repository)
    dataset_identity = family_contract["dataset_identity"]
    synthetic_identity = {
        "dataset": "navsim-fixture",
        "dataset_version": "v1",
        "dataset_root_metadata_sha256": "d" * 64,
    }
    if simulation_scope == "synthetic_cpu_fixture":
        if (
            dataset_identity != synthetic_identity
            or family_contract["family_id"] != "p003_primary"
        ):
            raise StatisticsError(
                "synthetic CPU calibration is restricted to the exact non-paper fixture"
            )
        minimum_outer_repetitions = SYNTHETIC_FIXTURE_SIMULATION_MINIMUM
    else:
        if dataset_identity == synthetic_identity:
            raise StatisticsError(
                "the synthetic fixture cannot enter confirmatory calibration"
            )
        minimum_outer_repetitions = PRODUCTION_POWER_SIMULATION_MINIMUM
    if simulation_repetitions < minimum_outer_repetitions:
        raise StatisticsError(
            "power/FWER outer simulation repetitions are below the scope minimum"
        )
    comparison_by_id = {
        item["comparison_id"]: item
        for item in family_contract["ordered_comparison_contracts"]
    }
    matrices, effects, pilot = load_joint_pilot(
        pilot_matrix,
        family_contract=family_contract,
        family_spec_sha256=family_digest,
    )
    pilot_digest = sha256(pilot_matrix)
    invocation = normalized_invocation(
        family_spec_repository_path=family_spec_repository_path,
        family_spec_sha256=family_digest,
        pilot_matrix_sha256=pilot_digest,
        family_contract=family_contract,
        simulation_scope=simulation_scope,
        target_power=target_power,
        simulation_repetitions=simulation_repetitions,
        simulation_seed=simulation_seed,
    )
    memo_key, _ = seed_design_replay_identity(
        normalized_invocation_payload=invocation,
        pilot_reference=pilot_reference,
        generator_module_repository_path=generator_module_repository_path,
        generator_module_sha256=generator_module_sha256,
        generator_entrypoint_repository_path=generator_entrypoint_repository_path,
        generator_entrypoint_sha256=generator_entrypoint_sha256,
        inference_module_repository_path=inference_module_repository_path,
        inference_module_sha256=inference_module_sha256,
    )
    cached_bytes = _memo_lookup(memo_key)
    if cached_bytes is not None:
        cached = json.loads(cached_bytes)
        expected_cached_bindings = {
            "schema": SEED_DESIGN_SCHEMA,
            "family_spec_sha256": family_digest,
            "pilot_matrix_sha256": pilot_digest,
            "pilot_matrix": pilot_reference,
            "simulation_scope": simulation_scope,
            "normalized_invocation": invocation,
            "replay_memo_key": memo_key,
            "generator_module_sha256": generator_module_sha256,
            "generator_entrypoint_sha256": generator_entrypoint_sha256,
            "inference_module_sha256": inference_module_sha256,
        }
        for field, expected in expected_cached_bindings.items():
            if cached.get(field) != expected:
                raise StatisticsError(
                    f"seed-design replay memo {field} binding mismatch"
                )
        require_finite_tree(cached, "cached seed-design payload")
        validate_seed_design_numeric_semantics(cached, "cached seed-design payload")
        return cached
    audit = simulate_family_design(
        matrices,
        effects,
        ordered_hypothesis_ids=list(family_contract["ordered_hypothesis_ids"]),
        ordered_hypothesis_bootstrap_repetitions=[
            comparison_by_id[item["comparison_id"]]["bootstrap_repetitions"]
            for item in family_contract["ordered_hypothesis_contracts"]
        ],
        ordered_hypothesis_kernel_seeds=[
            comparison_by_id[item["comparison_id"]]["bootstrap_seed"]
            + PDM_DEFAULT_METRICS.index(item["metric"])
            for item in family_contract["ordered_hypothesis_contracts"]
        ],
        seed_count=len(family_contract["expected_seed_ids"]),
        alpha=float(family_contract["family_alpha"]),
        outer_repetitions=simulation_repetitions,
        rng=np.random.default_rng(simulation_seed),
    )
    passed = (
        audit["minimum_hypothesis_power"] >= target_power
        and audit["heldout_null_fwer"] <= audit["null_fwer_tolerance"]
        and audit["bootstrap_resolution_pass"]
    )
    status = "passed" if passed else "failed"
    payload = {
        "schema": SEED_DESIGN_SCHEMA,
        "status": status,
        "design_split": "audit",
        "family_spec_repository_path": family_spec_repository_path,
        "family_spec_sha256": family_digest,
        "family_id": family_contract["family_id"],
        "dataset_identity": family_contract["dataset_identity"],
        "family_alpha": float(family_contract["family_alpha"]),
        "ordered_comparison_contracts": family_contract[
            "ordered_comparison_contracts"
        ],
        "ordered_hypothesis_contracts": family_contract[
            "ordered_hypothesis_contracts"
        ],
        "ordered_hypothesis_ids": family_contract["ordered_hypothesis_ids"],
        "total_hypothesis_count": family_contract["total_hypothesis_count"],
        "designed_seed_ids": (
            family_contract["expected_seed_ids"] if passed else []
        ),
        "minimum_relevant_effects": effects,
        "target_power": target_power,
        "simulation_scope": simulation_scope,
        "estimated_power": float(audit["minimum_hypothesis_power"]),
        "power_simulation_repetitions": simulation_repetitions,
        "simulation_seed": simulation_seed,
        "replay_memo_key": memo_key,
        "replay_memo_policy": (
            "in_process_post_computation_content_addressed_v1"
        ),
        "calibration_status": status,
        "calibration_method": (
            "joint_outer_resampling_exact_production_conditional_pvalues_global_holm"
        ),
        "power_method": "audit_only_two_sided_per_hypothesis_minimum_power",
        "pilot_matrix": pilot_reference,
        "pilot_matrix_sha256": pilot_digest,
        "audit_source_receipt_sha256": pilot["source_receipt_sha256"],
        "family_seed_audit": audit,
        "normalized_invocation": invocation,
        "generator_module_repository_path": generator_module_repository_path,
        "generator_module_sha256": _sha256_value(
            generator_module_sha256, "generator module SHA256"
        ),
        "generator_entrypoint_repository_path": generator_entrypoint_repository_path,
        "generator_entrypoint_sha256": _sha256_value(
            generator_entrypoint_sha256, "generator entrypoint SHA256"
        ),
        "inference_module_repository_path": inference_module_repository_path,
        "inference_module_sha256": _sha256_value(
            inference_module_sha256, "inference module SHA256"
        ),
    }
    require_finite_tree(payload, "seed-design payload")
    validate_seed_design_numeric_semantics(payload, "seed-design payload")
    _memo_store(memo_key, _canonical_json_bytes(payload))
    return payload

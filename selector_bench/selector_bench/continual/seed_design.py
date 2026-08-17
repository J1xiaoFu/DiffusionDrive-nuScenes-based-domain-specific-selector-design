"""Deterministic crossed seed-design computation shared by CLI and claim gate.

The claim validator must recompute the complete design from the frozen pilot
matrix.  A schema-consistent, handwritten power receipt is not evidence that
the preregistered design was executed.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from selector_bench.continual.statistics import PDM_DEFAULT_METRICS, StatisticsError


PILOT_SCHEMA = "selector_bench.drive_cl_audit_pilot_matrix.v1"
SEED_DESIGN_SCHEMA = "selector_bench.drive_cl_seed_design.v2"
INVOCATION_SCHEMA = "selector_bench.drive_cl_seed_design_invocation.v1"


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


def validate_seed_design_numeric_semantics(
    payload: Mapping[str, Any], label: str
) -> None:
    """Reject bool/non-finite values in every semantic numeric design field."""

    for field in (
        "minimum_relevant_effect",
        "target_power",
        "estimated_power",
        "family_alpha",
    ):
        finite_real(payload.get(field), f"{label}.{field}")
    for field in (
        "family_metric_count",
        "minimum_seed_count_for_two_sided_family_tail_resolution",
        "power_simulation_repetitions",
        "simulation_seed",
    ):
        finite_integer(payload.get(field), f"{label}.{field}")
    for field in ("designed_seed_ids", "candidate_seed_ids"):
        values = payload.get(field)
        if not isinstance(values, list):
            raise StatisticsError(f"{label}.{field} must be a list")
        for index, value in enumerate(values):
            finite_integer(value, f"{label}.{field}[{index}]")
    invocation = payload.get("normalized_invocation")
    if not isinstance(invocation, Mapping):
        raise StatisticsError(f"{label}.normalized_invocation must be an object")
    for field in ("minimum_relevant_effect", "target_power", "family_alpha"):
        finite_real(invocation.get(field), f"{label}.normalized_invocation.{field}")
    for field in ("simulation_repetitions", "simulation_seed"):
        finite_integer(invocation.get(field), f"{label}.normalized_invocation.{field}")
    invocation_seeds = invocation.get("candidate_seed_ids")
    if not isinstance(invocation_seeds, list):
        raise StatisticsError(
            f"{label}.normalized_invocation.candidate_seed_ids must be a list"
        )
    for index, value in enumerate(invocation_seeds):
        finite_integer(
            value, f"{label}.normalized_invocation.candidate_seed_ids[{index}]"
        )
    audits = payload.get("candidate_seed_audits")
    if not isinstance(audits, list):
        raise StatisticsError(f"{label}.candidate_seed_audits must be a list")
    for index, audit in enumerate(audits):
        if not isinstance(audit, Mapping):
            raise StatisticsError(f"{label}.candidate_seed_audits[{index}] is malformed")
        for field in (
            "seed_count",
            "session_count",
            "calibration_simulations",
            "evaluation_simulations",
        ):
            finite_integer(audit.get(field), f"{label}.candidate_seed_audits[{index}].{field}")
        for field in (
            "critical_value",
            "heldout_null_fwer",
            "null_fwer_tolerance",
            "minimum_metric_power",
        ):
            finite_real(audit.get(field), f"{label}.candidate_seed_audits[{index}].{field}")
        metric_power = audit.get("metric_power")
        if not isinstance(metric_power, Mapping) or set(metric_power) != set(
            PDM_DEFAULT_METRICS
        ):
            raise StatisticsError(
                f"{label}.candidate_seed_audits[{index}].metric_power is malformed"
            )
        for metric, value in metric_power.items():
            finite_real(
                value,
                f"{label}.candidate_seed_audits[{index}].metric_power.{metric}",
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossed_standard_error_batch(matrix: np.ndarray) -> np.ndarray:
    """Return crossed seed/session standard errors for ``[R, S, N]`` arrays."""

    if matrix.ndim != 3 or not np.all(np.isfinite(matrix)):
        raise StatisticsError("crossed seed design requires [repetition, seed, session]")
    _, seed_count, session_count = matrix.shape
    if seed_count < 2 or session_count < 2:
        raise StatisticsError("crossed seed design requires two seeds and sessions")
    grand = matrix.mean(axis=(1, 2))
    row_means = matrix.mean(axis=2)
    column_means = matrix.mean(axis=1)
    residual = (
        matrix
        - row_means[:, :, None]
        - column_means[:, None, :]
        + grand[:, None, None]
    )
    residual_variance = np.square(residual).sum(axis=(1, 2)) / (
        (seed_count - 1) * (session_count - 1)
    )
    seed_component = np.maximum(
        0.0,
        np.var(row_means, axis=1, ddof=1) - residual_variance / session_count,
    )
    session_component = np.maximum(
        0.0,
        np.var(column_means, axis=1, ddof=1) - residual_variance / seed_count,
    )
    result = np.sqrt(
        np.maximum(
            seed_component / seed_count
            + session_component / session_count
            + residual_variance / (seed_count * session_count),
            0.0,
        )
    )
    if not np.all(np.isfinite(result)):
        raise StatisticsError("crossed seed standard errors are non-finite")
    return result


def statistic_batch(matrix: np.ndarray) -> np.ndarray:
    standard_error = crossed_standard_error_batch(matrix)
    point = matrix.mean(axis=(1, 2))
    epsilon = np.finfo(np.float64).eps
    result = np.empty_like(point)
    regular = standard_error > epsilon
    np.divide(np.abs(point), standard_error, out=result, where=regular)
    if np.any(~regular & (np.abs(point) > epsilon)):
        raise StatisticsError("crossed seed statistic is non-finite for degenerate data")
    result[~regular] = 0.0
    if not np.all(np.isfinite(result)):
        raise StatisticsError("crossed seed statistic is non-finite")
    return result


def load_pilot(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import json

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise StatisticsError("invalid pilot matrix JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PILOT_SCHEMA:
        raise StatisticsError("unsupported pilot matrix schema")
    if payload.get("split") != "audit":
        raise StatisticsError("seed design may use audit-only pilot evidence")
    if tuple(payload.get("metrics", [])) != PDM_DEFAULT_METRICS:
        raise StatisticsError("pilot matrix must contain the frozen seven metrics")
    pilot_seed_ids = payload.get("seed_ids")
    session_ids = payload.get("session_ids")
    if (
        not isinstance(pilot_seed_ids, list)
        or len(pilot_seed_ids) < 2
        or len(pilot_seed_ids) != len(set(pilot_seed_ids))
        or not isinstance(session_ids, list)
        or len(session_ids) < 2
        or len(session_ids) != len(set(session_ids))
    ):
        raise StatisticsError("pilot matrix needs at least two unique seeds and sessions")
    for index, value in enumerate(pilot_seed_ids):
        finite_integer(value, f"pilot seed_ids[{index}]")
    if any(not isinstance(value, str) or not value for value in session_ids):
        raise StatisticsError("pilot session IDs must be non-empty strings")
    differences = payload.get("candidate_minus_baseline")
    if not isinstance(differences, dict):
        raise StatisticsError("pilot matrix lacks candidate-minus-baseline values")
    shape = (len(pilot_seed_ids), len(session_ids))
    matrices: dict[str, np.ndarray] = {}
    for metric in PDM_DEFAULT_METRICS:
        matrix = require_finite_array(differences.get(metric), f"pilot matrix for {metric}")
        if matrix.shape != shape:
            raise StatisticsError(f"pilot matrix for {metric} has invalid shape/values")
        matrices[metric] = matrix
    sources = payload.get("source_receipt_sha256")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(value, str) or len(value) != 64 for value in sources
    ):
        raise StatisticsError("pilot matrix lacks content-addressed source receipts")
    return matrices, payload


def simulate_design(
    matrices: Mapping[str, np.ndarray],
    *,
    seed_count: int,
    effect: float,
    alpha: float,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Vectorized deterministic null calibration and alternative power replay."""

    seed_count = finite_integer(seed_count, "seed count")
    repetitions = finite_integer(repetitions, "simulation repetitions")
    effect = finite_real(effect, "minimum relevant effect")
    alpha = finite_real(alpha, "family alpha")
    if seed_count < 2 or repetitions < 2 or not 0.0 < alpha < 1.0:
        raise StatisticsError("seed simulation inputs are outside their finite domain")
    metric_names = tuple(PDM_DEFAULT_METRICS)
    centered = {
        metric: matrix - float(matrix.mean()) for metric, matrix in matrices.items()
    }
    pilot_seed_count, session_count = next(iter(centered.values())).shape
    seed_indices = rng.integers(
        0, pilot_seed_count, size=(repetitions, seed_count)
    )
    session_indices = rng.integers(
        0, session_count, size=(repetitions, session_count)
    )
    null_statistics = np.empty((repetitions, len(metric_names)), dtype=np.float64)
    alternative_statistics = np.empty_like(null_statistics)
    for metric_index, metric in enumerate(metric_names):
        sampled = centered[metric][
            seed_indices[:, :, None], session_indices[:, None, :]
        ]
        null_statistics[:, metric_index] = statistic_batch(sampled)
        alternative_statistics[:, metric_index] = statistic_batch(sampled + effect)
    if not np.all(np.isfinite(null_statistics)) or not np.all(
        np.isfinite(alternative_statistics)
    ):
        raise StatisticsError("seed simulation produced non-finite statistics")
    calibration_count = repetitions // 2
    evaluation_count = repetitions - calibration_count
    family_null = np.max(null_statistics[:calibration_count], axis=1)
    critical_value = float(np.percentile(family_null, 100.0 * (1.0 - alpha)))
    evaluation_null = null_statistics[calibration_count:]
    evaluation_alternative = alternative_statistics[calibration_count:]
    heldout_null_fwer = float(
        np.mean(np.max(evaluation_null, axis=1) > critical_value)
    )
    metric_power = {
        metric: float(
            np.mean(evaluation_alternative[:, index] > critical_value)
        )
        for index, metric in enumerate(metric_names)
    }
    monte_carlo_tolerance = 3.0 * math.sqrt(
        max(alpha * (1.0 - alpha) / max(evaluation_count, 1), 0.0)
    )
    result = {
        "seed_count": seed_count,
        "session_count": session_count,
        "critical_value": critical_value,
        "heldout_null_fwer": heldout_null_fwer,
        "null_fwer_tolerance": alpha + monte_carlo_tolerance,
        "metric_power": metric_power,
        "minimum_metric_power": min(metric_power.values()),
        "calibration_simulations": calibration_count,
        "evaluation_simulations": evaluation_count,
    }
    require_finite_tree(result, "seed simulation output")
    return result


def normalized_invocation(
    *,
    pilot_matrix_sha256: str,
    candidate_seed_ids: Sequence[int],
    minimum_relevant_effect: float,
    target_power: float,
    family_alpha: float,
    simulation_repetitions: int,
    simulation_seed: int,
) -> dict[str, Any]:
    candidate_ids = [
        finite_integer(value, "candidate seed ID") for value in candidate_seed_ids
    ]
    effect = finite_real(minimum_relevant_effect, "minimum relevant effect")
    power = finite_real(target_power, "target power")
    alpha = finite_real(family_alpha, "family alpha")
    repetitions = finite_integer(simulation_repetitions, "simulation repetitions")
    seed = finite_integer(simulation_seed, "simulation seed")
    payload = {
        "schema": INVOCATION_SCHEMA,
        "pilot_matrix_sha256": pilot_matrix_sha256,
        "candidate_seed_ids": candidate_ids,
        "minimum_relevant_effect": effect,
        "target_power": power,
        "family_alpha": alpha,
        "simulation_repetitions": repetitions,
        "simulation_seed": seed,
    }
    require_finite_tree(payload, "normalized seed-design invocation")
    return payload


def build_seed_design_payload(
    pilot_matrix: Path,
    *,
    pilot_reference: str,
    candidate_seed_ids: Sequence[int],
    minimum_relevant_effect: float,
    target_power: float,
    family_alpha: float,
    simulation_repetitions: int,
    simulation_seed: int,
    generator_module_repository_path: str,
    generator_module_sha256: str,
    generator_entrypoint_repository_path: str,
    generator_entrypoint_sha256: str,
) -> dict[str, Any]:
    simulation_repetitions = finite_integer(
        simulation_repetitions, "simulation repetitions"
    )
    simulation_seed = finite_integer(simulation_seed, "simulation seed")
    minimum_relevant_effect = finite_real(
        minimum_relevant_effect, "minimum relevant effect"
    )
    target_power = finite_real(target_power, "target power")
    family_alpha = finite_real(family_alpha, "family alpha")
    if simulation_repetitions < 10_000:
        raise StatisticsError("confirmatory power/calibration requires 10000 simulations")
    candidate_ids = tuple(
        finite_integer(value, "candidate seed ID") for value in candidate_seed_ids
    )
    if len(candidate_ids) < 2 or len(candidate_ids) != len(set(candidate_ids)):
        raise StatisticsError("candidate seed IDs need at least two unique values")
    if minimum_relevant_effect <= 0.0:
        raise StatisticsError("minimum relevant effect must be positive")
    if not 0.0 < target_power < 1.0 or not 0.0 < family_alpha < 1.0:
        raise StatisticsError("power and alpha must lie strictly between zero and one")
    matrices, pilot = load_pilot(pilot_matrix)
    pilot_digest = sha256(pilot_matrix)
    invocation = normalized_invocation(
        pilot_matrix_sha256=pilot_digest,
        candidate_seed_ids=candidate_ids,
        minimum_relevant_effect=minimum_relevant_effect,
        target_power=target_power,
        family_alpha=family_alpha,
        simulation_repetitions=simulation_repetitions,
        simulation_seed=simulation_seed,
    )
    tail_resolution_minimum = math.ceil(
        1.0 - math.log2(family_alpha / len(PDM_DEFAULT_METRICS))
    )
    generator = np.random.default_rng(simulation_seed)
    audits = []
    selected: dict[str, Any] | None = None
    for seed_count in range(tail_resolution_minimum, len(candidate_ids) + 1):
        audit = simulate_design(
            matrices,
            seed_count=seed_count,
            effect=minimum_relevant_effect,
            alpha=family_alpha,
            repetitions=simulation_repetitions,
            rng=generator,
        )
        audits.append(audit)
        if (
            audit["minimum_metric_power"] >= target_power
            and audit["heldout_null_fwer"] <= audit["null_fwer_tolerance"]
        ):
            selected = audit
            break
    status = "passed" if selected is not None else "failed"
    designed_seed_ids = (
        list(candidate_ids[: int(selected["seed_count"])])
        if selected is not None
        else []
    )
    payload = {
        "schema": SEED_DESIGN_SCHEMA,
        "status": status,
        "design_split": "audit",
        "designed_seed_ids": designed_seed_ids,
        "minimum_relevant_effect": float(minimum_relevant_effect),
        "target_power": float(target_power),
        "estimated_power": (
            float(selected["minimum_metric_power"]) if selected is not None else 0.0
        ),
        "family_alpha": float(family_alpha),
        "family_metric_count": len(PDM_DEFAULT_METRICS),
        "minimum_seed_count_for_two_sided_family_tail_resolution": (
            tail_resolution_minimum
        ),
        "tail_resolution_rule": "2^(1-n_seed) <= family_alpha / family_metric_count",
        "power_simulation_repetitions": int(simulation_repetitions),
        "simulation_seed": int(simulation_seed),
        "calibration_status": status,
        "calibration_method": (
            "heldout_null_crossed_resampling_max_statistic_over_seven_metrics"
        ),
        "power_method": "audit_only_crossed_resampling_at_minimum_relevant_effect",
        "pilot_matrix": pilot_reference,
        "pilot_matrix_sha256": pilot_digest,
        "audit_source_receipt_sha256": pilot["source_receipt_sha256"],
        "candidate_seed_ids": list(candidate_ids),
        "candidate_seed_audits": audits,
        "normalized_invocation": invocation,
        "generator_module_repository_path": generator_module_repository_path,
        "generator_module_sha256": generator_module_sha256,
        "generator_entrypoint_repository_path": generator_entrypoint_repository_path,
        "generator_entrypoint_sha256": generator_entrypoint_sha256,
    }
    require_finite_tree(payload, "seed-design payload")
    validate_seed_design_numeric_semantics(payload, "seed-design payload")
    return payload

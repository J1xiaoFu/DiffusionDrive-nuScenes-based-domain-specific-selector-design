#!/usr/bin/env python3
"""Design confirmatory seed count from audit-only crossed pilot matrices."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.continual.claim_protocol import SEED_DESIGN_SCHEMA, sha256
from selector_bench.continual.statistics import PDM_DEFAULT_METRICS, StatisticsError


PILOT_SCHEMA = "selector_bench.drive_cl_audit_pilot_matrix.v1"


def crossed_standard_error(matrix: np.ndarray) -> float:
    seed_count, session_count = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    residual_variance = float(
        np.square(residual).sum() / ((seed_count - 1) * (session_count - 1))
    )
    seed_component = max(
        0.0, float(np.var(row_means, ddof=1)) - residual_variance / session_count
    )
    session_component = max(
        0.0, float(np.var(column_means, ddof=1)) - residual_variance / seed_count
    )
    return math.sqrt(
        max(
            seed_component / seed_count
            + session_component / session_count
            + residual_variance / (seed_count * session_count),
            0.0,
        )
    )


def statistic(matrix: np.ndarray) -> float:
    standard_error = crossed_standard_error(matrix)
    point = float(matrix.mean())
    if standard_error <= np.finfo(np.float64).eps:
        return 0.0 if abs(point) <= np.finfo(np.float64).eps else math.inf
    return abs(point / standard_error)


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
    return args


def load_pilot(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    matrices: dict[str, np.ndarray] = {}
    differences = payload.get("candidate_minus_baseline")
    if not isinstance(differences, dict):
        raise StatisticsError("pilot matrix lacks candidate-minus-baseline values")
    shape = (len(pilot_seed_ids), len(session_ids))
    for metric in PDM_DEFAULT_METRICS:
        matrix = np.asarray(differences.get(metric), dtype=np.float64)
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise StatisticsError(f"pilot matrix for {metric} has invalid shape/values")
        matrices[metric] = matrix
    sources = payload.get("source_receipt_sha256")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(value, str) or len(value) != 64 for value in sources
    ):
        raise StatisticsError("pilot matrix lacks content-addressed source receipts")
    return matrices, payload


def simulate_design(
    matrices: dict[str, np.ndarray],
    *,
    seed_count: int,
    effect: float,
    alpha: float,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    metric_names = tuple(PDM_DEFAULT_METRICS)
    centered = {metric: matrix - float(matrix.mean()) for metric, matrix in matrices.items()}
    pilot_seed_count, session_count = next(iter(centered.values())).shape
    calibration_count = repetitions // 2
    evaluation_count = repetitions - calibration_count
    null_statistics = np.empty((repetitions, len(metric_names)), dtype=np.float64)
    alternative_statistics = np.empty_like(null_statistics)
    for repetition in range(repetitions):
        seed_indices = rng.integers(0, pilot_seed_count, size=seed_count)
        session_indices = rng.integers(0, session_count, size=session_count)
        for metric_index, metric in enumerate(metric_names):
            sampled = centered[metric][np.ix_(seed_indices, session_indices)]
            null_statistics[repetition, metric_index] = statistic(sampled)
            alternative_statistics[repetition, metric_index] = statistic(sampled + effect)
    family_null = np.max(null_statistics[:calibration_count], axis=1)
    critical_value = float(np.percentile(family_null, 100.0 * (1.0 - alpha)))
    evaluation_null = null_statistics[calibration_count:]
    evaluation_alternative = alternative_statistics[calibration_count:]
    null_fwer = float(np.mean(np.max(evaluation_null, axis=1) > critical_value))
    metric_power = {
        metric: float(np.mean(evaluation_alternative[:, index] > critical_value))
        for index, metric in enumerate(metric_names)
    }
    monte_carlo_tolerance = 3.0 * math.sqrt(
        max(alpha * (1.0 - alpha) / max(evaluation_count, 1), 0.0)
    )
    return {
        "seed_count": seed_count,
        "session_count": session_count,
        "critical_value": critical_value,
        "heldout_null_fwer": null_fwer,
        "null_fwer_tolerance": alpha + monte_carlo_tolerance,
        "metric_power": metric_power,
        "minimum_metric_power": min(metric_power.values()),
        "calibration_simulations": calibration_count,
        "evaluation_simulations": evaluation_count,
    }


def main() -> None:
    args = parse_args()
    matrices, pilot = load_pilot(args.pilot_matrix)
    candidate_ids = args.candidate_seed_ids
    tail_resolution_minimum = math.ceil(
        1.0 - math.log2(args.family_alpha / len(PDM_DEFAULT_METRICS))
    )
    generator = np.random.default_rng(args.simulation_seed)
    audits = []
    selected: dict[str, Any] | None = None
    for seed_count in range(tail_resolution_minimum, len(candidate_ids) + 1):
        audit = simulate_design(
            matrices,
            seed_count=seed_count,
            effect=args.minimum_relevant_effect,
            alpha=args.family_alpha,
            repetitions=args.simulation_repetitions,
            rng=generator,
        )
        audits.append(audit)
        if (
            audit["minimum_metric_power"] >= args.target_power
            and audit["heldout_null_fwer"] <= audit["null_fwer_tolerance"]
        ):
            selected = audit
            break
    status = "passed" if selected is not None else "failed"
    designed_seed_ids = (
        list(candidate_ids[: int(selected["seed_count"])]) if selected is not None else []
    )
    payload = {
        "schema": SEED_DESIGN_SCHEMA,
        "status": status,
        "design_split": "audit",
        "designed_seed_ids": designed_seed_ids,
        "minimum_relevant_effect": args.minimum_relevant_effect,
        "target_power": args.target_power,
        "estimated_power": (
            selected["minimum_metric_power"] if selected is not None else 0.0
        ),
        "family_alpha": args.family_alpha,
        "family_metric_count": len(PDM_DEFAULT_METRICS),
        "minimum_seed_count_for_two_sided_family_tail_resolution": tail_resolution_minimum,
        "tail_resolution_rule": "2^(1-n_seed) <= family_alpha / family_metric_count",
        "power_simulation_repetitions": args.simulation_repetitions,
        "simulation_seed": args.simulation_seed,
        "calibration_status": "passed" if selected is not None else "failed",
        "calibration_method": (
            "heldout_null_crossed_resampling_max_statistic_over_seven_metrics"
        ),
        "power_method": "audit_only_crossed_resampling_at_minimum_relevant_effect",
        "pilot_matrix": str(args.pilot_matrix.resolve()),
        "pilot_matrix_sha256": sha256(args.pilot_matrix),
        "audit_source_receipt_sha256": pilot["source_receipt_sha256"],
        "candidate_seed_ids": list(candidate_ids),
        "candidate_seed_audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

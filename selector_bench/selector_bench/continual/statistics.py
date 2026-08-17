"""Log-clustered statistics and continual-learning matrix metrics."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import numpy as np

from selector_bench.continual.navsim_protocol import session_id_from_log


class StatisticsError(ValueError):
    pass


def read_pdm_rows(
    path: str | Path,
    *,
    expected_tokens: Sequence[str] | set[str] | None = None,
    require_all_valid: bool = True,
) -> dict[str, dict[str, float]]:
    """Read a PDM CSV and fail closed on invalid, duplicate, or missing rows."""

    rows: dict[str, dict[str, float]] = {}
    observed: set[str] = set()
    invalid: list[str] = []
    with Path(path).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            token = (raw.get("token") or "").strip()
            if not token or token == "average":
                continue
            if token in observed:
                raise StatisticsError(f"duplicate PDM token in {path}: {token}")
            observed.add(token)
            if (raw.get("valid") or "").strip().lower() not in {"true", "1"}:
                invalid.append(token)
                continue
            values: dict[str, float] = {}
            for key, raw_value in raw.items():
                if key in {None, "", "token", "valid"} or not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values[key] = value
            rows[token] = values
    if require_all_valid and invalid:
        raise StatisticsError(f"{len(invalid)} invalid PDM rows in {path}: {invalid[:3]}")
    if expected_tokens is not None:
        expected = set(expected_tokens)
        if observed != expected:
            raise StatisticsError(
                f"PDM token mismatch in {path}: "
                f"missing={len(expected-observed)} extra={len(observed-expected)}"
            )
        if require_all_valid and set(rows) != expected:
            raise StatisticsError(f"PDM result is incomplete after validity checks: {path}")
    if not rows:
        raise StatisticsError(f"no valid PDM rows in {path}")
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def holm_adjusted_pvalues(
    pvalues: Mapping[Hashable, float],
) -> dict[Hashable, float]:
    """Holm step-down family-wise-error correction."""

    if not pvalues:
        return {}
    ordered = sorted(pvalues, key=lambda key: (float(pvalues[key]), str(key)))
    count = len(ordered)
    result: dict[Hashable, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        value = float(pvalues[key])
        if not 0.0 <= value <= 1.0:
            raise StatisticsError(f"invalid p-value for {key!r}: {value}")
        running = max(running, min(1.0, (count - rank) * value))
        result[key] = running
    return result


def paired_cluster_bootstrap(
    rows_a: Mapping[str, Mapping[str, float]],
    rows_b: Mapping[str, Mapping[str, float]],
    token_to_cluster: Mapping[str, str],
    *,
    metrics: Sequence[str],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap complete clusters; report B minus A."""

    common = sorted(set(rows_a) & set(rows_b) & set(token_to_cluster))
    if not common:
        raise StatisticsError("paired bootstrap has no common protocol tokens")
    clusters: dict[str, list[str]] = defaultdict(list)
    for token in common:
        clusters[token_to_cluster[token]].append(token)
    cluster_names = sorted(clusters)
    if len(cluster_names) < 2:
        raise StatisticsError("paired bootstrap needs at least two complete clusters")
    rng = random.Random(seed)
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        differences = {
            log: float(
                np.mean(
                    [
                        rows_b[token][metric] - rows_a[token][metric]
                        for token in tokens
                        if metric in rows_a[token] and metric in rows_b[token]
                    ]
                )
            )
            for log, tokens in clusters.items()
            if any(metric in rows_a[token] and metric in rows_b[token] for token in tokens)
        }
        names = sorted(differences)
        if len(names) < 2:
            continue
        samples = [
            float(np.mean([differences[rng.choice(names)] for _ in names]))
            for _ in range(repetitions)
        ]
        values = np.asarray(samples)
        probability_greater = float(np.mean(values > 0.0))
        lower_tail = (1.0 + float(np.sum(values <= 0.0))) / (repetitions + 1.0)
        upper_tail = (1.0 + float(np.sum(values >= 0.0))) / (repetitions + 1.0)
        point = float(np.mean([differences[name] for name in names]))
        result[metric] = {
            "difference_b_minus_a": point,
            "ci95_low": _percentile(samples, 2.5),
            "ci95_high": _percentile(samples, 97.5),
            "probability_b_greater_a": probability_greater,
            "bootstrap_two_sided_pvalue": min(1.0, 2.0 * min(lower_tail, upper_tail)),
            "paired_token_count": float(len(common)),
            "cluster_count": float(len(names)),
        }
    adjusted = holm_adjusted_pvalues(
        {
            metric: values["bootstrap_two_sided_pvalue"]
            for metric, values in result.items()
        }
    )
    for metric, value in adjusted.items():
        result[metric]["holm_adjusted_pvalue"] = value
    return result


def paired_session_bootstrap(
    rows_a: Mapping[str, Mapping[str, float]],
    rows_b: Mapping[str, Mapping[str, float]],
    token_to_log: Mapping[str, str],
    *,
    metrics: Sequence[str],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Pair results by complete timestamp/vehicle capture session."""

    token_to_session = {
        token: session_id_from_log(log_name)
        for token, log_name in token_to_log.items()
    }
    result = paired_cluster_bootstrap(
        rows_a,
        rows_b,
        token_to_session,
        metrics=metrics,
        repetitions=repetitions,
        seed=seed,
    )
    for values in result.values():
        values["session_cluster_count"] = values.pop("cluster_count")
    return result


def paired_log_bootstrap(
    rows_a: Mapping[str, Mapping[str, float]],
    rows_b: Mapping[str, Mapping[str, float]],
    token_to_log: Mapping[str, str],
    *,
    metrics: Sequence[str],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compatibility wrapper; new paper results must use session bootstrap."""

    result = paired_cluster_bootstrap(
        rows_a,
        rows_b,
        token_to_log,
        metrics=metrics,
        repetitions=repetitions,
        seed=seed,
    )
    for values in result.values():
        values["log_cluster_count"] = values.pop("cluster_count")
    return result


def hierarchical_seed_session_bootstrap(
    seed_rows_a: Mapping[int, Mapping[str, Mapping[str, float]]],
    seed_rows_b: Mapping[int, Mapping[str, Mapping[str, float]]],
    token_to_log: Mapping[str, str],
    *,
    metrics: Sequence[str],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Resample training seeds, then paired capture sessions within each seed."""

    seed_ids = sorted(set(seed_rows_a) & set(seed_rows_b))
    if len(seed_ids) < 2:
        raise StatisticsError("hierarchical bootstrap needs at least two training seeds")
    token_to_session = {
        token: session_id_from_log(log_name)
        for token, log_name in token_to_log.items()
    }
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed_id in seed_ids:
        common = sorted(
            set(seed_rows_a[seed_id])
            & set(seed_rows_b[seed_id])
            & set(token_to_session)
        )
        clusters: dict[str, list[str]] = defaultdict(list)
        for token in common:
            clusters[token_to_session[token]].append(token)
        per_seed[seed_id] = {}
        for metric in metrics:
            differences = {
                session: float(
                    np.mean(
                        [
                            seed_rows_b[seed_id][token][metric]
                            - seed_rows_a[seed_id][token][metric]
                            for token in tokens
                            if metric in seed_rows_a[seed_id][token]
                            and metric in seed_rows_b[seed_id][token]
                        ]
                    )
                )
                for session, tokens in clusters.items()
                if any(
                    metric in seed_rows_a[seed_id][token]
                    and metric in seed_rows_b[seed_id][token]
                    for token in tokens
                )
            }
            if len(differences) < 2:
                raise StatisticsError(
                    f"seed {seed_id} has fewer than two sessions for {metric}"
                )
            per_seed[seed_id][metric] = differences

    rng = random.Random(seed)
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        seed_points = [
            float(np.mean(list(per_seed[seed_id][metric].values())))
            for seed_id in seed_ids
        ]
        samples = []
        for _ in range(repetitions):
            sampled_seeds = [rng.choice(seed_ids) for _ in seed_ids]
            sampled_seed_means = []
            for seed_id in sampled_seeds:
                session_values = list(per_seed[seed_id][metric].values())
                sampled_seed_means.append(
                    float(
                        np.mean(
                            [rng.choice(session_values) for _ in session_values]
                        )
                    )
                )
            samples.append(float(np.mean(sampled_seed_means)))
        values = np.asarray(samples)
        lower_tail = (1.0 + float(np.sum(values <= 0.0))) / (repetitions + 1.0)
        upper_tail = (1.0 + float(np.sum(values >= 0.0))) / (repetitions + 1.0)
        result[metric] = {
            "difference_b_minus_a": float(np.mean(seed_points)),
            "ci95_low": _percentile(samples, 2.5),
            "ci95_high": _percentile(samples, 97.5),
            "probability_b_greater_a": float(np.mean(values > 0.0)),
            "bootstrap_two_sided_pvalue": min(1.0, 2.0 * min(lower_tail, upper_tail)),
            "training_seed_count": float(len(seed_ids)),
            "session_cluster_count_min": float(
                min(len(per_seed[item][metric]) for item in seed_ids)
            ),
        }
    adjusted = holm_adjusted_pvalues(
        {
            metric: values["bootstrap_two_sided_pvalue"]
            for metric, values in result.items()
        }
    )
    for metric, value in adjusted.items():
        result[metric]["holm_adjusted_pvalue"] = value
    return result


def continual_matrix_metrics(
    matrix: Sequence[Sequence[float]],
    *,
    untrained_domain_scores: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute final average, BWT, optional FWT, and observed-curve area.

    Row i is the checkpoint after stage i+1; column j is evaluation domain j+1.
    Cells above the observed diagonal may be NaN when future domains are sealed.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise StatisticsError("continual matrix must be square with at least two stages")
    stages = values.shape[0]
    diagonal = np.diag(values)
    if not np.isfinite(diagonal).all() or not np.isfinite(values[-1]).all():
        raise StatisticsError("continual diagonal and final row must be finite")
    bwt = float(np.mean(values[-1, :-1] - diagonal[:-1]))
    per_domain_auc = []
    for domain in range(stages):
        observed = values[domain:, domain]
        observed = observed[np.isfinite(observed)]
        per_domain_auc.append(float(np.mean(observed)))
    result = {
        "final_average": float(np.mean(values[-1])),
        "backward_transfer": bwt,
        "stage_curve_area_mean": float(np.mean(per_domain_auc)),
    }
    if untrained_domain_scores is not None:
        baseline = np.asarray(untrained_domain_scores, dtype=np.float64)
        if baseline.shape != (stages,) or not np.isfinite(baseline).all():
            raise StatisticsError("untrained_domain_scores has the wrong shape")
        forward = [values[domain - 1, domain] - baseline[domain] for domain in range(1, stages)]
        if not np.isfinite(forward).all():
            raise StatisticsError("FWT requires pre-exposure matrix cells")
        result["forward_transfer"] = float(np.mean(forward))
    return result

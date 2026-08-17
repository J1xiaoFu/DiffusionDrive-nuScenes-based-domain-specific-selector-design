"""Log-clustered statistics and continual-learning matrix metrics."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from selector_bench.continual.navsim_protocol import session_id_from_log


class StatisticsError(ValueError):
    pass


def read_pdm_rows(path: str | Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with Path(path).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            token = (raw.get("token") or "").strip()
            if not token or token == "average" or (raw.get("valid") or "").lower() not in {
                "true",
                "1",
            }:
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
    if not rows:
        raise StatisticsError(f"no valid PDM rows in {path}")
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def paired_log_bootstrap(
    rows_a: Mapping[str, Mapping[str, float]],
    rows_b: Mapping[str, Mapping[str, float]],
    token_to_log: Mapping[str, str],
    *,
    metrics: Sequence[str],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap complete log clusters; report B minus A."""

    common = sorted(set(rows_a) & set(rows_b) & set(token_to_log))
    if not common:
        raise StatisticsError("paired bootstrap has no common protocol tokens")
    clusters: dict[str, list[str]] = defaultdict(list)
    for token in common:
        clusters[token_to_log[token]].append(token)
    cluster_names = sorted(clusters)
    if len(cluster_names) < 2:
        raise StatisticsError("paired bootstrap needs at least two complete logs")
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
        point = float(np.mean([differences[name] for name in names]))
        result[metric] = {
            "difference_b_minus_a": point,
            "ci95_low": _percentile(samples, 2.5),
            "ci95_high": _percentile(samples, 97.5),
            "probability_b_greater_a": float(np.mean(np.asarray(samples) > 0.0)),
            "paired_token_count": float(len(common)),
            "log_cluster_count": float(len(names)),
        }
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

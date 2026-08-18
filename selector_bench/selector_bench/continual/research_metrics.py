"""Paper-facing scientific summaries for FTF-OPD experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from selector_bench.continual.statistics import StatisticsError, continual_matrix_metrics


def continual_driving_summary(
    matrix: Sequence[Sequence[float]],
    *,
    untrained_domain_scores: Sequence[float] | None = None,
) -> dict[str, float]:
    """Summarize stability, plasticity and worst-case old-domain behavior."""

    values = np.asarray(matrix, dtype=np.float64)
    result = continual_matrix_metrics(
        matrix, untrained_domain_scores=untrained_domain_scores
    )
    diagonal = np.diag(values)
    final = values[-1]
    old_final = final[:-1]
    forgetting = [
        float(np.nanmax(values[domain:, domain]) - final[domain])
        for domain in range(values.shape[0] - 1)
    ]
    result.update(
        {
            "final_old_stage_average": float(np.mean(old_final)),
            "final_worst_old_stage": float(np.min(old_final)),
            "mean_new_stage_plasticity": float(np.mean(diagonal)),
            "last_stage_plasticity": float(diagonal[-1]),
            "mean_old_stage_forgetting": float(np.mean(forgetting)),
            "stability_coordinate": float(np.mean(old_final)),
            "plasticity_coordinate": float(diagonal[-1]),
        }
    )
    return result


def trigger_quality(
    audit_epochs: Sequence[float],
    predicted: Sequence[bool],
    oracle: Sequence[bool],
) -> dict[str, float | int | None]:
    """Measure trigger precision/recall and first-detection delay per oracle episode."""

    epochs = np.asarray(audit_epochs, dtype=np.float64)
    prediction = np.asarray(predicted)
    truth = np.asarray(oracle)
    if (
        epochs.ndim != 1
        or epochs.size == 0
        or prediction.shape != epochs.shape
        or truth.shape != epochs.shape
        or prediction.dtype != np.bool_
        or truth.dtype != np.bool_
        or not np.isfinite(epochs).all()
        or np.any(np.diff(epochs) <= 0.0)
    ):
        raise StatisticsError("trigger series must be aligned booleans at increasing finite epochs")
    true_positive = int(np.sum(prediction & truth))
    false_positive = int(np.sum(prediction & ~truth))
    false_negative = int(np.sum(~prediction & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    starts = np.flatnonzero(truth & np.concatenate(([True], ~truth[:-1])))
    delays: list[float] = []
    missed = 0
    for start in starts:
        episode_end = np.flatnonzero(~truth[start:])
        stop = start + int(episode_end[0]) if episode_end.size else len(truth)
        detected = np.flatnonzero(prediction[start:stop])
        if detected.size:
            index = start + int(detected[0])
            delays.append(float(epochs[index] - epochs[start]))
        else:
            missed += 1
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)),
        "true_positive_audits": true_positive,
        "false_positive_audits": false_positive,
        "false_negative_audits": false_negative,
        "oracle_episode_count": int(len(starts)),
        "missed_oracle_episodes": missed,
        "mean_detection_delay_epoch": float(np.mean(delays)) if delays else None,
    }


def support_condition_curve(
    coverage: Sequence[float],
    checkpoint_score: Sequence[float],
    replay_score: Sequence[float],
    *,
    bin_edges: Sequence[float] = (0.0, 0.5, 0.75, 0.9, 1.0000001),
    noninferiority_margin: float = 0.02,
    bootstrap_repetitions: int = 2000,
    seed: int = 0,
) -> list[dict[str, float | int | bool]]:
    """Estimate checkpoint-minus-replay gaps over predeclared support bins."""

    support = np.asarray(coverage, dtype=np.float64)
    checkpoint = np.asarray(checkpoint_score, dtype=np.float64)
    replay = np.asarray(replay_score, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if (
        support.ndim != 1
        or support.size == 0
        or checkpoint.shape != support.shape
        or replay.shape != support.shape
        or not np.isfinite(support).all()
        or not np.isfinite(checkpoint).all()
        or not np.isfinite(replay).all()
        or np.any((support < 0.0) | (support > 1.0))
        or edges.ndim != 1
        or len(edges) < 2
        or np.any(np.diff(edges) <= 0.0)
        or bootstrap_repetitions <= 0
        or noninferiority_margin < 0.0
    ):
        raise StatisticsError("invalid support-condition inputs")
    rng = np.random.default_rng(seed)
    gap = checkpoint - replay
    result: list[dict[str, float | int | bool]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (support >= low) & (support < high)
        values = gap[mask]
        if values.size == 0:
            continue
        sampled = values[
            rng.integers(0, values.size, size=(bootstrap_repetitions, values.size))
        ].mean(axis=1)
        lower = float(np.quantile(sampled, 0.025))
        result.append(
            {
                "bin_index": index,
                "coverage_low": float(low),
                "coverage_high": float(min(high, 1.0)),
                "count": int(values.size),
                "checkpoint_minus_replay_mean": float(np.mean(values)),
                "ci95_low": lower,
                "ci95_high": float(np.quantile(sampled, 0.975)),
                "noninferior_at_margin": bool(lower >= -noninferiority_margin),
            }
        )
    return result


def pareto_front(
    points: Mapping[str, tuple[float, float]],
) -> tuple[str, ...]:
    """Return methods not dominated on (stability, plasticity), both larger better."""

    if not points:
        raise StatisticsError("Pareto analysis needs at least one method")
    normalized: dict[str, tuple[float, float]] = {}
    for method, point in points.items():
        array = np.asarray(point, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise StatisticsError("Pareto coordinates must be finite pairs")
        normalized[str(method)] = (float(array[0]), float(array[1]))
    front = []
    for method, point in normalized.items():
        dominated = any(
            other != method
            and other_point[0] >= point[0]
            and other_point[1] >= point[1]
            and other_point != point
            for other, other_point in normalized.items()
        )
        if not dominated:
            front.append(method)
    return tuple(sorted(front))

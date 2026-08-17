"""Log-clustered statistics and continual-learning matrix metrics."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from selector_bench.continual.navsim_protocol import session_id_from_log


class StatisticsError(ValueError):
    pass


PDM_DEFAULT_METRICS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
    "driving_direction_compliance",
)

PDM_METADATA_FIELDS = ("token", "log", "valid")


def read_pdm_rows(
    path: str | Path,
    *,
    expected_tokens: Sequence[str] | set[str] | None = None,
    require_all_valid: bool = True,
    required_metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Read a PDM CSV and fail closed on invalid rows or metric cells.

    ``required_metrics`` is mandatory for claim-bearing callers.  It remains
    optional here only so small diagnostic tables can infer every non-key
    column as a metric; inferred columns are held to the same numeric/finite
    contract.
    """

    rows: dict[str, dict[str, float]] = {}
    observed: set[str] = set()
    invalid: list[str] = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise StatisticsError(f"PDM CSV has no header: {path}")
        normalized_fields = [str(value).strip() for value in fieldnames]
        if normalized_fields != fieldnames:
            raise StatisticsError(f"PDM CSV header contains surrounding whitespace: {path}")
        if len(normalized_fields) != len(set(normalized_fields)):
            raise StatisticsError(f"PDM CSV has duplicate header fields: {path}")
        if not {"token", "valid"}.issubset(normalized_fields):
            raise StatisticsError(f"PDM CSV must contain token and valid fields: {path}")
        metrics = tuple(
            str(value).strip()
            for value in (
                required_metrics
                if required_metrics is not None
                else [
                    value
                    for value in normalized_fields
                    if value not in PDM_METADATA_FIELDS
                ]
            )
        )
        if not metrics or any(not value for value in metrics) or len(metrics) != len(set(metrics)):
            raise StatisticsError("required PDM metrics must be non-empty and unique")
        missing_columns = sorted(set(metrics) - set(normalized_fields))
        if missing_columns:
            raise StatisticsError(
                f"PDM CSV is missing required metric columns in {path}: {missing_columns}"
            )
        unknown_columns = sorted(
            set(normalized_fields) - set(PDM_METADATA_FIELDS) - set(metrics)
        )
        if unknown_columns:
            raise StatisticsError(
                f"PDM CSV has unregistered columns in {path}: {unknown_columns}"
            )
        for row_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise StatisticsError(f"PDM CSV has excess cells at {path}:{row_number}")
            token = (raw.get("token") or "").strip()
            if token == "average":
                continue
            if not token:
                raise StatisticsError(f"blank PDM token at {path}:{row_number}")
            if token in observed:
                raise StatisticsError(f"duplicate PDM token in {path}: {token}")
            observed.add(token)
            if (raw.get("valid") or "").strip().lower() not in {"true", "1"}:
                invalid.append(token)
                continue
            values: dict[str, float] = {}
            for key in metrics:
                raw_value = (raw.get(key) or "").strip()
                if not raw_value:
                    raise StatisticsError(
                        f"missing required metric {key!r} for token {token} in {path}"
                    )
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise StatisticsError(
                        f"non-numeric required metric {key!r}={raw_value!r} "
                        f"for token {token} in {path}"
                    ) from exc
                if not math.isfinite(value):
                    raise StatisticsError(
                        f"non-finite required metric {key!r}={raw_value!r} "
                        f"for token {token} in {path}"
                    )
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


def holm_rejection_matrix(pvalues: np.ndarray, *, alpha: float) -> np.ndarray:
    """Apply the Holm step-down rejection rule to each complete p-value row."""

    matrix = np.asarray(pvalues, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] < 1
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise StatisticsError("Holm p-value matrix must be finite, two-dimensional and in [0, 1]")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise StatisticsError("family-wise alpha must be a finite real number")
    family_alpha = float(alpha)
    if not math.isfinite(family_alpha) or not 0.0 < family_alpha < 1.0:
        raise StatisticsError("family-wise alpha must lie strictly between zero and one")
    order = np.argsort(matrix, axis=1, kind="stable")
    ordered = np.take_along_axis(matrix, order, axis=1)
    thresholds = family_alpha / (
        matrix.shape[1] - np.arange(matrix.shape[1], dtype=np.float64)
    )
    ordered_rejections = np.logical_and.accumulate(ordered <= thresholds, axis=1)
    result = np.zeros_like(ordered_rejections, dtype=bool)
    np.put_along_axis(result, order, ordered_rejections, axis=1)
    return result


def complete_holm_family(
    pvalues: Mapping[str, float],
    *,
    expected_hypothesis_ids: Sequence[str],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Apply Holm once, after proving the preregistered family is complete."""

    expected = tuple(str(value) for value in expected_hypothesis_ids)
    if not expected or any(not value for value in expected) or len(expected) != len(set(expected)):
        raise StatisticsError("expected hypothesis IDs must be non-empty and unique")
    if set(pvalues) != set(expected):
        raise StatisticsError(
            "observed p-value IDs must exactly equal the preregistered global family"
        )
    if not 0.0 < float(alpha) < 1.0:
        raise StatisticsError("family-wise alpha must lie strictly between zero and one")
    adjusted = holm_adjusted_pvalues(pvalues)
    raw_matrix = np.asarray(
        [[float(pvalues[hypothesis_id]) for hypothesis_id in expected]],
        dtype=np.float64,
    )
    rejected = holm_rejection_matrix(raw_matrix, alpha=alpha)[0]
    return {
        hypothesis_id: {
            "raw_pvalue": float(pvalues[hypothesis_id]),
            "holm_adjusted_pvalue": float(adjusted[hypothesis_id]),
            "reject_global_null": bool(rejected[index]),
        }
        for index, hypothesis_id in enumerate(expected)
    }


def _validate_paired_inputs(
    rows_a: Mapping[str, Mapping[str, float]],
    rows_b: Mapping[str, Mapping[str, float]],
    token_to_cluster: Mapping[str, str],
    metrics: Sequence[str],
) -> tuple[list[str], tuple[str, ...]]:
    expected = set(token_to_cluster)
    if not expected:
        raise StatisticsError("paired comparison has no preregistered protocol tokens")
    if set(rows_a) != expected or set(rows_b) != expected:
        raise StatisticsError(
            "paired token sets must exactly equal the preregistered token mapping"
        )
    metric_names = tuple(str(value) for value in metrics)
    if not metric_names or len(metric_names) != len(set(metric_names)):
        raise StatisticsError("paired metrics must be non-empty and unique")
    expected_metric_set = set(metric_names)
    for label, rows in (("A", rows_a), ("B", rows_b)):
        for token in expected:
            observed_metric_set = set(rows[token])
            if observed_metric_set != expected_metric_set:
                raise StatisticsError(
                    f"method {label} token {token} metric set differs from the "
                    f"preregistered family: missing={sorted(expected_metric_set-observed_metric_set)} "
                    f"extra={sorted(observed_metric_set-expected_metric_set)}"
                )
            nonfinite = [
                metric
                for metric in metric_names
                if not math.isfinite(float(rows[token][metric]))
            ]
            if nonfinite:
                raise StatisticsError(
                    f"method {label} token {token} has non-finite metrics: {nonfinite}"
                )
    return sorted(expected), metric_names


def _studentized_statistic(values: np.ndarray) -> float:
    point = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    if standard_error <= np.finfo(np.float64).eps:
        if abs(point) <= np.finfo(np.float64).eps:
            return 0.0
        return math.copysign(math.inf, point)
    return point / standard_error


def _one_way_cluster_inference(
    differences: Sequence[float], *, repetitions: int, seed: int
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise StatisticsError("cluster inference needs at least two finite clusters")
    if repetitions <= 0:
        raise StatisticsError("bootstrap repetitions must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    bootstrap = values[indices].mean(axis=1)
    point = float(values.mean())

    centered = values - point
    observed_statistic = abs(_studentized_statistic(values))
    null_statistics = np.asarray(
        [abs(_studentized_statistic(centered[index])) for index in indices],
        dtype=np.float64,
    )
    exceedances = int(np.sum(null_statistics >= observed_statistic))
    pvalue = (1.0 + exceedances) / (repetitions + 1.0)
    return {
        "difference_b_minus_a": point,
        "ci95_low": _percentile(bootstrap, 2.5),
        "ci95_high": _percentile(bootstrap, 97.5),
        "probability_b_greater_a": float(np.mean(bootstrap > 0.0)),
        "null_centered_studentized_pvalue": float(pvalue),
    }


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

    common, metric_names = _validate_paired_inputs(
        rows_a, rows_b, token_to_cluster, metrics
    )
    clusters: dict[str, list[str]] = defaultdict(list)
    for token in common:
        clusters[token_to_cluster[token]].append(token)
    cluster_names = sorted(clusters)
    if len(cluster_names) < 2:
        raise StatisticsError("paired bootstrap needs at least two complete clusters")
    result: dict[str, dict[str, float]] = {}
    for metric_index, metric in enumerate(metric_names):
        differences = {
            log: float(
                np.mean(
                    [
                        rows_b[token][metric] - rows_a[token][metric]
                        for token in tokens
                    ]
                )
            )
            for log, tokens in clusters.items()
        }
        names = sorted(differences)
        if len(names) < 2:
            raise StatisticsError(f"paired comparison has fewer than two clusters for {metric}")
        result[metric] = _one_way_cluster_inference(
            [differences[name] for name in names],
            repetitions=repetitions,
            seed=seed + metric_index,
        )
        result[metric].update(
            {
                "paired_token_count": float(len(common)),
                "cluster_count": float(len(names)),
            }
        )
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


def _crossed_standard_error(matrix: np.ndarray) -> float:
    """Non-negative random-effects SE for a balanced seed x session matrix."""

    seed_count, session_count = matrix.shape
    if seed_count < 2 or session_count < 2:
        raise StatisticsError("crossed inference needs at least two seeds and sessions")
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
    variance = (
        seed_component / seed_count
        + session_component / session_count
        + residual_variance / (seed_count * session_count)
    )
    return math.sqrt(max(variance, 0.0))


def _crossed_studentized_statistic(matrix: np.ndarray) -> float:
    point = float(matrix.mean())
    standard_error = _crossed_standard_error(matrix)
    if standard_error <= np.finfo(np.float64).eps:
        if abs(point) <= np.finfo(np.float64).eps:
            return 0.0
        return math.copysign(math.inf, point)
    return point / standard_error


def _crossed_studentized_statistic_batch(matrix: np.ndarray) -> np.ndarray:
    """Vectorized counterpart of ``_crossed_studentized_statistic``."""

    if matrix.ndim != 3 or matrix.shape[1] < 2 or matrix.shape[2] < 2:
        raise StatisticsError("crossed statistic batch requires [draw, seed, session]")
    if not np.all(np.isfinite(matrix)):
        raise StatisticsError("crossed statistic batch contains non-finite values")
    _, seed_count, session_count = matrix.shape
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
    standard_error = np.sqrt(
        np.maximum(
            seed_component / seed_count
            + session_component / session_count
            + residual_variance / (seed_count * session_count),
            0.0,
        )
    )
    epsilon = np.finfo(np.float64).eps
    result = np.empty_like(grand)
    regular = standard_error > epsilon
    np.divide(np.abs(grand), standard_error, out=result, where=regular)
    degenerate_nonzero = ~regular & (np.abs(grand) > epsilon)
    result[degenerate_nonzero] = math.inf
    result[~regular & ~degenerate_nonzero] = 0.0
    return result


def _resample_count_weights(indices: np.ndarray, category_count: int) -> np.ndarray:
    """Convert frozen resampling indices to normalized multinomial weights."""

    draws = np.asarray(indices)
    if (
        draws.ndim != 2
        or draws.shape[1] != category_count
        or np.any(draws < 0)
        or np.any(draws >= category_count)
    ):
        raise StatisticsError("crossed bootstrap resampling indices are malformed")
    counts = np.zeros((draws.shape[0], category_count), dtype=np.float64)
    rows = np.repeat(np.arange(draws.shape[0]), category_count)
    np.add.at(counts, (rows, draws.reshape(-1)), 1.0)
    return counts / float(category_count)


def _crossed_statistics_from_weighted_resamples(
    matrices: np.ndarray,
    seed_weights: np.ndarray,
    session_weights: np.ndarray,
) -> np.ndarray:
    """Studentized statistics without materializing [draw, R, seed, session]."""

    values = np.asarray(matrices, dtype=np.float64)
    if (
        values.ndim != 3
        or seed_weights.ndim != 2
        or session_weights.ndim != 2
        or seed_weights.shape[0] != session_weights.shape[0]
        or seed_weights.shape[1] != values.shape[1]
        or session_weights.shape[1] != values.shape[2]
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(seed_weights))
        or not np.all(np.isfinite(session_weights))
    ):
        raise StatisticsError("crossed bootstrap weighted-resample inputs are malformed")
    draw_count, seed_count, session_count = values.shape
    repetitions = seed_weights.shape[0]
    row_means = np.einsum(
        "rt,dst->drs", session_weights, values, optimize=True
    )
    column_means = np.einsum(
        "rs,dst->drt", seed_weights, values, optimize=True
    )
    grand = np.einsum("rs,drs->dr", seed_weights, row_means, optimize=True)
    row_sum_squares = seed_count * np.einsum(
        "rs,drs->dr",
        seed_weights,
        np.square(row_means - grand[:, :, None]),
        optimize=True,
    )
    column_sum_squares = session_count * np.einsum(
        "rt,drt->dr",
        session_weights,
        np.square(column_means - grand[:, :, None]),
        optimize=True,
    )
    row_second_moments = np.einsum(
        "rt,dst->drs", session_weights, np.square(values), optimize=True
    )
    total_second_moment = np.einsum(
        "rs,drs->dr", seed_weights, row_second_moments, optimize=True
    )
    total_sum_squares = seed_count * session_count * np.maximum(
        total_second_moment - np.square(grand), 0.0
    )
    residual_sum_squares = np.maximum(
        total_sum_squares
        - session_count * row_sum_squares
        - seed_count * column_sum_squares,
        0.0,
    )
    residual_variance = residual_sum_squares / (
        (seed_count - 1) * (session_count - 1)
    )
    row_variance = row_sum_squares / (seed_count - 1)
    column_variance = column_sum_squares / (session_count - 1)
    seed_component = np.maximum(
        0.0, row_variance - residual_variance / session_count
    )
    session_component = np.maximum(
        0.0, column_variance - residual_variance / seed_count
    )
    standard_error = np.sqrt(
        np.maximum(
            seed_component / seed_count
            + session_component / session_count
            + residual_variance / (seed_count * session_count),
            0.0,
        )
    )
    epsilon = np.finfo(np.float64).eps
    result = np.empty((draw_count, repetitions), dtype=np.float64)
    regular = standard_error > epsilon
    np.divide(np.abs(grand), standard_error, out=result, where=regular)
    degenerate_nonzero = ~regular & (np.abs(grand) > epsilon)
    result[degenerate_nonzero] = math.inf
    result[~regular & ~degenerate_nonzero] = 0.0
    return result


def crossed_matrix_bootstrap_shift_pvalues_batch(
    matrices: np.ndarray,
    *,
    constant_shifts: Sequence[float],
    repetitions: int,
    seed: int,
    matrix_batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Run production conditional p-values for constant-shift alternatives.

    Each input matrix is centered separately, exactly as in the paper comparison
    path. Constant shifts leave that conditional null invariant, so one null
    resample supplies null/positive/negative power draws without changing the
    per-observation production test.
    """

    values = np.asarray(matrices, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1] < 2
        or values.shape[2] < 2
        or not np.all(np.isfinite(values))
    ):
        raise StatisticsError(
            "crossed matrix batch inference requires finite [draw, seed, session]"
        )
    raw_shifts = tuple(constant_shifts)
    if not raw_shifts:
        raise StatisticsError("crossed bootstrap requires a non-empty shift family")
    for value in raw_shifts:
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int, np.floating, np.integer))
            or not math.isfinite(float(value))
        ):
            raise StatisticsError("crossed bootstrap shifts must be finite reals")
    shifts = np.asarray(raw_shifts, dtype=np.float64)
    if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)):
        raise StatisticsError("crossed bootstrap repetitions must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise StatisticsError("crossed bootstrap seed must be an integer")
    if isinstance(matrix_batch_size, bool) or not isinstance(
        matrix_batch_size, (int, np.integer)
    ):
        raise StatisticsError("crossed bootstrap matrix batch size must be an integer")
    repetitions = int(repetitions)
    matrix_batch_size = int(matrix_batch_size)
    if repetitions <= 0 or matrix_batch_size <= 0:
        raise StatisticsError(
            "crossed bootstrap repetitions and matrix batch size must be positive"
        )
    generator = np.random.default_rng(int(seed))
    seed_indices = generator.integers(
        0, values.shape[1], size=(repetitions, values.shape[1])
    )
    session_indices = generator.integers(
        0, values.shape[2], size=(repetitions, values.shape[2])
    )
    seed_weights = _resample_count_weights(seed_indices, values.shape[1])
    session_weights = _resample_count_weights(session_indices, values.shape[2])
    pvalues = np.empty((values.shape[0], len(shifts)), dtype=np.float64)
    observed_statistics = np.empty_like(pvalues)
    for start in range(0, values.shape[0], matrix_batch_size):
        stop = min(start + matrix_batch_size, values.shape[0])
        observed = values[start:stop]
        centered = observed - observed.mean(axis=(1, 2), keepdims=True)
        null_statistics = _crossed_statistics_from_weighted_resamples(
            centered, seed_weights, session_weights
        )
        shifted = observed[:, None, :, :] + shifts[None, :, None, None]
        observed_batch = _crossed_studentized_statistic_batch(
            shifted.reshape(
                (stop - start) * len(shifts),
                values.shape[1],
                values.shape[2],
            )
        ).reshape(stop - start, len(shifts))
        for shift_index in range(len(shifts)):
            exceedances = np.sum(
                null_statistics >= observed_batch[:, shift_index, None],
                axis=1,
                dtype=np.int64,
            )
            pvalues[start:stop, shift_index] = (
                1.0 + exceedances.astype(np.float64)
            ) / (repetitions + 1.0)
        observed_statistics[start:stop] = observed_batch
    result = {
        "pvalues": pvalues,
        "observed_statistics": observed_statistics,
    }
    return result


def crossed_matrix_bootstrap_pvalues_batch(
    matrices: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    matrix_batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Run the exact production conditional crossed bootstrap in a batch."""

    shifted = crossed_matrix_bootstrap_shift_pvalues_batch(
        matrices,
        constant_shifts=(0.0,),
        repetitions=repetitions,
        seed=seed,
        matrix_batch_size=matrix_batch_size,
    )
    return {
        "pvalues": shifted["pvalues"][:, 0],
        "observed_statistics": shifted["observed_statistics"][:, 0],
    }


def crossed_matrix_bootstrap_inference(
    matrix: np.ndarray, *, repetitions: int, seed: int
) -> dict[str, object]:
    """Exact shared matrix kernel for production crossed-bootstrap p-values."""

    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 2
        or not np.all(np.isfinite(values))
    ):
        raise StatisticsError("crossed matrix inference requires a finite 2D matrix")
    if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)):
        raise StatisticsError("crossed bootstrap repetitions must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise StatisticsError("crossed bootstrap seed must be an integer")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise StatisticsError("crossed bootstrap repetitions must be positive")
    generator = np.random.default_rng(int(seed))
    seed_indices = generator.integers(
        0, values.shape[0], size=(repetitions, values.shape[0])
    )
    session_indices = generator.integers(
        0, values.shape[1], size=(repetitions, values.shape[1])
    )
    sampled = values[seed_indices[:, :, None], session_indices[:, None, :]]
    bootstrap = sampled.mean(axis=(1, 2))
    conditional = crossed_matrix_bootstrap_pvalues_batch(
        values[None, :, :], repetitions=repetitions, seed=seed
    )
    centered = values - float(values.mean())
    null_sampled = centered[
        seed_indices[:, :, None], session_indices[:, None, :]
    ]
    null_statistics = _crossed_studentized_statistic_batch(null_sampled)
    observed_statistic = float(conditional["observed_statistics"][0])
    return {
        "bootstrap_means": bootstrap,
        "null_statistics": null_statistics,
        "observed_statistic": float(observed_statistic),
        "null_centered_studentized_pvalue": float(conditional["pvalues"][0]),
    }


def crossed_seed_session_bootstrap(
    seed_rows_a: Mapping[int, Mapping[str, Mapping[str, float]]],
    seed_rows_b: Mapping[int, Mapping[str, Mapping[str, float]]],
    token_to_log: Mapping[str, str],
    *,
    metrics: Sequence[str],
    expected_seed_ids: Sequence[int],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Independently resample crossed training seeds and evaluation sessions."""

    expected_seeds = set(int(value) for value in expected_seed_ids)
    if not expected_seeds or len(expected_seeds) != len(tuple(expected_seed_ids)):
        raise StatisticsError("expected seed IDs must be non-empty and unique")
    if set(seed_rows_a) != expected_seeds or set(seed_rows_b) != expected_seeds:
        raise StatisticsError("method seed sets must exactly equal expected seed IDs")
    seed_ids = sorted(expected_seeds)
    if len(seed_ids) < 2:
        raise StatisticsError("crossed bootstrap needs at least two training seeds")
    token_to_session = {
        token: session_id_from_log(log_name)
        for token, log_name in token_to_log.items()
    }
    expected_tokens = set(token_to_session)
    metric_names = tuple(str(value) for value in metrics)
    if not metric_names or len(metric_names) != len(set(metric_names)):
        raise StatisticsError("crossed metrics must be non-empty and unique")
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed_id in seed_ids:
        common, _ = _validate_paired_inputs(
            seed_rows_a[seed_id],
            seed_rows_b[seed_id],
            token_to_session,
            metric_names,
        )
        if set(common) != expected_tokens:
            raise StatisticsError(f"seed {seed_id} token set mismatch")
        clusters: dict[str, list[str]] = defaultdict(list)
        for token in common:
            clusters[token_to_session[token]].append(token)
        per_seed[seed_id] = {}
        for metric in metric_names:
            differences = {
                session: float(
                    np.mean(
                        [
                            seed_rows_b[seed_id][token][metric]
                            - seed_rows_a[seed_id][token][metric]
                            for token in tokens
                        ]
                    )
                )
                for session, tokens in clusters.items()
            }
            if len(differences) < 2:
                raise StatisticsError(
                    f"seed {seed_id} has fewer than two sessions for {metric}"
                )
            per_seed[seed_id][metric] = differences

    session_names = sorted(
        {session for values in per_seed.values() for session in values[metric_names[0]]}
    )
    for seed_id in seed_ids:
        for metric in metric_names:
            if set(per_seed[seed_id][metric]) != set(session_names):
                raise StatisticsError(
                    f"seed {seed_id} does not contain the exact crossed session set for {metric}"
                )

    result: dict[str, dict[str, float]] = {}
    for metric_index, metric in enumerate(metric_names):
        matrix = np.asarray(
            [
                [per_seed[seed_id][metric][session] for session in session_names]
                for seed_id in seed_ids
            ],
            dtype=np.float64,
        )
        point = float(matrix.mean())
        inference = crossed_matrix_bootstrap_inference(
            matrix, repetitions=repetitions, seed=seed + metric_index
        )
        bootstrap = np.asarray(inference["bootstrap_means"], dtype=np.float64)
        result[metric] = {
            "difference_b_minus_a": point,
            "ci95_low": _percentile(bootstrap, 2.5),
            "ci95_high": _percentile(bootstrap, 97.5),
            "probability_b_greater_a": float(np.mean(bootstrap > 0.0)),
            "null_centered_studentized_pvalue": float(
                inference["null_centered_studentized_pvalue"]
            ),
            "training_seed_count": float(len(seed_ids)),
            "session_cluster_count": float(len(session_names)),
            "crossed_seed_session_design": 1.0,
            "minimum_cluster_dimension": float(min(len(seed_ids), len(session_names))),
            "small_seed_inference_caveat": float(len(seed_ids) < 5),
        }
    return result


def hierarchical_seed_session_bootstrap(
    seed_rows_a: Mapping[int, Mapping[str, Mapping[str, float]]],
    seed_rows_b: Mapping[int, Mapping[str, Mapping[str, float]]],
    token_to_log: Mapping[str, str],
    *,
    metrics: Sequence[str],
    expected_seed_ids: Sequence[int],
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compatibility name for the corrected crossed seed/session design."""

    return crossed_seed_session_bootstrap(
        seed_rows_a,
        seed_rows_b,
        token_to_log,
        metrics=metrics,
        expected_seed_ids=expected_seed_ids,
        repetitions=repetitions,
        seed=seed,
    )


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

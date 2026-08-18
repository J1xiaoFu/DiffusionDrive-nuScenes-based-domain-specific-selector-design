"""Explanatory support-overlap and gradient-conflict measurements for Drive-OPD."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class SupportAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class SupportAnalysis:
    old_covered_by_current: float
    current_knn_radius: float
    rbf_mmd_squared: float
    gradient_cosine: float
    gradient_angle_degrees: float


def _matrix(value: np.ndarray, label: str, minimum_rows: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < minimum_rows or array.shape[1] < 1:
        raise SupportAnalysisError(f"{label} must be a 2D matrix with enough rows")
    if not np.isfinite(array).all():
        raise SupportAnalysisError(f"{label} contains non-finite values")
    return array


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = (
        np.square(left).sum(axis=1, keepdims=True)
        + np.square(right).sum(axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distance, 0.0)


def knn_support_coverage(
    old_features: np.ndarray,
    current_features: np.ndarray,
    *,
    k: int = 5,
    radius_quantile: float = 0.95,
) -> tuple[float, float]:
    """Measure how much old support is covered by the current-stage support.

    The radius is the selected quantile of current-to-current leave-one-out
    kNN distances.  Old points are covered when their nearest current point is
    inside that data-scaled radius.
    """

    old = _matrix(old_features, "old features")
    current = _matrix(current_features, "current features")
    if old.shape[1] != current.shape[1]:
        raise SupportAnalysisError("old/current feature dimensions differ")
    if not 1 <= k < current.shape[0]:
        raise SupportAnalysisError("k must be smaller than the current sample count")
    if not 0.5 <= radius_quantile < 1.0:
        raise SupportAnalysisError("radius quantile must lie in [0.5,1)")
    within = _squared_distances(current, current)
    np.fill_diagonal(within, np.inf)
    kth = np.partition(within, k - 1, axis=1)[:, k - 1]
    radius_squared = float(np.quantile(kth, radius_quantile))
    old_nearest = _squared_distances(old, current).min(axis=1)
    coverage = float(np.mean(old_nearest <= radius_squared))
    return coverage, math.sqrt(max(radius_squared, 0.0))


def rbf_mmd_squared(
    old_features: np.ndarray,
    current_features: np.ndarray,
) -> float:
    """Biased, non-negative RBF MMD with a pooled median bandwidth."""

    old = _matrix(old_features, "old features")
    current = _matrix(current_features, "current features")
    if old.shape[1] != current.shape[1]:
        raise SupportAnalysisError("old/current feature dimensions differ")
    pooled = np.concatenate([old, current], axis=0)
    distances = _squared_distances(pooled, pooled)
    positive = distances[distances > 0.0]
    bandwidth_squared = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / max(2.0 * bandwidth_squared, np.finfo(np.float64).eps)
    kernel_xx = np.exp(-gamma * _squared_distances(old, old)).mean()
    kernel_yy = np.exp(-gamma * _squared_distances(current, current)).mean()
    kernel_xy = np.exp(-gamma * _squared_distances(old, current)).mean()
    return float(max(kernel_xx + kernel_yy - 2.0 * kernel_xy, 0.0))


def gradient_conflict(
    old_gradient: np.ndarray,
    current_gradient: np.ndarray,
) -> tuple[float, float]:
    old = np.asarray(old_gradient, dtype=np.float64).reshape(-1)
    current = np.asarray(current_gradient, dtype=np.float64).reshape(-1)
    if old.shape != current.shape or old.size == 0:
        raise SupportAnalysisError("old/current gradients must have the same non-empty shape")
    if not np.isfinite(old).all() or not np.isfinite(current).all():
        raise SupportAnalysisError("gradients contain non-finite values")
    denominator = float(np.linalg.norm(old) * np.linalg.norm(current))
    if denominator == 0.0:
        raise SupportAnalysisError("gradient angle is undefined for a zero vector")
    cosine = float(np.clip(np.dot(old, current) / denominator, -1.0, 1.0))
    return cosine, float(np.degrees(np.arccos(cosine)))


def analyze_support(
    old_features: np.ndarray,
    current_features: np.ndarray,
    old_gradient: np.ndarray,
    current_gradient: np.ndarray,
    *,
    k: int = 5,
    radius_quantile: float = 0.95,
) -> SupportAnalysis:
    coverage, radius = knn_support_coverage(
        old_features, current_features, k=k, radius_quantile=radius_quantile
    )
    cosine, angle = gradient_conflict(old_gradient, current_gradient)
    return SupportAnalysis(
        old_covered_by_current=coverage,
        current_knn_radius=radius,
        rbf_mmd_squared=rbf_mmd_squared(old_features, current_features),
        gradient_cosine=cosine,
        gradient_angle_degrees=angle,
    )

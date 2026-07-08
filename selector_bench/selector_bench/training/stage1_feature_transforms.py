from __future__ import annotations

from typing import Any

import numpy as np

from selector_bench.core.feature_store import FeatureRecord
from selector_bench.core.manifest import ManifestRecord


def compute_raw_feature(record: ManifestRecord, features: FeatureRecord, spec: dict[str, Any]) -> float:
    kind = str(spec.get("kind", spec.get("source", "manifest_field")))
    if kind == "manifest_field":
        return float(getattr(record, str(spec["field"])))
    if kind == "metadata_field":
        value = _lookup_nested(record.metadata, str(spec["field"]))
        return float(value) if value is not None else float(spec.get("default", 0.0))
    if kind == "teacher_loss":
        return float(record.teacher_loss)
    if kind == "teacher_uncertainty":
        return float(record.teacher_uncertainty)
    if kind == "agent_count_norm":
        return float(record.agent_count) / float(spec.get("scale", 32.0))
    if kind == "ego_speed_norm":
        return float(record.ego_speed) / float(spec.get("scale", 30.0))
    if kind == "scene_norm":
        return _norm(features.scene_hidden)
    if kind == "interaction_norm":
        return _norm(features.interaction_hidden)
    if kind == "traj_hidden_norm":
        return _norm(features.traj_hidden) if features.traj_hidden is not None else 0.0
    if kind == "traj_speed_mean":
        traj = _traj(features.traj_raw)
        if traj.shape[0] < 2:
            return 0.0
        velocity = np.diff(traj, axis=0)
        return float(np.linalg.norm(velocity, axis=1).mean())
    if kind == "traj_accel_abs_mean":
        traj = _traj(features.traj_raw)
        if traj.shape[0] < 3:
            return 0.0
        accel = np.diff(traj, n=2, axis=0)
        return float(np.linalg.norm(accel, axis=1).mean())
    if kind == "traj_jerk_abs_mean":
        traj = _traj(features.traj_raw)
        if traj.shape[0] < 4:
            return 0.0
        jerk = np.diff(traj, n=3, axis=0)
        return float(np.linalg.norm(jerk, axis=1).mean())
    if kind == "traj_curvature_abs_mean":
        traj = _traj(features.traj_raw)
        if traj.shape[0] < 3:
            return 0.0
        velocity = np.diff(traj, axis=0)
        headings = np.arctan2(velocity[:, 1], velocity[:, 0])
        turns = np.diff(np.unwrap(headings))
        return float(np.abs(turns).mean()) if turns.size else 0.0
    if kind == "loss_component":
        key = str(spec["key"])
        value = (record.metadata or {}).get("loss_components", {}).get(key)
        return float(value) if value is not None else float(spec.get("default", 0.0))
    raise ValueError(f"Unsupported stage1 feature kind: {kind}")


def normalize_feature_values(
    raw_by_feature: dict[str, list[float]],
    specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, dict[str, float | str]]]:
    values: dict[str, list[float]] = {}
    stats: dict[str, dict[str, float | str]] = {}
    for name, raw_values in raw_by_feature.items():
        spec = specs.get(name, {})
        mode = str(spec.get("normalization", "identity"))
        arr = np.asarray(raw_values, dtype=np.float64)
        if mode == "identity":
            norm = arr
            entry: dict[str, float | str] = {"mode": mode}
        elif mode == "standard":
            mean = float(arr.mean()) if arr.size else 0.0
            std = float(arr.std() + 1e-6)
            norm = (arr - mean) / std
            entry = {"mode": mode, "mean": mean, "std": std}
        elif mode == "minmax":
            min_value = float(arr.min()) if arr.size else 0.0
            max_value = float(arr.max()) if arr.size else 0.0
            denom = max(max_value - min_value, 1e-6)
            norm = (arr - min_value) / denom
            entry = {"mode": mode, "min": min_value, "max": max_value}
        elif mode == "rank":
            order = np.argsort(arr, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(arr.size, dtype=np.float64)
            norm = ranks / max(arr.size - 1, 1)
            entry = {"mode": mode}
        else:
            raise ValueError(f"Unsupported normalization mode for {name}: {mode}")
        scale = float(spec.get("output_scale", 1.0))
        offset = float(spec.get("output_offset", 0.0))
        values[name] = (norm * scale + offset).astype(float).tolist()
        stats[name] = entry
    return values, stats


def _traj(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2) if arr.size % 2 == 0 else arr.reshape(-1, 1)
    if arr.shape[1] >= 2:
        return arr[:, :2]
    return np.pad(arr, ((0, 0), (0, 2 - arr.shape[1])), mode="constant")


def _norm(array: np.ndarray | None) -> float:
    if array is None:
        return 0.0
    arr = np.asarray(array, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr) / np.sqrt(arr.size))


def _lookup_nested(data: dict[str, Any] | None, path: str) -> Any:
    current: Any = data or {}
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current

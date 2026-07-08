from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.training.stage1_domain_targets import load_stage1_target_config
from selector_bench.training.stage1_feature_transforms import compute_raw_feature, normalize_feature_values
from selector_bench.utils.hashing import hash_jsonable
from selector_bench.utils.io import write_json, write_jsonl


def fit_stage1_domain_clusters(
    config_path: str | Path,
    labels_output: str | Path | None = None,
    model_output: str | Path | None = None,
    profile_output: str | Path | None = None,
) -> dict[str, Any]:
    config = load_stage1_target_config(config_path)
    feature_dir = config["feature_dir"]
    split = config.get("split", "train")
    cluster_cfg = dict(config.get("clustering", {}))
    num_clusters = int(cluster_cfg.get("num_clusters", 4))
    seed = int(cluster_cfg.get("seed", 0))
    max_iter = int(cluster_cfg.get("max_iter", 100))
    tolerance = float(cluster_cfg.get("tolerance", 1e-6))
    feature_specs = dict(config.get("cluster_features", config.get("features", {})))
    if not feature_specs:
        raise ValueError("Cluster config requires cluster_features or features.")

    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    records = store.manifest.filter(split=split)
    tokens = [record.sample_token for record in records]
    raw_by_feature = {name: [] for name in feature_specs}
    for record in records:
        features = store.get(record.sample_token)
        for name, spec in feature_specs.items():
            raw_by_feature[name].append(compute_raw_feature(record, features, spec))
    values_by_feature, normalizer = normalize_feature_values(raw_by_feature, feature_specs)
    feature_names = list(feature_specs)
    matrix = np.asarray([[values_by_feature[name][idx] for name in feature_names] for idx in range(len(records))], dtype=np.float64)
    weights = np.asarray([float(feature_specs[name].get("cluster_weight", feature_specs[name].get("weight", 1.0))) for name in feature_names])
    matrix = matrix * weights.reshape(1, -1)

    labels, centroids, iterations = _kmeans(matrix, num_clusters=num_clusters, seed=seed, max_iter=max_iter, tolerance=tolerance)
    label_prefix = str(cluster_cfg.get("label_prefix", "cluster_"))
    label_names = [f"{label_prefix}{int(label)}" for label in labels.tolist()]

    rows = [
        {
            "sample_token": token,
            "split": record.split,
            "cluster_id": int(label),
            "domain_label": label_name,
            "domain_source": "cluster",
        }
        for token, record, label, label_name in zip(tokens, records, labels, label_names)
    ]
    if labels_output is None:
        labels_output = config.get("outputs", {}).get("labels_path")
    if model_output is None:
        model_output = config.get("outputs", {}).get("cluster_model_path")
    if profile_output is None:
        profile_output = config.get("outputs", {}).get("cluster_profile_path")
    if labels_output:
        write_jsonl(labels_output, rows)

    profile = _build_profile(records, labels, feature_names, values_by_feature)
    model = {
        "format": "selector_bench.stage1_domain_clusters.v0",
        "config_path": str(config_path),
        "config_hash": hash_jsonable(config),
        "feature_dir": str(feature_dir),
        "split": split,
        "feature_names": feature_names,
        "feature_weights": weights.astype(float).tolist(),
        "normalization": normalizer,
        "num_clusters": num_clusters,
        "seed": seed,
        "max_iter": max_iter,
        "iterations": iterations,
        "centroids": centroids.astype(float).tolist(),
        "cluster_counts": {str(idx): int(np.sum(labels == idx)) for idx in range(num_clusters)},
        "profile": profile,
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="stage1_domain_cluster_model",
            run_id=str(config.get("run_id", "stage1_domain_clusters")),
            config={"config_path": str(config_path), "config_hash": hash_jsonable(config)},
            input_paths=[Path(feature_dir) / "manifest.jsonl", config_path],
            summary={"num_records": len(records), "num_clusters": num_clusters},
        ),
        "cluster_model": model,
    }
    if model_output:
        write_json(model_output, artifact)
    if profile_output:
        _write_profile(profile_output, model)
    return {"labels": rows, "cluster_model": artifact}


def _kmeans(
    matrix: np.ndarray,
    num_clusters: int,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("k-means requires a non-empty 2-D feature matrix.")
    if num_clusters <= 0 or num_clusters > matrix.shape[0]:
        raise ValueError("num_clusters must be in [1, num_records].")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, matrix.shape[0]))
    centroids = [matrix[first]]
    while len(centroids) < num_clusters:
        distances = _min_squared_distance(matrix, np.asarray(centroids))
        total = float(distances.sum())
        if total <= 0:
            candidates = [idx for idx in range(matrix.shape[0]) if not any(np.allclose(matrix[idx], c) for c in centroids)]
            next_idx = candidates[0] if candidates else len(centroids)
        else:
            probs = distances / total
            next_idx = int(rng.choice(matrix.shape[0], p=probs))
        centroids.append(matrix[next_idx])
    centroid_array = np.asarray(centroids, dtype=np.float64)
    labels = np.zeros((matrix.shape[0],), dtype=np.int64)
    for iteration in range(1, max_iter + 1):
        distances = ((matrix[:, None, :] - centroid_array[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        new_centroids = centroid_array.copy()
        for idx in range(num_clusters):
            members = matrix[labels == idx]
            if members.size:
                new_centroids[idx] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(_min_squared_distance(matrix, centroid_array)))
                new_centroids[idx] = matrix[farthest]
        shift = float(np.max(np.abs(new_centroids - centroid_array)))
        centroid_array = new_centroids
        if shift <= tolerance:
            return labels, centroid_array, iteration
    return labels, centroid_array, max_iter


def _min_squared_distance(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).min(axis=1)


def _build_profile(records, labels: np.ndarray, feature_names: list[str], values_by_feature: dict[str, list[float]]) -> list[dict[str, Any]]:
    profile = []
    for cluster_id in sorted({int(item) for item in labels.tolist()}):
        indices = np.flatnonzero(labels == cluster_id)
        locations: dict[str, int] = {}
        for idx in indices.tolist():
            locations[records[idx].location] = locations.get(records[idx].location, 0) + 1
        feature_means = {
            name: float(np.asarray(values_by_feature[name], dtype=np.float64)[indices].mean())
            for name in feature_names
            if indices.size
        }
        profile.append(
            {
                "cluster_id": cluster_id,
                "domain_label": f"cluster_{cluster_id}",
                "size": int(indices.size),
                "locations": locations,
                "feature_means": feature_means,
                "representative_tokens": [records[idx].sample_token for idx in indices[:5].tolist()],
            }
        )
    return profile


def _write_profile(path: str | Path, model: dict[str, Any]) -> None:
    lines = [
        "# Stage 1 Domain Cluster Profile",
        "",
        f"- feature_dir: `{model['feature_dir']}`",
        f"- split: `{model['split']}`",
        f"- config_hash: `{model['config_hash']}`",
        "",
    ]
    for item in model["profile"]:
        lines.extend(
            [
                f"## {item['domain_label']}",
                "",
                f"- size: {item['size']}",
                f"- locations: {item['locations']}",
                f"- representative_tokens: {item['representative_tokens']}",
                "- feature_means:",
            ]
        )
        for name, value in item["feature_means"].items():
            lines.append(f"  - {name}: {value:.6f}")
        lines.append("")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")

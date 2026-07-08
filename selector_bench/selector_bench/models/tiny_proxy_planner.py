from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.core.manifest import ManifestRecord
from selector_bench.utils.io import write_json


def train_tiny_proxy_seed(
    feature_dir: str | Path,
    output_path: str | Path,
    ridge_alpha: float = 1e-3,
    run_id: str = "tiny_proxy_seed_lightweight",
) -> dict[str, Any]:
    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    train_tokens = store.manifest.tokens(split="train")
    val_tokens = store.manifest.tokens(split="val")
    x_train, y_train = build_proxy_xy(store, train_tokens)
    x_val, y_val = build_proxy_xy(store, val_tokens)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_aug = _augment((x_train - x_mean) / x_std)
    x_val_aug = _augment((x_val - x_mean) / x_std) if len(x_val) else np.zeros((0, x_train_aug.shape[1]))

    eye = np.eye(x_train_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    weights = np.linalg.solve(
        x_train_aug.T @ x_train_aug + ridge_alpha * eye,
        x_train_aug.T @ y_train,
    )

    train_pred = x_train_aug @ weights
    val_pred = x_val_aug @ weights if len(x_val_aug) else np.zeros_like(y_val)
    metrics = {
        "train_mse": _mse(train_pred, y_train),
        "val_mse": _mse(val_pred, y_val),
        "num_train": int(len(train_tokens)),
        "num_val": int(len(val_tokens)),
        "input_dim": int(x_train.shape[1]),
        "output_dim": int(y_train.shape[1]),
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="tiny_proxy_seed",
            run_id=run_id,
            config={"ridge_alpha": ridge_alpha},
            input_paths=[Path(feature_dir) / "manifest.jsonl"],
            metrics=metrics,
            summary={"num_train": len(train_tokens), "num_val": len(val_tokens)},
        ),
        "proxy": {
            "format": "selector_bench.tiny_proxy_ridge.v1",
            "feature_names": proxy_feature_names(store.manifest.filter(split="train")),
            "x_mean": x_mean.squeeze(0).tolist(),
            "x_std": x_std.squeeze(0).tolist(),
            "weights": weights.tolist(),
            "ridge_alpha": ridge_alpha,
        },
    }
    write_json(output_path, artifact)
    return artifact


def build_proxy_xy(store: FeatureStore, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for token in tokens:
        record = store.manifest.get(token)
        features = store.get(token)
        traj = features.traj_raw
        speed_col = traj[:, 2] if traj.ndim == 2 and traj.shape[1] > 2 else traj.reshape(-1)
        x_rows.append(
            [
                float(np.linalg.norm(features.scene_hidden) / np.sqrt(features.scene_hidden.size)),
                float(
                    np.linalg.norm(features.interaction_hidden)
                    / np.sqrt(features.interaction_hidden.size)
                ),
                float(np.mean(np.abs(speed_col))),
                float(record.teacher_loss),
                float(record.teacher_uncertainty),
                float(record.agent_count) / 32.0,
                float(record.ego_speed) / 30.0,
                float(record.domain_id),
            ]
        )
        y_rows.append(
            [
                float(traj[-1, 0] - traj[0, 0]),
                float(traj[-1, 1] - traj[0, 1]),
                float(speed_col[-1]),
            ]
        )
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def proxy_feature_names(records: list[ManifestRecord]) -> list[str]:
    _ = records
    return [
        "scene_norm",
        "interaction_norm",
        "traj_speed_mean",
        "teacher_loss",
        "teacher_uncertainty",
        "agent_count",
        "ego_speed",
        "domain_id",
    ]


def _augment(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean((pred - target) ** 2))

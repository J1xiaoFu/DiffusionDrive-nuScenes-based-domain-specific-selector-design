from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.utils.io import read_json, write_json
from selector_bench.utils.seed import seed_everything


SELECTOR_FEATURES = [
    "scene_norm",
    "interaction_norm",
    "traj_speed_mean",
    "teacher_loss",
    "teacher_uncertainty",
    "agent_count",
    "ego_speed",
]


def train_selector_stage2_lightweight(
    feature_dir: str | Path,
    proxy_path: str | Path,
    init_selector_path: str | Path,
    output_path: str | Path,
    iterations: int = 30,
    budget: float = 0.2,
    lr: float = 0.05,
    ema_alpha: float = 0.8,
    seed: int = 1,
    run_id: str = "stage2_lightweight",
) -> dict[str, Any]:
    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    proxy_artifact = read_json(proxy_path)
    init_artifact = read_json(init_selector_path)
    selector = init_artifact.get("selector", init_artifact)
    weights = {name: float(selector.get("weights", {}).get(name, 0.0)) for name in SELECTOR_FEATURES}
    bias = float(selector.get("weights", {}).get("bias", 0.0))
    domain_bias = {
        str(key): float(value)
        for key, value in selector.get("domain_bias", {}).items()
    }

    rng = seed_everything(seed)
    tokens = store.manifest.tokens(split="train")
    feature_matrix, domain_ids, proxy_losses = _build_stage2_arrays(store, tokens)
    k = max(1, int(np.ceil(len(tokens) * budget)))

    reward_ema = 0.0
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        scores = _score(feature_matrix, domain_ids, weights, bias, domain_bias)
        high_idx = np.argsort(-scores)[:k]
        ref_idx = rng.choice(len(tokens), size=k, replace=False)
        high_loss = float(proxy_losses[high_idx].mean())
        ref_loss = float(proxy_losses[ref_idx].mean())
        reward = ref_loss - high_loss
        reward_ema = ema_alpha * reward_ema + (1.0 - ema_alpha) * reward
        advantage = reward - reward_ema

        direction = feature_matrix[high_idx].mean(axis=0) - feature_matrix[ref_idx].mean(axis=0)
        for col, name in enumerate(SELECTOR_FEATURES):
            weights[name] += lr * advantage * float(direction[col])
        bias += lr * advantage * 0.01

        for domain_id in sorted(set(domain_ids.tolist())):
            high_mask = domain_ids[high_idx] == domain_id
            ref_mask = domain_ids[ref_idx] == domain_id
            if high_mask.any() or ref_mask.any():
                key = str(int(domain_id))
                domain_bias[key] = domain_bias.get(key, 0.0) + lr * advantage * (
                    float(high_mask.mean()) - float(ref_mask.mean())
                )

        history.append(
            {
                "iteration": iteration,
                "reward": reward,
                "reward_ema": reward_ema,
                "advantage": advantage,
                "high_proxy_loss": high_loss,
                "ref_proxy_loss": ref_loss,
                "selected_count": int(k),
            }
        )

    final_scores = _score(feature_matrix, domain_ids, weights, bias, domain_bias)
    final_idx = np.argsort(-final_scores)[:k]
    final_loss = float(proxy_losses[final_idx].mean())
    random_losses = []
    for _ in range(20):
        ref_idx = rng.choice(len(tokens), size=k, replace=False)
        random_losses.append(float(proxy_losses[ref_idx].mean()))
    metrics = {
        "iterations": iterations,
        "budget": budget,
        "selected_count": int(k),
        "final_selected_proxy_loss": final_loss,
        "random_proxy_loss_mean": float(np.mean(random_losses)),
        "delta_random_minus_selected": float(np.mean(random_losses) - final_loss),
        "last_reward": history[-1]["reward"] if history else 0.0,
        "proxy_format": proxy_artifact.get("proxy", {}).get("format", "unknown"),
    }

    output = {
        "metadata": make_artifact_metadata(
            artifact_type="selector_checkpoint",
            run_id=run_id,
            config={
                "stage": "stage2_lightweight",
                "iterations": iterations,
                "budget": budget,
                "lr": lr,
                "ema_alpha": ema_alpha,
                "seed": seed,
            },
            input_paths=[
                Path(feature_dir) / "manifest.jsonl",
                proxy_path,
                init_selector_path,
            ],
            metrics=metrics,
            summary={"num_train_records": len(tokens), "selected_count": int(k)},
        ),
        "selector": {
            "format": "selector_bench.linear_selector.v1",
            "weights": {"bias": bias, **weights},
            "domain_bias": domain_bias,
            "stage2_history": history,
        },
    }
    write_json(output_path, output)
    return output


def _build_stage2_arrays(store: FeatureStore, tokens: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    domain_ids = []
    proxy_losses = []
    for token in tokens:
        record = store.manifest.get(token)
        features = store.get(token)
        traj = features.traj_raw
        speed_col = traj[:, 2] if traj.ndim == 2 and traj.shape[1] > 2 else traj.reshape(-1)
        rows.append(
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
            ]
        )
        domain_ids.append(record.domain_id)
        utility = float(
            record.teacher_loss
            + 0.5 * record.teacher_uncertainty
            + 0.05 * (record.agent_count / 32.0)
        )
        # Smoke oracle: higher utility stands in for lower downstream
        # validation loss after warm-start. Real Stage 2 replaces this with
        # actual TinyProxy validation loss.
        proxy_losses.append(-utility)
    matrix = np.asarray(rows, dtype=np.float64)
    matrix = (matrix - matrix.mean(axis=0, keepdims=True)) / (matrix.std(axis=0, keepdims=True) + 1e-6)
    return (
        matrix,
        np.asarray(domain_ids, dtype=np.int64),
        np.asarray(proxy_losses, dtype=np.float64),
    )


def _score(
    feature_matrix: np.ndarray,
    domain_ids: np.ndarray,
    weights: dict[str, float],
    bias: float,
    domain_bias: dict[str, float],
) -> np.ndarray:
    weight_vector = np.asarray([weights[name] for name in SELECTOR_FEATURES], dtype=np.float64)
    scores = feature_matrix @ weight_vector + bias
    for idx, domain_id in enumerate(domain_ids):
        scores[idx] += domain_bias.get(str(int(domain_id)), 0.0)
    return scores

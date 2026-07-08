from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.budget_allocator import budget_count, proportional_domain_budget
from selector_bench.core.feature_store import FeatureRecord, FeatureStore
from selector_bench.core.manifest import ManifestRecord
from selector_bench.core.score_provider import OurSelectorScoreProvider
from selector_bench.training.stage1_domain_targets import load_target_score_table
from selector_bench.utils.hashing import hash_jsonable
from selector_bench.utils.io import ensure_dir, read_json, read_jsonl, write_json


FEATURE_BLOCKS = [
    "scene_hidden",
    "interaction_hidden",
    "traj_hidden",
    "traj_raw",
    "teacher_loss",
    "teacher_uncertainty",
    "agent_count_norm",
    "ego_speed_norm",
]


def train_neural_selector_stage1_imitation(
    feature_dir: str | Path,
    teacher_selector_path: str | Path | None,
    output_path: str | Path,
    run_id: str = "stage2_neural_stage1_imitation",
    hidden_dim: int = 64,
    epochs: int = 400,
    lr: float = 0.03,
    l2: float = 1e-4,
    holdout_ratio: float = 0.2,
    seed: int = 7,
    use_domain_bias: bool = True,
    target_score_path: str | Path | None = None,
    use_domain_residual_head: bool = False,
) -> dict[str, Any]:
    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    tokens = store.manifest.tokens(split="train")
    if not tokens:
        raise ValueError("No train records found in feature manifest.")

    if target_score_path is not None:
        target_tokens, teacher_scores, domain_labels, _ = load_target_score_table(target_score_path)
        train_token_set = set(tokens)
        pairs = [
            (token, float(score), str(label))
            for token, score, label in zip(target_tokens, teacher_scores.tolist(), domain_labels)
            if token in train_token_set
        ]
        if not pairs:
            raise ValueError(f"No target-score rows matched train records in {feature_dir}.")
        tokens = [item[0] for item in pairs]
        teacher_scores = np.asarray([item[1] for item in pairs], dtype=np.float64)
        domain_keys = [item[2] for item in pairs]
        target_source = "stage1_target_scores_jsonl"
    else:
        if teacher_selector_path is None:
            raise ValueError("teacher_selector_path is required when target_score_path is not provided.")
        teacher_scores = _score_teacher_selector(store, teacher_selector_path, tokens)
        domain_keys = [str(store.manifest.get(token).domain_id) for token in tokens]
        target_source = "linear_selector_checkpoint"

    features = _build_feature_matrix(store, tokens)
    domain_values = sorted(set(domain_keys))
    domain_to_col = {domain_key: idx for idx, domain_key in enumerate(domain_values)}
    domain_cols = np.asarray([domain_to_col[item] for item in domain_keys], dtype=np.int64)

    x_mean = features.mean(axis=0, keepdims=True)
    x_std = features.std(axis=0, keepdims=True) + 1e-6
    x = (features - x_mean) / x_std
    y_mean = float(teacher_scores.mean())
    y_std = float(teacher_scores.std() + 1e-6)
    y = ((teacher_scores - y_mean) / y_std).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    train_idx, holdout_idx = _split_indices(len(tokens), holdout_ratio, rng)
    params = _init_params(
        x.shape[1],
        hidden_dim,
        len(domain_values),
        rng,
        use_domain_bias,
        use_domain_residual_head,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        metrics = _train_epoch(
            params=params,
            x=x[train_idx],
            y=y[train_idx],
            domain_cols=domain_cols[train_idx],
            lr=lr,
            l2=l2,
            use_domain_bias=use_domain_bias,
        )
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0:
            train_mse = _mse(_predict_normalized(params, x[train_idx], domain_cols[train_idx], use_domain_bias), y[train_idx])
            holdout_mse = (
                _mse(_predict_normalized(params, x[holdout_idx], domain_cols[holdout_idx], use_domain_bias), y[holdout_idx])
                if holdout_idx.size
                else train_mse
            )
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(metrics["loss"]),
                    "train_mse": float(train_mse),
                    "holdout_mse": float(holdout_mse),
                }
            )

    train_pred = _predict_normalized(params, x[train_idx], domain_cols[train_idx], use_domain_bias)
    holdout_pred = (
        _predict_normalized(params, x[holdout_idx], domain_cols[holdout_idx], use_domain_bias)
        if holdout_idx.size
        else train_pred
    )
    metrics = {
        "train_mse_normalized": float(_mse(train_pred, y[train_idx])),
        "holdout_mse_normalized": float(_mse(holdout_pred, y[holdout_idx]) if holdout_idx.size else _mse(train_pred, y[train_idx])),
        "num_train_records": int(train_idx.size),
        "num_holdout_records": int(holdout_idx.size),
        "teacher_score_mean": y_mean,
        "teacher_score_std": y_std,
    }

    selector = {
        "format": "selector_bench.neural_selector.v0",
        "architecture": {
            "type": "mlp_per_sample",
            "activation": "tanh",
            "hidden_dim": hidden_dim,
            "use_domain_bias": use_domain_bias,
            "use_domain_residual_head": use_domain_residual_head,
        },
        "feature_spec": {
            "blocks": FEATURE_BLOCKS,
            "input_dim": int(x.shape[1]),
            "normalization": "train_split_standard_score",
        },
        "normalization": {
            "input_mean": x_mean.reshape(-1).astype(float).tolist(),
            "input_std": x_std.reshape(-1).astype(float).tolist(),
            "target_mean": y_mean,
            "target_std": y_std,
        },
        "domain_id_to_index": {str(domain_id): idx for domain_id, idx in domain_to_col.items()},
        "domain_label_to_index": {str(domain_id): idx for domain_id, idx in domain_to_col.items()},
        "token_domain_label": {token: label for token, label in zip(tokens, domain_keys)}
        if target_score_path is not None
        else {},
        "training_target": {
            "source": target_source,
            "teacher_selector_path": str(teacher_selector_path) if teacher_selector_path is not None else None,
            "target_score_path": str(target_score_path) if target_score_path is not None else None,
        },
        "params": _params_to_json(params),
        "training_history": history,
    }
    input_paths = [Path(feature_dir) / "manifest.jsonl"]
    if teacher_selector_path is not None:
        input_paths.append(Path(teacher_selector_path))
    if target_score_path is not None:
        input_paths.append(Path(target_score_path))

    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="selector_checkpoint",
            run_id=run_id,
            config={
                "stage": "stage2_neural_stage1_imitation",
                "target_source": target_source,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "lr": lr,
                "l2": l2,
                "holdout_ratio": holdout_ratio,
                "seed": seed,
                "use_domain_bias": use_domain_bias,
                "use_domain_residual_head": use_domain_residual_head,
            },
            input_paths=input_paths,
            metrics=metrics,
            summary={
                "format": selector["format"],
                "num_records": len(tokens),
                "num_domains": len(domain_values),
                "input_dim": int(x.shape[1]),
            },
        ),
        "selector": selector,
    }
    write_json(output_path, artifact)
    return artifact


def score_neural_selector(
    feature_store: FeatureStore,
    selector_checkpoint: str | Path | dict[str, Any],
    tokens: list[str],
) -> np.ndarray:
    artifact = read_json(selector_checkpoint) if not isinstance(selector_checkpoint, dict) else selector_checkpoint
    selector = artifact.get("selector", artifact)
    if selector.get("format") != "selector_bench.neural_selector.v0":
        raise ValueError(f"Unsupported selector format: {selector.get('format')}")

    params = _params_from_json(selector["params"])
    x = _build_feature_matrix(feature_store, tokens)
    norm = selector["normalization"]
    mean = np.asarray(norm["input_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(norm["input_std"], dtype=np.float64).reshape(1, -1)
    x = (x - mean) / np.maximum(std, 1e-6)
    domain_cols = _domain_columns_for_tokens(feature_store, selector, tokens)
    use_domain_bias = bool(selector.get("architecture", {}).get("use_domain_bias", True))
    pred = _predict_normalized(params, x, domain_cols, use_domain_bias).reshape(-1)
    scores = pred * float(norm["target_std"]) + float(norm["target_mean"])
    token_offsets = selector.get("score_adjustments", {}).get("token_offsets", {})
    if token_offsets:
        offsets = np.asarray([float(token_offsets.get(token, 0.0)) for token in tokens], dtype=np.float64)
        scores = scores + offsets
    return scores.astype(np.float32)



def train_neural_selector_stage2_feedback(
    feature_dir: str | Path,
    init_selector_path: str | Path,
    episode_jsonl: str | Path,
    output_path: str | Path,
    run_id: str = "stage2_neural_feedback",
    epochs: int = 200,
    lr: float = 0.01,
    l2: float = 1e-4,
    reward_scale: float | None = None,
    target_clip: float = 2.0,
    min_abs_reward: float = 0.0,
    negative_token_penalty: float = 0.0,
    negative_random_bonus: float = 0.0,
    negative_reward_max: float = -1e-12,
    seed: int = 17,
) -> dict[str, Any]:
    """Fine-tune a neural selector from subset-level DiffusionDrive feedback.

    Each episode teaches the selector that the selected subset's mean score
    should exceed the random baseline's mean score by a normalized reward.
    Negative rewards therefore push the selector away from its own sampled
    subset relative to the same-budget random baseline.
    """

    if epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if target_clip <= 0:
        raise ValueError("target_clip must be positive.")
    if negative_token_penalty < 0:
        raise ValueError("negative_token_penalty must be non-negative.")
    if negative_random_bonus < 0:
        raise ValueError("negative_random_bonus must be non-negative.")

    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    init_artifact = read_json(init_selector_path)
    selector = init_artifact.get("selector", init_artifact)
    if selector.get("format") != "selector_bench.neural_selector.v0":
        raise ValueError(f"Unsupported selector format: {selector.get('format')}")

    episodes = _load_feedback_episodes(episode_jsonl, store, min_abs_reward=min_abs_reward)
    if not episodes:
        raise ValueError(f"No usable feedback episodes found in {episode_jsonl}.")

    scale = _resolve_reward_scale(episodes, reward_scale)
    for episode in episodes:
        episode["target_gap"] = float(np.clip(float(episode["reward"]) / scale, -target_clip, target_clip))

    params = _params_from_json(selector["params"])
    norm = selector["normalization"]
    mean = np.asarray(norm["input_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(norm["input_std"], dtype=np.float64).reshape(1, -1)
    use_domain_bias = bool(selector.get("architecture", {}).get("use_domain_bias", True))
    token_cache = _build_feedback_token_cache(store, selector, episodes, mean, std)

    before_metrics = _feedback_metrics(params, token_cache, episodes, use_domain_bias)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    order = np.arange(len(episodes), dtype=np.int64)
    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_losses = []
        for idx in order.tolist():
            episode = episodes[idx]
            metrics = _train_feedback_episode(
                params=params,
                token_cache=token_cache,
                selector_tokens=episode["selector_tokens"],
                random_tokens=episode["random_tokens"],
                target_gap=float(episode["target_gap"]),
                lr=lr,
                l2=l2,
                use_domain_bias=use_domain_bias,
            )
            epoch_losses.append(metrics["loss"])
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0:
            eval_metrics = _feedback_metrics(params, token_cache, episodes, use_domain_bias)
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                    "feedback_mse": float(eval_metrics["feedback_mse"]),
                    "preference_accuracy": float(eval_metrics["preference_accuracy"]),
                    "mean_score_gap": float(eval_metrics["mean_score_gap"]),
                }
            )

    after_metrics = _feedback_metrics(params, token_cache, episodes, use_domain_bias)
    output_selector = dict(selector)
    output_selector["params"] = _params_to_json(params)
    token_score_offsets = _build_negative_feedback_token_offsets(
        episodes=episodes,
        negative_token_penalty=negative_token_penalty,
        negative_random_bonus=negative_random_bonus,
        negative_reward_max=negative_reward_max,
        target_std=float(norm["target_std"]),
    )
    if token_score_offsets:
        output_selector["score_adjustments"] = {
            "type": "token_offsets",
            "units": "selector_score",
            "source": "negative_feedback_calibration",
            "negative_token_penalty": negative_token_penalty,
            "negative_random_bonus": negative_random_bonus,
            "negative_reward_max": negative_reward_max,
            "token_offsets": token_score_offsets,
        }
        calibrated_after_metrics = _feedback_metrics(
            params,
            token_cache,
            episodes,
            use_domain_bias,
            token_score_offsets=token_score_offsets,
            target_std=float(norm["target_std"]),
        )
    else:
        output_selector.pop("score_adjustments", None)
        calibrated_after_metrics = after_metrics
    output_selector["stage2_feedback"] = {
        "source": "stage2_episode_jsonl",
        "init_selector_path": str(init_selector_path),
        "episode_jsonl": str(episode_jsonl),
        "reward_scale": scale,
        "target_clip": target_clip,
        "min_abs_reward": min_abs_reward,
        "negative_token_penalty": negative_token_penalty,
        "negative_random_bonus": negative_random_bonus,
        "negative_reward_max": negative_reward_max,
        "epochs": epochs,
        "lr": lr,
        "l2": l2,
        "seed": seed,
        "score_adjustment_token_count": len(token_score_offsets),
        "episodes": [
            {
                "episode_id": str(episode["episode_id"]),
                "reward": float(episode["reward"]),
                "target_gap": float(episode["target_gap"]),
                "selector_selection_hash": str(episode["selector_selection_hash"]),
                "random_selection_hash": str(episode["random_selection_hash"]),
                "selector_count": len(episode["selector_tokens"]),
                "random_count": len(episode["random_tokens"]),
            }
            for episode in episodes
        ],
        "history": history,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "calibrated_after_metrics": calibrated_after_metrics,
    }

    selection_paths = sorted(
        {
            str(episode["selector_selection_path"])
            for episode in episodes
        }
        | {
            str(episode["random_selection_path"])
            for episode in episodes
        }
    )
    metrics = {
        "num_feedback_episodes": len(episodes),
        "reward_scale": scale,
        "before_feedback_mse": before_metrics["feedback_mse"],
        "after_feedback_mse": after_metrics["feedback_mse"],
        "before_preference_accuracy": before_metrics["preference_accuracy"],
        "after_preference_accuracy": after_metrics["preference_accuracy"],
        "before_mean_score_gap": before_metrics["mean_score_gap"],
        "after_mean_score_gap": after_metrics["mean_score_gap"],
        "after_calibrated_feedback_mse": calibrated_after_metrics["feedback_mse"],
        "after_calibrated_preference_accuracy": calibrated_after_metrics["preference_accuracy"],
        "score_adjustment_token_count": len(token_score_offsets),
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="selector_checkpoint",
            run_id=run_id,
            config={
                "stage": "stage2_neural_feedback",
                "epochs": epochs,
                "lr": lr,
                "l2": l2,
                "reward_scale": reward_scale,
                "resolved_reward_scale": scale,
                "target_clip": target_clip,
                "min_abs_reward": min_abs_reward,
                "negative_token_penalty": negative_token_penalty,
                "negative_random_bonus": negative_random_bonus,
                "negative_reward_max": negative_reward_max,
                "seed": seed,
            },
            input_paths=[
                Path(feature_dir) / "manifest.jsonl",
                init_selector_path,
                episode_jsonl,
                *[Path(path) for path in selection_paths],
            ],
            metrics=metrics,
            summary={
                "format": output_selector["format"],
                "num_feedback_episodes": len(episodes),
                "num_feedback_tokens": len(token_cache),
            },
        ),
        "selector": output_selector,
    }
    write_json(output_path, artifact)
    return artifact


def _domain_columns_for_tokens(feature_store: FeatureStore, selector: dict[str, Any], tokens: list[str]) -> np.ndarray:
    if "domain_label_to_index" in selector:
        label_to_col = {str(key): int(value) for key, value in selector["domain_label_to_index"].items()}
        token_labels = {str(key): str(value) for key, value in selector.get("token_domain_label", {}).items()}
        cols = []
        for token in tokens:
            record = feature_store.manifest.get(token)
            candidates = [token_labels.get(token), str(record.domain_id), record.domain_name, record.location]
            cols.append(next((label_to_col[item] for item in candidates if item in label_to_col), -1))
        return np.asarray(cols, dtype=np.int64)
    domain_to_col = {int(key): int(value) for key, value in selector["domain_id_to_index"].items()}
    return np.asarray(
        [domain_to_col.get(int(feature_store.manifest.get(token).domain_id), -1) for token in tokens],
        dtype=np.int64,
    )


def sample_gumbel_topk_selection(
    feature_dir: str | Path,
    selector_checkpoint: str | Path,
    output_path: str | Path,
    run_id: str = "stage2_neural_gumbel_topk",
    budget: float = 0.2,
    split: str = "train",
    temperature: float = 0.15,
    seed: int = 13,
    replay_episode_jsonl: str | Path | None = None,
    replay_penalty: float = 0.0,
    replay_reward_max: float = -1e-12,
    exclude_tokens: set[str] | None = None,
) -> dict[str, Any]:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    excluded = exclude_tokens or set()
    records = [
        record for record in store.manifest.filter(split=split)
        if record.sample_token not in excluded
    ]
    tokens = [record.sample_token for record in records]
    scores = score_neural_selector(store, selector_checkpoint, tokens)
    replay_tokens = (
        _selector_tokens_from_feedback_episodes(
            replay_episode_jsonl,
            store=store,
            split=split,
            reward_max=replay_reward_max,
        )
        if replay_episode_jsonl is not None and replay_penalty > 0
        else set()
    )
    if replay_tokens:
        scores = np.asarray(scores, dtype=np.float64).copy()
        for idx, token in enumerate(tokens):
            if token in replay_tokens:
                scores[idx] -= replay_penalty
    total_budget = budget_count(len(records), budget)
    domain_budget = proportional_domain_budget(records, total_budget)
    rng = np.random.default_rng(seed)

    selected: list[str] = []
    noisy_score_by_token: dict[str, float] = {}
    score_by_token = {token: float(score) for token, score in zip(tokens, scores)}
    by_domain: dict[int, list[ManifestRecord]] = {}
    for record in records:
        by_domain.setdefault(record.domain_id, []).append(record)
    for domain_id, domain_records in sorted(by_domain.items()):
        k = domain_budget.get(domain_id, 0)
        if k <= 0:
            continue
        domain_tokens = [record.sample_token for record in domain_records]
        clean = np.asarray([score_by_token[token] for token in domain_tokens], dtype=np.float64)
        noisy = clean + temperature * _sample_gumbel(rng, len(domain_tokens))
        for token, value in zip(domain_tokens, noisy):
            noisy_score_by_token[token] = float(value)
        order = sorted(range(len(domain_tokens)), key=lambda idx: (-float(noisy[idx]), domain_tokens[idx]))
        selected.extend(domain_tokens[idx] for idx in order[:k])

    domain_counts = {
        str(domain_id): sum(1 for token in selected if store.manifest.get(token).domain_id == domain_id)
        for domain_id in sorted(by_domain)
    }
    selected_clean_scores = [score_by_token[token] for token in selected]
    selected_noisy_scores = [noisy_score_by_token[token] for token in selected]
    selected_replay_overlap = sorted(token for token in selected if token in replay_tokens)
    payload = {
        "baseline": "neural_gumbel_topk",
        "score_source": "selector_bench.neural_selector.v0",
        "budget_ratio": budget,
        "budget_mode": "ratio" if budget <= 1 else "exact_count",
        "budget_count": total_budget,
        "split": split,
        "selected_count": len(selected),
        "selected_tokens": selected,
        "selection_hash": hash_jsonable(selected),
        "domain_budget": {str(key): value for key, value in domain_budget.items()},
        "domain_counts": domain_counts,
        "replay_avoidance": {
            "episode_jsonl": str(replay_episode_jsonl) if replay_episode_jsonl is not None else None,
            "penalty": replay_penalty,
            "reward_max": replay_reward_max,
            "candidate_replay_token_count": len(replay_tokens),
            "selected_replay_overlap_count": len(selected_replay_overlap),
            "selected_replay_overlap_tokens": selected_replay_overlap,
        },
        "exclusions": {
            "excluded_token_count": len(excluded),
        },
        "score_summary": {
            "all_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "selected_score_mean": float(np.mean(selected_clean_scores)) if selected_clean_scores else 0.0,
            "selected_score_min": float(np.min(selected_clean_scores)) if selected_clean_scores else 0.0,
            "selected_score_max": float(np.max(selected_clean_scores)) if selected_clean_scores else 0.0,
            "selected_noisy_score_mean": float(np.mean(selected_noisy_scores)) if selected_noisy_scores else 0.0,
            "temperature": temperature,
        },
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="selection",
            run_id=run_id,
            config={
                "sampler": "domain_stratified_gumbel_topk",
                "features": str(feature_dir),
                "selector": str(selector_checkpoint),
                "budget": budget,
                "budget_count": int(budget) if budget > 1 else None,
                "split": split,
                "temperature": temperature,
                "seed": seed,
                "excluded_token_count": len(excluded),
                "replay_episode_jsonl": str(replay_episode_jsonl) if replay_episode_jsonl is not None else None,
                "replay_penalty": replay_penalty,
                "replay_reward_max": replay_reward_max,
            },
            input_paths=[Path(feature_dir) / "manifest.jsonl", selector_checkpoint],
            summary={
                "selected_count": len(selected),
                "selection_hash": payload["selection_hash"],
            },
        ),
        "selection": payload,
    }
    write_json(output_path, artifact)
    return artifact


def _selector_tokens_from_feedback_episodes(
    episode_jsonl: str | Path,
    store: FeatureStore,
    split: str,
    reward_max: float,
) -> set[str]:
    manifest_tokens = set(store.manifest.tokens(split=split))
    tokens: set[str] = set()
    for row in read_jsonl(episode_jsonl):
        if row.get("schema") != "selector_bench.stage2_episode.v0":
            raise ValueError(f"Unexpected episode schema in {episode_jsonl}: {row.get('schema')}")
        if float(row.get("reward", 0.0)) > reward_max:
            continue
        selection_path = Path(row["selector_selection"]["path"])
        artifact = read_json(selection_path)
        tokens.update(
            token
            for token in artifact["selection"]["selected_tokens"]
            if token in manifest_tokens
        )
    return tokens


def reward_cost(metrics: dict[str, float], rho_box: float = 0.05, rho_obj: float = 0.05) -> float:
    return float(metrics["l2"]) + rho_box * float(metrics.get("obj_box_col", 0.0)) + rho_obj * float(metrics.get("obj_col", 0.0))


def build_episode_record(
    episode_id: str,
    selector_selection_path: str | Path,
    random_selection_path: str | Path,
    selector_metrics: dict[str, float],
    random_metrics: dict[str, float],
    rho_box: float = 0.05,
    rho_obj: float = 0.05,
    train_work_dirs: dict[str, str] | None = None,
    eval_work_dirs: dict[str, str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    selector_cost = reward_cost(selector_metrics, rho_box=rho_box, rho_obj=rho_obj)
    random_cost = reward_cost(random_metrics, rho_box=rho_box, rho_obj=rho_obj)
    selector_artifact = read_json(selector_selection_path)
    random_artifact = read_json(random_selection_path)
    return {
        "schema": "selector_bench.stage2_episode.v0",
        "episode_id": episode_id,
        "selector_selection": {
            "path": str(selector_selection_path),
            "selection_hash": selector_artifact["selection"]["selection_hash"],
            "selected_count": selector_artifact["selection"]["selected_count"],
        },
        "random_same_budget_selection": {
            "path": str(random_selection_path),
            "selection_hash": random_artifact["selection"]["selection_hash"],
            "selected_count": random_artifact["selection"]["selected_count"],
        },
        "train_work_dirs": train_work_dirs or {},
        "eval_work_dirs": eval_work_dirs or {},
        "metrics": {
            "selector": selector_metrics,
            "random_same_budget": random_metrics,
        },
        "reward_definition": {
            "cost": "J(S)=L2+rho_box*obj_box_col+rho_obj*obj_col",
            "reward": "R=J(random_same_budget)-J(selector)",
            "rho_box": rho_box,
            "rho_obj": rho_obj,
            "collision_units": "record values exactly as emitted by standalone eval; choose rho values accordingly",
        },
        "cost": {
            "selector": selector_cost,
            "random_same_budget": random_cost,
        },
        "reward": random_cost - selector_cost,
        "notes": notes,
    }


def reweight_episode_record(
    record: dict[str, Any],
    rho_box: float,
    rho_obj: float | None = None,
    episode_id_suffix: str = "",
    notes_suffix: str = "",
) -> dict[str, Any]:
    """Return a copy of an episode row with reward/cost recomputed.

    This is intended for sensitivity or collision-aware training buffers where
    the downstream metrics and selected samples are unchanged, but the scalar
    reward definition is deliberately different.
    """

    if record.get("schema") != "selector_bench.stage2_episode.v0":
        raise ValueError(f"Unexpected episode schema: {record.get('schema')}")
    if rho_box < 0:
        raise ValueError("rho_box must be non-negative.")

    existing_definition = record.get("reward_definition", {})
    resolved_rho_obj = (
        float(existing_definition.get("rho_obj", 0.05))
        if rho_obj is None
        else float(rho_obj)
    )
    if resolved_rho_obj < 0:
        raise ValueError("rho_obj must be non-negative.")

    output = json.loads(json.dumps(record))
    if episode_id_suffix:
        output["episode_id"] = f"{output['episode_id']}{episode_id_suffix}"

    selector_metrics = {
        key: float(value)
        for key, value in output["metrics"]["selector"].items()
        if isinstance(value, (int, float))
    }
    random_metrics = {
        key: float(value)
        for key, value in output["metrics"]["random_same_budget"].items()
        if isinstance(value, (int, float))
    }
    selector_cost = reward_cost(
        selector_metrics,
        rho_box=float(rho_box),
        rho_obj=resolved_rho_obj,
    )
    random_cost = reward_cost(
        random_metrics,
        rho_box=float(rho_box),
        rho_obj=resolved_rho_obj,
    )

    output["reward_definition"] = {
        **existing_definition,
        "cost": "J(S)=L2+rho_box*obj_box_col+rho_obj*obj_col",
        "reward": "R=J(random_same_budget)-J(selector)",
        "rho_box": float(rho_box),
        "rho_obj": resolved_rho_obj,
        "collision_units": existing_definition.get(
            "collision_units",
            "record values exactly as emitted by standalone eval; choose rho values accordingly",
        ),
    }
    output["cost"] = {
        "selector": selector_cost,
        "random_same_budget": random_cost,
    }
    output["reward"] = random_cost - selector_cost
    if notes_suffix:
        original_notes = str(output.get("notes", ""))
        output["notes"] = (original_notes + " " + notes_suffix).strip()
    return output


def append_episode_record(path: str | Path, record: dict[str, Any]) -> None:
    if record.get("schema") != "selector_bench.stage2_episode.v0":
        raise ValueError("Unexpected episode schema.")
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


def write_episode_schema(path: str | Path) -> dict[str, Any]:
    schema = {
        "schema": "selector_bench.stage2_episode.v0",
        "description": "One JSONL row per selector/random same-budget DiffusionDrive feedback episode.",
        "required_keys": [
            "episode_id",
            "selector_selection",
            "random_same_budget_selection",
            "train_work_dirs",
            "eval_work_dirs",
            "metrics",
            "reward_definition",
            "cost",
            "reward",
            "notes",
        ],
        "metric_keys": ["l2", "obj_box_col", "obj_col", "runtime_s", "peak_gpu_mb"],
        "reward": "J(S)=L2+rho_box*obj_box_col+rho_obj*obj_col; R=J(random_same_budget)-J(selector)",
    }
    write_json(path, schema)
    return schema



def _load_feedback_episodes(
    episode_jsonl: str | Path,
    store: FeatureStore,
    min_abs_reward: float,
) -> list[dict[str, Any]]:
    rows = read_jsonl(episode_jsonl)
    manifest_tokens = set(store.manifest.tokens(split="train"))
    episodes: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for row in rows:
        if row.get("schema") != "selector_bench.stage2_episode.v0":
            raise ValueError(f"Unexpected episode schema in {episode_jsonl}: {row.get('schema')}")
        episode_id = str(row.get("episode_id", ""))
        if episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode_id)
        reward = float(row["reward"])
        if abs(reward) < min_abs_reward:
            continue
        selector_path = Path(row["selector_selection"]["path"])
        random_path = Path(row["random_same_budget_selection"]["path"])
        selector_artifact = read_json(selector_path)
        random_artifact = read_json(random_path)
        selector_tokens = [
            token
            for token in selector_artifact["selection"]["selected_tokens"]
            if token in manifest_tokens
        ]
        random_tokens = [
            token
            for token in random_artifact["selection"]["selected_tokens"]
            if token in manifest_tokens
        ]
        if not selector_tokens or not random_tokens:
            continue
        episodes.append(
            {
                "episode_id": episode_id,
                "reward": reward,
                "selector_tokens": selector_tokens,
                "random_tokens": random_tokens,
                "selector_selection_path": selector_path,
                "random_selection_path": random_path,
                "selector_selection_hash": row["selector_selection"]["selection_hash"],
                "random_selection_hash": row["random_same_budget_selection"]["selection_hash"],
            }
        )
    return episodes


def _resolve_reward_scale(episodes: list[dict[str, Any]], reward_scale: float | None) -> float:
    if reward_scale is not None:
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive when provided.")
        return float(reward_scale)
    rewards = np.asarray([abs(float(episode["reward"])) for episode in episodes], dtype=np.float64)
    positive = rewards[rewards > 1e-12]
    if not positive.size:
        return 1.0
    return float(max(np.median(positive), 1e-6))


def _build_feedback_token_cache(
    store: FeatureStore,
    selector: dict[str, Any],
    episodes: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, tuple[np.ndarray, int]]:
    all_tokens = sorted(
        {
            token
            for episode in episodes
            for token in [*episode["selector_tokens"], *episode["random_tokens"]]
        }
    )
    raw = _build_feature_matrix(store, all_tokens)
    x = (raw - mean) / np.maximum(std, 1e-6)
    domain_cols = _domain_columns_for_tokens(store, selector, all_tokens)
    return {
        token: (x[idx : idx + 1], int(domain_cols[idx]))
        for idx, token in enumerate(all_tokens)
    }


def _build_negative_feedback_token_offsets(
    episodes: list[dict[str, Any]],
    negative_token_penalty: float,
    negative_random_bonus: float,
    negative_reward_max: float,
    target_std: float,
) -> dict[str, float]:
    if negative_token_penalty <= 0 and negative_random_bonus <= 0:
        return {}
    offsets: dict[str, float] = {}
    score_scale = max(float(target_std), 1e-6)
    for episode in episodes:
        if float(episode["reward"]) > negative_reward_max:
            continue
        weight = abs(float(episode.get("target_gap", 0.0)))
        if weight <= 0:
            continue
        selector_delta = -negative_token_penalty * weight * score_scale
        random_delta = negative_random_bonus * weight * score_scale
        if negative_token_penalty > 0:
            for token in episode["selector_tokens"]:
                offsets[token] = offsets.get(token, 0.0) + selector_delta
        if negative_random_bonus > 0:
            for token in episode["random_tokens"]:
                offsets[token] = offsets.get(token, 0.0) + random_delta
    return {token: float(value) for token, value in sorted(offsets.items()) if abs(value) > 1e-12}


def _arrays_for_tokens(
    token_cache: dict[str, tuple[np.ndarray, int]],
    tokens: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([token_cache[token][0] for token in tokens], axis=0)
    domain_cols = np.asarray([token_cache[token][1] for token in tokens], dtype=np.int64)
    return x, domain_cols


def _train_feedback_episode(
    params: dict[str, np.ndarray],
    token_cache: dict[str, tuple[np.ndarray, int]],
    selector_tokens: list[str],
    random_tokens: list[str],
    target_gap: float,
    lr: float,
    l2: float,
    use_domain_bias: bool,
) -> dict[str, float]:
    x_selector, cols_selector = _arrays_for_tokens(token_cache, selector_tokens)
    x_random, cols_random = _arrays_for_tokens(token_cache, random_tokens)
    x = np.concatenate([x_selector, x_random], axis=0)
    domain_cols = np.concatenate([cols_selector, cols_random], axis=0)
    pred, hidden = _forward(params, x, domain_cols, use_domain_bias)
    selector_count = max(1, x_selector.shape[0])
    random_count = max(1, x_random.shape[0])
    selector_mean = float(np.mean(pred[:selector_count]))
    random_mean = float(np.mean(pred[selector_count:]))
    gap = selector_mean - random_mean
    err = gap - target_gap
    loss = float(err**2 + _l2_penalty(params, l2))

    grad_pred = np.zeros_like(pred)
    grad_pred[:selector_count, 0] = 2.0 * err / selector_count
    grad_pred[selector_count:, 0] = -2.0 * err / random_count
    _apply_prediction_gradient(params, x, domain_cols, hidden, grad_pred, lr, l2, use_domain_bias)
    return {"loss": loss, "gap": gap, "target_gap": target_gap}


def _feedback_metrics(
    params: dict[str, np.ndarray],
    token_cache: dict[str, tuple[np.ndarray, int]],
    episodes: list[dict[str, Any]],
    use_domain_bias: bool,
    token_score_offsets: dict[str, float] | None = None,
    target_std: float = 1.0,
) -> dict[str, float]:
    rows = []
    correct = 0
    comparable = 0
    for episode in episodes:
        x_selector, cols_selector = _arrays_for_tokens(token_cache, episode["selector_tokens"])
        x_random, cols_random = _arrays_for_tokens(token_cache, episode["random_tokens"])
        pred_selector = _predict_normalized(params, x_selector, cols_selector, use_domain_bias)
        pred_random = _predict_normalized(params, x_random, cols_random, use_domain_bias)
        if token_score_offsets:
            score_scale = max(float(target_std), 1e-6)
            selector_offsets = np.asarray(
                [float(token_score_offsets.get(token, 0.0)) / score_scale for token in episode["selector_tokens"]],
                dtype=np.float64,
            ).reshape(-1, 1)
            random_offsets = np.asarray(
                [float(token_score_offsets.get(token, 0.0)) / score_scale for token in episode["random_tokens"]],
                dtype=np.float64,
            ).reshape(-1, 1)
            pred_selector = pred_selector + selector_offsets
            pred_random = pred_random + random_offsets
        gap = float(np.mean(pred_selector) - np.mean(pred_random))
        target = float(episode["target_gap"])
        reward = float(episode["reward"])
        rows.append((gap, target))
        if abs(reward) > 1e-12:
            comparable += 1
            if gap == 0.0 or np.sign(gap) == np.sign(reward):
                correct += 1
    if not rows:
        return {
            "feedback_mse": 0.0,
            "preference_accuracy": 0.0,
            "mean_score_gap": 0.0,
            "mean_target_gap": 0.0,
        }
    gaps = np.asarray([row[0] for row in rows], dtype=np.float64)
    targets = np.asarray([row[1] for row in rows], dtype=np.float64)
    return {
        "feedback_mse": float(np.mean((gaps - targets) ** 2)),
        "preference_accuracy": float(correct / comparable) if comparable else 0.0,
        "mean_score_gap": float(np.mean(gaps)),
        "mean_target_gap": float(np.mean(targets)),
    }


def _score_teacher_selector(store: FeatureStore, teacher_selector_path: str | Path, tokens: list[str]) -> np.ndarray:
    provider = OurSelectorScoreProvider(store.manifest, store, teacher_selector_path)
    return provider.score(tokens).astype(np.float64)


def _build_feature_matrix(store: FeatureStore, tokens: list[str]) -> np.ndarray:
    rows = []
    for token in tokens:
        record = store.manifest.get(token)
        features = store.get(token)
        rows.append(_feature_vector(record, features))
    return np.asarray(rows, dtype=np.float64)


def _feature_vector(record: ManifestRecord, features: FeatureRecord) -> np.ndarray:
    blocks = [
        np.asarray(features.scene_hidden, dtype=np.float64).reshape(-1),
        np.asarray(features.interaction_hidden, dtype=np.float64).reshape(-1),
    ]
    if features.traj_hidden is not None:
        blocks.append(np.asarray(features.traj_hidden, dtype=np.float64).reshape(-1))
    else:
        blocks.append(np.zeros(0, dtype=np.float64))
    blocks.extend(
        [
            np.asarray(features.traj_raw, dtype=np.float64).reshape(-1),
            np.asarray([record.teacher_loss], dtype=np.float64),
            np.asarray([record.teacher_uncertainty], dtype=np.float64),
            np.asarray([record.agent_count / 32.0], dtype=np.float64),
            np.asarray([record.ego_speed / 30.0], dtype=np.float64),
        ]
    )
    return np.concatenate(blocks, axis=0)


def _split_indices(num_items: int, holdout_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_items, dtype=np.int64)
    rng.shuffle(indices)
    holdout_count = int(round(num_items * holdout_ratio)) if num_items >= 10 else 0
    holdout_count = min(max(holdout_count, 0), max(num_items - 1, 0))
    holdout = np.sort(indices[:holdout_count])
    train = np.sort(indices[holdout_count:])
    return train, holdout


def _init_params(
    input_dim: int,
    hidden_dim: int,
    domain_count: int,
    rng: np.random.Generator,
    use_domain_bias: bool,
    use_domain_residual_head: bool,
) -> dict[str, np.ndarray]:
    scale1 = np.sqrt(2.0 / max(input_dim + hidden_dim, 1))
    scale2 = np.sqrt(2.0 / max(hidden_dim + 1, 1))
    params = {
        "w1": rng.normal(0.0, scale1, size=(input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim), dtype=np.float64),
        "w2": rng.normal(0.0, scale2, size=(hidden_dim, 1)),
        "b2": np.zeros((1, 1), dtype=np.float64),
        "domain_bias": np.zeros((domain_count,), dtype=np.float64)
        if use_domain_bias
        else np.zeros((0,), dtype=np.float64),
    }
    params["domain_residual_w"] = (
        rng.normal(0.0, scale2, size=(domain_count, hidden_dim))
        if use_domain_residual_head
        else np.zeros((0, hidden_dim), dtype=np.float64)
    )
    return params

def _train_epoch(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    domain_cols: np.ndarray,
    lr: float,
    l2: float,
    use_domain_bias: bool,
) -> dict[str, float]:
    pred, hidden = _forward(params, x, domain_cols, use_domain_bias)
    err = pred - y
    n = max(1, x.shape[0])
    loss = float(np.mean(err**2) + _l2_penalty(params, l2))
    grad_pred = (2.0 / n) * err
    _apply_prediction_gradient(params, x, domain_cols, hidden, grad_pred, lr, l2, use_domain_bias)
    return {"loss": loss}

def _forward(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    domain_cols: np.ndarray,
    use_domain_bias: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ params["w1"] + params["b1"])
    pred = hidden @ params["w2"] + params["b2"]
    if use_domain_bias and params["domain_bias"].size:
        known_bias = (domain_cols >= 0) & (domain_cols < params["domain_bias"].shape[0])
        pred = pred.copy()
        pred[known_bias, 0] += params["domain_bias"][domain_cols[known_bias]]
    residual = params.get("domain_residual_w")
    if residual is not None and residual.size:
        known_residual = (domain_cols >= 0) & (domain_cols < residual.shape[0])
        pred = pred.copy()
        pred[known_residual, 0] += np.sum(
            hidden[known_residual] * residual[domain_cols[known_residual]],
            axis=1,
        )
    return pred, hidden

def _apply_prediction_gradient(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    domain_cols: np.ndarray,
    hidden: np.ndarray,
    grad_pred: np.ndarray,
    lr: float,
    l2: float,
    use_domain_bias: bool,
) -> None:
    grad_w2 = hidden.T @ grad_pred + l2 * params["w2"]
    grad_b2 = grad_pred.sum(axis=0, keepdims=True)
    grad_hidden = grad_pred @ params["w2"].T

    residual = params.get("domain_residual_w")
    grad_residual = None
    if residual is not None and residual.size:
        grad_residual = np.zeros_like(residual)
        for row_idx, (col, grad) in enumerate(zip(domain_cols, grad_pred[:, 0])):
            if 0 <= col < residual.shape[0]:
                grad_hidden[row_idx] += grad * residual[col]
                grad_residual[col] += grad * hidden[row_idx]
        grad_residual += l2 * residual

    grad_hidden_raw = grad_hidden * (1.0 - hidden**2)
    grad_w1 = x.T @ grad_hidden_raw + l2 * params["w1"]
    grad_b1 = grad_hidden_raw.sum(axis=0, keepdims=True)

    params["w1"] -= lr * grad_w1
    params["b1"] -= lr * grad_b1
    params["w2"] -= lr * grad_w2
    params["b2"] -= lr * grad_b2
    if grad_residual is not None:
        params["domain_residual_w"] -= lr * grad_residual
    if use_domain_bias and params["domain_bias"].size:
        grad_domain = np.zeros_like(params["domain_bias"])
        for col, grad in zip(domain_cols, grad_pred[:, 0]):
            if 0 <= col < grad_domain.shape[0]:
                grad_domain[col] += grad
        params["domain_bias"] -= lr * grad_domain


def _l2_penalty(params: dict[str, np.ndarray], l2: float) -> float:
    penalty = np.mean(params["w1"] ** 2) + np.mean(params["w2"] ** 2)
    residual = params.get("domain_residual_w")
    if residual is not None and residual.size:
        penalty += float(np.mean(residual**2))
    return float(l2 * penalty)

def _predict_normalized(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    domain_cols: np.ndarray,
    use_domain_bias: bool,
) -> np.ndarray:
    return _forward(params, x, domain_cols, use_domain_bias)[0]


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean((pred - target) ** 2))


def _params_to_json(params: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: value.astype(float).tolist() for key, value in params.items()}


def _params_from_json(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    params = {key: np.asarray(value, dtype=np.float64) for key, value in raw.items()}
    if "domain_residual_w" not in params:
        hidden_dim = int(params["w2"].shape[0]) if "w2" in params else 0
        params["domain_residual_w"] = np.zeros((0, hidden_dim), dtype=np.float64)
    return params


def _sample_gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=size)
    return -np.log(-np.log(u))

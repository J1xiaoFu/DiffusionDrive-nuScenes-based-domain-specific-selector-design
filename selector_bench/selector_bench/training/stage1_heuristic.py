from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.utils.io import write_json


DEFAULT_WEIGHTS = {
    "bias": 0.0,
    "scene_norm": 0.06,
    "interaction_norm": 0.10,
    "traj_speed_mean": 0.015,
    "teacher_loss": 0.45,
    "teacher_uncertainty": 0.30,
    "agent_count": 0.08,
    "ego_speed": 0.04,
}


def build_stage1_selector_checkpoint(
    feature_dir: str | Path,
    output_path: str | Path,
    run_id: str = "stage1_lightweight",
) -> dict[str, Any]:
    store = FeatureStore.from_feature_dir(feature_dir)
    manifest = store.manifest
    train_records = manifest.filter(split="train")

    domain_scores: dict[int, list[float]] = {}
    for record in train_records:
        score = (
            0.45 * record.teacher_loss
            + 0.30 * record.teacher_uncertainty
            + 0.08 * (record.agent_count / 32.0)
            + 0.04 * (record.ego_speed / 30.0)
        )
        domain_scores.setdefault(record.domain_id, []).append(float(score))

    global_mean = float(
        np.mean([score for scores in domain_scores.values() for score in scores])
    ) if domain_scores else 0.0
    domain_bias = {
        str(domain_id): float(np.mean(scores) - global_mean)
        for domain_id, scores in sorted(domain_scores.items())
    }

    selector = {
        "format": "selector_bench.linear_selector.v1",
        "weights": DEFAULT_WEIGHTS,
        "domain_bias": domain_bias,
        "feature_spec": {
            "scene_hidden": "l2_norm_sqrt_dim",
            "interaction_hidden": "l2_norm_sqrt_dim",
            "traj_speed_mean": "mean_abs_column_2",
            "teacher_loss": "manifest_value",
            "teacher_uncertainty": "manifest_value",
        },
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="selector_checkpoint",
            run_id=run_id,
            config={"stage": "stage1_lightweight"},
            input_paths=[Path(feature_dir) / "manifest.jsonl"],
            summary={"num_train_records": len(train_records), "num_domains": len(domain_scores)},
        ),
        "selector": selector,
    }
    write_json(output_path, artifact)
    return artifact

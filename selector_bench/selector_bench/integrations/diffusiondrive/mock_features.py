from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.manifest import Manifest, ManifestRecord
from selector_bench.utils.io import ensure_dir, write_json
from selector_bench.utils.seed import seed_everything


DOMAIN_NAMES = ["urban", "campus", "highway", "night", "rain"]


def generate_mock_feature_cache(
    output_dir: str | Path,
    max_train_samples: int = 120,
    max_val_samples: int = 30,
    num_domains: int = 3,
    seed: int = 1,
    traj_steps: int = 8,
    traj_dim: int = 6,
    scene_dim: int = 256,
    interaction_dim: int = 256,
) -> dict[str, Any]:
    if num_domains < 1:
        raise ValueError("num_domains must be >= 1.")

    output = ensure_dir(output_dir)
    rng = seed_everything(seed)
    records: list[ManifestRecord] = []
    split_sizes = {"train": max_train_samples, "val": max_val_samples}

    for split, size in split_sizes.items():
        feature_split_dir = ensure_dir(output / "features" / split)
        for idx in range(size):
            domain_id = idx % num_domains
            domain_name = DOMAIN_NAMES[domain_id] if domain_id < len(DOMAIN_NAMES) else f"domain_{domain_id}"
            token = f"{split}_{idx:06d}"
            scene_token = f"{split}_scene_{idx // 20:04d}"

            domain_scale = 1.0 + 0.15 * domain_id
            agent_count = int(rng.integers(2 + domain_id, 18 + domain_id))
            ego_speed = float(max(0.0, rng.normal(7.0 + domain_id * 2.5, 1.5)))

            traj_raw = rng.normal(0.0, 1.0, size=(traj_steps, traj_dim)).astype(np.float32)
            traj_raw[:, 0:2] = np.cumsum(traj_raw[:, 0:2], axis=0)
            if traj_dim > 2:
                traj_raw[:, 2] = np.abs(rng.normal(ego_speed, 0.8, size=traj_steps))

            scene_hidden = rng.normal(
                loc=0.08 * domain_id,
                scale=domain_scale,
                size=scene_dim,
            ).astype(np.float32)
            interaction_hidden = rng.normal(
                loc=0.03 * agent_count,
                scale=1.0 + 0.05 * domain_id,
                size=interaction_dim,
            ).astype(np.float32)
            traj_summary = np.concatenate(
                [traj_raw.mean(axis=0), traj_raw.std(axis=0), np.zeros(32, dtype=np.float32)]
            )[:32].astype(np.float32)

            base_loss = 0.35 + 0.035 * domain_id + 0.008 * agent_count
            teacher_loss = float(max(0.01, rng.normal(base_loss, 0.035)))
            teacher_uncertainty = float(
                np.clip(rng.normal(0.20 + 0.025 * domain_id + 0.006 * agent_count, 0.04), 0.0, 1.0)
            )

            relative_path = Path("features") / split / f"{token}.npz"
            np.savez_compressed(
                feature_split_dir / f"{token}.npz",
                traj_raw=traj_raw,
                traj_hidden=traj_summary,
                scene_hidden=scene_hidden,
                interaction_hidden=interaction_hidden,
                domain_id=np.asarray(domain_id, dtype=np.int64),
                teacher_loss=np.asarray(teacher_loss, dtype=np.float32),
                teacher_uncertainty=np.asarray(teacher_uncertainty, dtype=np.float32),
            )

            records.append(
                ManifestRecord(
                    sample_token=token,
                    scene_token=scene_token,
                    timestamp=idx,
                    split=split,
                    domain_id=domain_id,
                    domain_name=domain_name,
                    location="mock_city",
                    weather_tag="clear" if domain_id % 2 == 0 else "rain",
                    time_tag="day" if domain_id % 3 else "night",
                    route_cmd=["straight", "left", "right"][domain_id % 3],
                    ego_speed=ego_speed,
                    agent_count=agent_count,
                    feature_path=str(relative_path),
                    teacher_loss=teacher_loss,
                    teacher_uncertainty=teacher_uncertainty,
                    metadata={"source": "mock_diffusiondrive_features"},
                )
            )

    manifest = Manifest.from_records(records)
    manifest_path = output / "manifest.jsonl"
    manifest.save(manifest_path)

    stats = {
        "manifest": manifest.summary(),
        "feature_shapes": {
            "traj_raw": [traj_steps, traj_dim],
            "traj_hidden": [32],
            "scene_hidden": [scene_dim],
            "interaction_hidden": [interaction_dim],
        },
        "seed": seed,
        "lightweight": True,
    }
    write_json(output / "feature_stats.json", stats)
    write_json(
        output / "artifact_metadata.json",
        make_artifact_metadata(
            artifact_type="feature_cache",
            run_id="mock_features",
            config={
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "num_domains": num_domains,
                "seed": seed,
            },
            summary=stats["manifest"],
        ),
    )
    return stats

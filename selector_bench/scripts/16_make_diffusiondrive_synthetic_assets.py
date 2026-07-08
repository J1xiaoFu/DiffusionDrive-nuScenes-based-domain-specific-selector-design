"""Create tiny synthetic DiffusionDrive assets for code-path smoke tests.

These files only satisfy model construction shape requirements. They are not
valid statistical anchors and must not be used for reporting model quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _save(path: Path, array: np.ndarray, force: bool) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        existing = np.load(path)
        return {
            "path": str(path),
            "shape": list(existing.shape),
            "created": False,
            "reason": "exists",
        }
    np.save(path, array.astype(np.float32))
    return {
        "path": str(path),
        "shape": list(array.shape),
        "created": True,
        "reason": "synthetic_smoke_asset",
    }


def make_det_anchors() -> np.ndarray:
    xs = np.linspace(-45.0, 45.0, 30)
    ys = np.linspace(-45.0, 45.0, 30)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    z = np.zeros((grid.shape[0], 1), dtype=np.float32)
    defaults = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    defaults = np.repeat(defaults[None], grid.shape[0], axis=0)
    return np.concatenate([grid, z, defaults], axis=1)


def make_map_anchors() -> np.ndarray:
    centers_x = np.linspace(-25.0, 25.0, 10)
    centers_y = np.linspace(-50.0, 50.0, 10)
    centers = np.stack(np.meshgrid(centers_x, centers_y, indexing="xy"), axis=-1)
    centers = centers.reshape(-1, 2)
    line = np.stack([np.zeros(20), np.linspace(-4.0, 4.0, 20)], axis=-1)
    return centers[:, None, :] + line[None, :, :]


def make_motion_anchors() -> np.ndarray:
    anchors = np.zeros((10, 6, 12, 2), dtype=np.float32)
    t = np.linspace(0.5, 6.0, 12)
    lateral_templates = np.array([-2.0, -1.0, -0.25, 0.25, 1.0, 2.0])
    class_speed = np.array([1.2, 1.0, 0.7, 0.9, 0.8, 0.0, 0.9, 0.6, 0.35, 0.0])
    for cls_idx, speed in enumerate(class_speed):
        for mode_idx, lateral in enumerate(lateral_templates):
            anchors[cls_idx, mode_idx, :, 0] = speed * t
            anchors[cls_idx, mode_idx, :, 1] = lateral * (t / t.max()) ** 1.4
    return anchors


def make_plan_anchors() -> np.ndarray:
    anchors = np.zeros((3, 6, 6, 2), dtype=np.float32)
    t = np.linspace(0.5, 3.0, 6)
    speed_templates = np.array([0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
    command_bias = np.array([-2.0, 0.0, 2.0])
    for cmd_idx, bias in enumerate(command_bias):
        for mode_idx, speed in enumerate(speed_templates):
            anchors[cmd_idx, mode_idx, :, 0] = speed * t
            anchors[cmd_idx, mode_idx, :, 1] = bias * (t / t.max()) ** 1.5
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="DiffusionDrive",
        help="Path to the DiffusionDrive checkout.",
    )
    parser.add_argument(
        "--metadata",
        default="selector_bench/artifacts/diffusiondrive_assets/synthetic_anchor_metadata.json",
        help="Where to write generation metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files. Do not use this over real anchors.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    kmeans_dir = repo / "data" / "kmeans"
    records = [
        _save(kmeans_dir / "kmeans_det_900.npy", make_det_anchors(), args.force),
        _save(kmeans_dir / "kmeans_map_100.npy", make_map_anchors(), args.force),
        _save(kmeans_dir / "kmeans_motion_6.npy", make_motion_anchors(), args.force),
        _save(kmeans_dir / "kmeans_plan_6.npy", make_plan_anchors(), args.force),
    ]

    metadata = {
        "asset_type": "synthetic_diffusiondrive_smoke_anchors",
        "warning": "Shape-compatible smoke assets only; not valid for quality metrics.",
        "repo": str(repo),
        "records": records,
    }
    metadata_path = Path(args.metadata).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

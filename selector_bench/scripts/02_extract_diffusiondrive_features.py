#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.integrations.diffusiondrive.mock_features import generate_mock_feature_cache
from selector_bench.integrations.diffusiondrive.real_mini_features import (
    generate_real_mini_feature_cache,
)
from selector_bench.integrations.diffusiondrive.teacher_features import (
    generate_teacher_feature_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Generate a small deterministic cache.")
    parser.add_argument(
        "--real-mini",
        action="store_true",
        help="Generate deterministic metadata features from real nuScenes mini info files.",
    )
    parser.add_argument(
        "--teacher-score",
        action="store_true",
        help="Run a DiffusionDrive checkpoint over labeled samples and extract genuine losses/hooks.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--diffusiondrive-root",
        default=str(ROOT.parent / "DiffusionDrive"),
        help="Path to the DiffusionDrive checkout for --teacher-score.",
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "artifacts"
            / "diffusiondrive_configs"
            / "diffusiondrive_small_stage2_smokeplus_mini_real_anchor_100iter.py"
        ),
        help="DiffusionDrive config for --teacher-score.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "work_dirs" / "mini_real_anchor_100iter" / "iter_100.pth"),
        help="DiffusionDrive checkpoint for --teacher-score.",
    )
    parser.add_argument("--split", default="train", choices=["train"], help="Dataset split for --teacher-score.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output cache.")
    parser.add_argument(
        "--train-info",
        default=str(
            ROOT.parent
            / "DiffusionDrive"
            / "data"
            / "infos"
            / "mini"
            / "nuscenes_infos_train.pkl"
        ),
    )
    parser.add_argument(
        "--val-info",
        default=str(
            ROOT.parent
            / "DiffusionDrive"
            / "data"
            / "infos"
            / "mini"
            / "nuscenes_infos_val.pkl"
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=120)
    parser.add_argument("--max-val-samples", type=int, default=30)
    parser.add_argument("--num-domains", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--traj-steps", type=int, default=8)
    parser.add_argument("--traj-dim", type=int, default=6)
    parser.add_argument("--scene-dim", type=int, default=256)
    parser.add_argument("--interaction-dim", type=int, default=256)
    args = parser.parse_args()

    modes = [args.mock, args.real_mini, args.teacher_score]
    if sum(bool(mode) for mode in modes) != 1:
        print("Choose exactly one extraction mode: --mock, --real-mini, or --teacher-score.")
        return 2

    if args.teacher_score:
        stats = generate_teacher_feature_cache(
            output_dir=args.output,
            diffusiondrive_root=args.diffusiondrive_root,
            config=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            max_samples=args.max_train_samples,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    elif args.real_mini:
        stats = generate_real_mini_feature_cache(
            output_dir=args.output,
            train_info=args.train_info,
            val_info=args.val_info,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            scene_dim=args.scene_dim,
            interaction_dim=args.interaction_dim,
            seed=args.seed,
        )
    else:
        stats = generate_mock_feature_cache(
            output_dir=args.output,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            num_domains=args.num_domains,
            seed=args.seed,
            traj_steps=args.traj_steps,
            traj_dim=args.traj_dim,
            scene_dim=args.scene_dim,
            interaction_dim=args.interaction_dim,
        )
    print(f"Wrote feature cache: {args.output}")
    print(stats["manifest"])
    if stats.get("teacher_signal_source"):
        print(
            {
                "teacher_signal_source": stats["teacher_signal_source"],
                "teacher_signal_caveat": stats.get("teacher_signal_caveat"),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

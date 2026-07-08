#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.utils.io import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DiffusionDrive info pkl files from a nuScenes partial plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--diffusiondrive-root",
        default=str(ROOT.parent / "DiffusionDrive"),
    )
    parser.add_argument(
        "--root-path",
        default=str(ROOT.parent / "DiffusionDrive" / "data" / "nuscenes_trainval_partial_20260706"),
    )
    parser.add_argument(
        "--out-path",
        default=str(ROOT.parent / "DiffusionDrive" / "data" / "infos" / "trainval_partial_20260706"),
    )
    parser.add_argument("--can-bus-root", default=None)
    parser.add_argument("--info-prefix", default="nuscenes_partial_1600_400")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--allow-missing-map-canbus-for-smoke", action="store_true")
    args = parser.parse_args()

    diffusiondrive_root = Path(args.diffusiondrive_root).resolve()
    sys.path.insert(0, str(diffusiondrive_root))
    from tools.data_converter.nuscenes_converter import create_nuscenes_infos

    plan = read_json(args.plan)
    train_scene_names = plan["selection"]["train"]["scene_names"]
    val_scene_names = plan["selection"]["val"]["scene_names"]
    can_bus_root = args.can_bus_root or args.root_path

    create_nuscenes_infos(
        root_path=args.root_path,
        out_path=args.out_path,
        can_bus_root_path=can_bus_root,
        info_prefix=args.info_prefix,
        version=plan.get("version", "v1.0-trainval"),
        max_sweeps=args.max_sweeps,
        allow_missing_map_canbus_for_smoke=args.allow_missing_map_canbus_for_smoke,
        train_scene_names=train_scene_names,
        val_scene_names=val_scene_names,
    )
    print(
        {
            "out_path": args.out_path,
            "info_prefix": args.info_prefix,
            "train_scenes": len(train_scene_names),
            "val_scenes": len(val_scene_names),
            "max_sweeps": args.max_sweeps,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build Time-only, Geography-only and Natural-mixed nuScenes research streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval-dir", type=Path, required=True)
    parser.add_argument("--official-population", choices=("train", "val"), default="train")
    parser.add_argument(
        "--geography-stage",
        action="append",
        required=True,
        help="comma-separated locations; repeat exactly three times",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from selector_bench.continual.nuscenes_cl_protocol import (
        build_geography_only_protocol,
        build_log_sessions,
        build_natural_mixed_protocol,
        build_time_only_protocol,
    )
    from selector_bench.integrations.diffusiondrive.nuscenes_partial import (
        build_scene_summaries,
        load_nuscenes_trainval_split_names,
        load_trainval_metadata,
    )

    if len(args.geography_stage) != 3:
        raise SystemExit("--geography-stage must be supplied exactly three times")
    split_names = load_nuscenes_trainval_split_names()
    if not split_names["train"] or split_names["_source"]["name"].startswith("fallback"):
        raise SystemExit("official nuScenes-devkit split lists are required; fallback is forbidden")
    metadata = load_trainval_metadata(args.trainval_dir)
    _, split_info = build_scene_summaries(metadata, split_names=split_names)
    if split_info["fallback_split_used"] or split_info["unknown_scene_count"]:
        raise SystemExit("official split mapping is incomplete or used a fallback")
    sessions = build_log_sessions(
        metadata,
        eligible_scene_names=set(split_names[args.official_population]),
    )
    location_groups = tuple(
        tuple(part.strip() for part in raw.split(",") if part.strip())
        for raw in args.geography_stage
    )
    protocols = (
        build_time_only_protocol(sessions, seed=args.seed),
        build_geography_only_protocol(
            sessions, location_groups=location_groups, seed=args.seed
        ),
        build_natural_mixed_protocol(sessions, seed=args.seed),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        atomic_json(args.output_dir / f"{protocol['protocol_id']}.json", protocol)
    summary = {
        "schema": "selector_bench.nuscenes_controlled_cl_collection.v1",
        "official_population": args.official_population,
        "atomic_unit": "complete_nuscenes_log_token",
        "protocols": [
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_type": protocol["protocol_type"],
                "logs": len(protocol["session_inventory"]),
                "stage_samples": [
                    sum(
                        stage["splits"][split]["sample_count"]
                        for split in ("train", "audit", "test")
                    )
                    for stage in protocol["stages"]
                ],
            }
            for protocol in protocols
        ],
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

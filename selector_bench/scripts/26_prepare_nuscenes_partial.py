#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.integrations.diffusiondrive.nuscenes_partial import (
    build_blob_manifest,
    build_partial_plan,
    extract_blob_files,
    extract_blob_files_resumable,
    extract_metadata_archive,
    extract_zip_archive,
)
from selector_bench.utils.io import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or materialize a whole-scene, domain-balanced nuScenes trainval subset."
    )
    parser.add_argument(
        "--source-trainval-dir",
        default="/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval",
    )
    parser.add_argument(
        "--target-root",
        default=str(ROOT.parent / "DiffusionDrive" / "data" / "nuscenes_trainval_partial"),
    )
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--target-train-samples", type=int, default=1600)
    parser.add_argument("--target-val-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--include-sweeps", action="store_true")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Only write the selection plan.")
    parser.add_argument("--extract-metadata", action="store_true")
    parser.add_argument("--extract-blobs", action="store_true")
    parser.add_argument("--build-blob-manifest", action="store_true")
    parser.add_argument("--extract-expansions", action="store_true")
    parser.add_argument(
        "--map-expansion-zip",
        default="/root/autodl-pub/nuScenes/Mapexpansion/nuScenes-map-expansion-v1.3.zip",
    )
    parser.add_argument(
        "--can-bus-zip",
        default="/root/autodl-pub/nuScenes/CANbusexpansion/can_bus.zip",
    )
    parser.add_argument("--blob-manifest-output")
    parser.add_argument("--blob-manifest-input")
    args = parser.parse_args()

    plan = build_partial_plan(
        source_trainval_dir=args.source_trainval_dir,
        target_root=args.target_root,
        target_train_samples=args.target_train_samples,
        target_val_samples=args.target_val_samples,
        seed=args.seed,
        version=args.version,
        include_sweeps=args.include_sweeps,
        max_sweeps=args.max_sweeps if args.include_sweeps else 0,
    )
    write_json(args.plan_output, plan)

    extraction: dict[str, object] = {}
    if args.extract_metadata:
        extract_metadata_archive(args.source_trainval_dir, args.target_root)
        extraction["metadata_extracted"] = True
    if args.extract_expansions:
        extraction["map_expansion"] = extract_zip_archive(args.map_expansion_zip, Path(args.target_root) / "maps")
        extraction["can_bus"] = extract_zip_archive(args.can_bus_zip, args.target_root)

    manifest = None
    if args.build_blob_manifest and not args.extract_blobs:
        manifest = build_blob_manifest(
            args.source_trainval_dir,
            plan["required_data"]["filenames"],
            output_path=args.blob_manifest_output,
        )
        extraction["blob_manifest"] = {
            "path": args.blob_manifest_output,
            "archive_count_scanned": manifest["archive_count_scanned"],
            "found_count": manifest["found_count"],
            "missing_count": manifest["missing_count"],
        }
    elif args.extract_blobs and args.blob_manifest_input:
        manifest = read_json(args.blob_manifest_input)
        extraction["blob_manifest"] = {
            "path": args.blob_manifest_input,
            "archive_count_scanned": manifest["archive_count_scanned"],
            "found_count": manifest["found_count"],
            "missing_count": manifest["missing_count"],
        }
    if args.extract_blobs:
        if args.blob_manifest_input:
            extraction["blobs"] = extract_blob_files(
                args.source_trainval_dir,
                args.target_root,
                plan["required_data"]["filenames"],
                blob_manifest=manifest,
            )
        elif args.blob_manifest_output:
            extraction["blobs"] = extract_blob_files_resumable(
                args.source_trainval_dir,
                args.target_root,
                plan["required_data"]["filenames"],
                manifest_output_path=args.blob_manifest_output,
            )
        else:
            extraction["blobs"] = extract_blob_files(
                args.source_trainval_dir,
                args.target_root,
                plan["required_data"]["filenames"],
                blob_manifest=manifest,
            )
    if extraction:
        plan["extraction"] = extraction
        write_json(args.plan_output, plan)

    _print_summary(plan, args.dry_run)
    return 0


def _print_summary(plan: dict, dry_run: bool) -> None:
    train = plan["selection"]["train"]
    val = plan["selection"]["val"]
    print(f"Wrote nuScenes partial plan: {plan['schema']}")
    print(f"Dry run: {dry_run}")
    print(
        {
            "train_scenes": train["scene_count"],
            "train_samples": train["sample_count"],
            "train_locations": train["sample_count_by_location"],
            "val_scenes": val["scene_count"],
            "val_samples": val["sample_count"],
            "val_locations": val["sample_count_by_location"],
            "required_files": plan["required_data"]["filename_count"],
        }
    )
    for warning in plan.get("warnings", []):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())

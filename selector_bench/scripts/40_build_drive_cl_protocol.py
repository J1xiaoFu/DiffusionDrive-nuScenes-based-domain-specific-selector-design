#!/usr/bin/env python3
"""Build leakage-safe NAVSIM Failure-Patch-CL and Chronological-CL manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _write_stage_yaml(path: Path, stage: dict[str, object]) -> None:
    lines: list[str] = []
    for split in ("train", "audit", "test"):
        lines.append(f"{split}_logs:")
        for log_name in stage["splits"][split]["logs"]:  # type: ignore[index]
            lines.append(f"  - {log_name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-index",
        type=Path,
        default=Path(
            "/home/rguo/rap_workspace/exp/training_cache/valid_cache_train.pkl"
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path(
            "/home/rguo/rap_workspace/fused_cluster_labels_with_dd_raw_pdms.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-root-metadata-sha256", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--safety-fraction", type=float, default=0.3)
    parser.add_argument("--efficiency-fraction", type=float, default=0.3)
    parser.add_argument("--safe-pool-fraction", type=float, default=0.5)
    parser.add_argument("--progress-quantile", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from selector_bench.continual.navsim_protocol import (
        build_chronological_protocol,
        build_failure_patch_protocol,
        load_cache_inventory,
    )

    for path in (args.cache_index, args.metrics_csv):
        if not path.is_file():
            raise SystemExit(f"required source is missing: {path}")
    inventory = load_cache_inventory(args.cache_index)
    chronological = build_chronological_protocol(
        inventory,
        cache_index=args.cache_index,
        dataset_version=args.dataset_version,
        dataset_root_metadata_sha256=args.dataset_root_metadata_sha256,
        seed=args.seed,
    )
    failure_patch = build_failure_patch_protocol(
        inventory,
        cache_index=args.cache_index,
        metrics_csv=args.metrics_csv,
        dataset_version=args.dataset_version,
        dataset_root_metadata_sha256=args.dataset_root_metadata_sha256,
        seed=args.seed,
        safety_fraction=args.safety_fraction,
        efficiency_fraction=args.efficiency_fraction,
        safe_pool_fraction=args.safe_pool_fraction,
        progress_quantile=args.progress_quantile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for protocol in (chronological, failure_patch):
        protocol_dir = args.output_dir / protocol["protocol_id"]
        protocol_dir.mkdir(parents=True, exist_ok=True)
        _write_json(protocol_dir / "manifest.json", protocol)
        for stage in protocol["stages"]:
            _write_stage_yaml(
                protocol_dir / f"stage_{stage['stage_index']}_{stage['name']}.yaml",
                stage,
            )
        print(
            json.dumps(
                {
                    "protocol_id": protocol["protocol_id"],
                    "content_sha256": protocol["content_sha256"],
                    "inventory": protocol["inventory"],
                    "stages": [
                        {
                            "stage_index": stage["stage_index"],
                            "name": stage["name"],
                            "token_counts": {
                                split: stage["splits"][split]["token_count"]
                                for split in ("train", "audit", "test")
                            },
                        }
                        for stage in protocol["stages"]
                    ],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()

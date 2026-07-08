#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.utils.io import ensure_dir, read_json


def load_selected_tokens(selection_path: Path) -> set[str]:
    artifact = read_json(selection_path)
    tokens = artifact["selection"]["selected_tokens"]
    return set(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a DiffusionDrive nuScenes info pkl filtered by a selection artifact."
    )
    parser.add_argument("--source-info", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional sidecar describing the exported subset.",
    )
    args = parser.parse_args()

    source_info = Path(args.source_info)
    selection_path = Path(args.selection)
    output = Path(args.output)
    selected_tokens = load_selected_tokens(selection_path)

    with source_info.open("rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"]
    subset_infos = [info for info in infos if info.get("token") in selected_tokens]
    found_tokens = {info.get("token") for info in subset_infos}
    missing = sorted(selected_tokens - found_tokens)
    if missing:
        raise ValueError(
            f"{len(missing)} selected tokens were not found in {source_info}: {missing[:5]}"
        )

    exported = dict(payload)
    exported["infos"] = subset_infos
    metadata = dict(exported.get("metadata", {}))
    metadata.update(
        {
            "subset_source_info": str(source_info),
            "subset_source_selection": str(selection_path),
            "subset_selected_count": len(subset_infos),
            "subset_order": "source_info_order",
        }
    )
    exported["metadata"] = metadata

    ensure_dir(output.parent)
    with output.open("wb") as f:
        pickle.dump(exported, f)

    sidecar = {
        "source_info": str(source_info),
        "selection": str(selection_path),
        "output": str(output),
        "source_count": len(infos),
        "selected_count": len(subset_infos),
        "subset_order": "source_info_order",
    }
    if args.metadata_json:
        sidecar_path = Path(args.metadata_json)
        ensure_dir(sidecar_path.parent)
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export an exact protocol cell for deterministic NAVSIM PDM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--split", choices=("audit", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol_manifest.read_text())
    stage = next(
        item for item in protocol["stages"] if item["stage_index"] == args.stage_index
    )
    cell = stage["splits"][args.split]
    tokens = sorted(cell["tokens"])
    logs = sorted(cell["logs"])
    if not tokens or not logs:
        raise SystemExit("selected protocol cell is empty")
    token_file = args.output_dir / "tokens.txt"
    atomic_text(token_file, "\n".join(tokens) + "\n")
    receipt = {
        "schema": "selector_bench.drive_cl_eval_cell.v1",
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "protocol_manifest_sha256": sha256(args.protocol_manifest),
        "protocol_content_sha256": protocol["content_sha256"],
        "stage_index": args.stage_index,
        "stage_name": stage["name"],
        "split": args.split,
        "session_count": cell["session_count"],
        "log_count": len(logs),
        "token_count": len(tokens),
        "token_file": str(token_file.resolve()),
        "token_file_sha256": sha256(token_file),
        "deterministic_seed_env": "PDM_DETERMINISTIC_GLOBAL_SEED",
        "final_test_policy": (
            "configuration_must_be_frozen_before_use"
            if args.split == "test"
            else "method_and_trigger_selection_allowed"
        ),
    }
    atomic_text(
        args.output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a portable, training-only derivative of a Drive-CL manifest."""

from __future__ import annotations

import argparse
import copy
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


def tokens_sha256(tokens: list[str]) -> str:
    return hashlib.sha256(("\n".join(tokens) + "\n").encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    return args


def main() -> None:
    args = parse_args()
    source = json.loads(args.manifest.read_text())
    portable = copy.deepcopy(source)
    portable["source_schema"] = portable["schema"]
    portable["schema"] = "selector_bench.drive_cl_training_manifest.v1"
    portable["source_manifest_sha256"] = sha256(args.manifest)
    portable["portable_derivative"] = True

    training_tokens: list[str] = []
    stage_counts: dict[str, int] = {}
    for stage in portable["stages"]:
        stage_index = str(stage["stage_index"])
        current = list(stage["splits"]["train"]["tokens"])
        training_tokens.extend(current)
        stage_counts[stage_index] = len(current)
        for cell in stage["splits"].values():
            # Training needs only token/log membership.  Cluster maps and session
            # lists remain in the authoritative full manifest used for statistics.
            cell.pop("token_to_log", None)
            cell.pop("sessions", None)

    if len(training_tokens) != len(set(training_tokens)):
        raise RuntimeError("training tokens overlap across stages")
    training_tokens = sorted(training_tokens)
    portable["remote_training_cache_contract"] = {
        "cache_only_train_splits": True,
        "audit_and_test_excluded": True,
        "stage_token_counts": stage_counts,
        "total_token_count": len(training_tokens),
        "sorted_token_sha256": tokens_sha256(training_tokens),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(portable, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
                "source_manifest_sha256": portable["source_manifest_sha256"],
                **portable["remote_training_cache_contract"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

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


def _membership_counts(cell: dict[str, object], label: str) -> dict[str, int]:
    arrays = {}
    for name in ("sessions", "logs", "tokens"):
        values = cell.get(name)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise RuntimeError(f"{label} {name} membership must be a unique list")
        arrays[name] = values
    token_to_log = cell.get("token_to_log")
    if not isinstance(token_to_log, dict) or set(token_to_log) != set(arrays["tokens"]):
        raise RuntimeError(f"{label} token_to_log must exactly cover tokens")
    counts = {
        "session_count": len(arrays["sessions"]),
        "log_count": len(arrays["logs"]),
        "token_count": len(arrays["tokens"]),
    }
    if any(cell.get(name) != value for name, value in counts.items()):
        raise RuntimeError(f"{label} declared membership counts are inconsistent")
    return counts


def build_training_derivative(
    source: dict[str, object], *, source_manifest_sha256: str
) -> dict[str, object]:
    """Remove every non-training identity while retaining only count summaries."""

    portable = copy.deepcopy(source)
    portable["source_schema"] = portable["schema"]
    portable["schema"] = "selector_bench.drive_cl_training_manifest.v1"
    portable["source_manifest_sha256"] = source_manifest_sha256
    portable["portable_derivative"] = True

    training_tokens: list[str] = []
    stage_counts: dict[str, int] = {}
    excluded_counts: dict[str, dict[str, dict[str, int]]] = {}
    for stage in portable["stages"]:
        stage_index = str(stage["stage_index"])
        splits = stage.get("splits")
        if not isinstance(splits, dict) or set(splits) != {"train", "audit", "test"}:
            raise RuntimeError(f"stage {stage_index} requires exact train/audit/test splits")
        current = list(splits["train"]["tokens"])
        _membership_counts(splits["train"], f"stage {stage_index} train")
        training_tokens.extend(current)
        stage_counts[stage_index] = len(current)
        excluded_counts[stage_index] = {}
        for split in ("audit", "test"):
            cell = splits[split]
            excluded_counts[stage_index][split] = _membership_counts(
                cell, f"stage {stage_index} {split}"
            )
            cell["sessions"] = []
            cell["logs"] = []
            cell["tokens"] = []
            cell["token_to_log"] = {}
            cell["session_count"] = 0
            cell["log_count"] = 0
            cell["token_count"] = 0
        # Failure-Patch selection metrics are session-addressed development
        # evidence and are not needed by a training-only derivative.
        stage.pop("selection_metrics", None)

    if len(training_tokens) != len(set(training_tokens)):
        raise RuntimeError("training tokens overlap across stages")
    training_tokens = sorted(training_tokens)
    portable["remote_training_cache_contract"] = {
        "cache_only_train_splits": True,
        "audit_and_test_excluded": True,
        "stage_token_counts": stage_counts,
        "excluded_membership_count_summary": excluded_counts,
        "total_token_count": len(training_tokens),
        "sorted_token_sha256": tokens_sha256(training_tokens),
    }
    return portable


def main() -> None:
    args = parse_args()
    source = json.loads(args.manifest.read_text())
    portable = build_training_derivative(
        source, source_manifest_sha256=sha256(args.manifest)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(portable, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
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
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

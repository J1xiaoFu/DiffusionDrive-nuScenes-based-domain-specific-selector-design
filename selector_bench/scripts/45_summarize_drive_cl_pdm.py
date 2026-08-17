#!/usr/bin/env python3
"""Validate and summarize one sealed/audit Drive-CL PDM evaluation cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in (args.csv, args.receipt, args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.bootstrap_repetitions <= 0:
        parser.error("bootstrap repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    receipt = json.loads(args.receipt.read_text())
    manifest = json.loads(Path(receipt["protocol_manifest"]).read_text())
    stage = next(
        item for item in manifest["stages"] if item["stage_index"] == receipt["stage_index"]
    )
    cell = stage["splits"][receipt["split"]]
    expected = set(cell["tokens"])
    rows: dict[str, dict[str, float]] = {}
    invalid: list[str] = []
    with args.csv.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            token = (raw.get("token") or "").strip()
            if not token or token == "average":
                continue
            valid = (raw.get("valid") or "").strip().lower() in {"true", "1"}
            if not valid:
                invalid.append(token)
                continue
            values: dict[str, float] = {}
            for key, raw_value in raw.items():
                if key in {None, "", "token", "valid"} or not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values[key] = value
            rows[token] = values
    observed = set(rows) | set(invalid)
    if observed != expected:
        raise RuntimeError(
            f"PDM cell mismatch: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    metrics = sorted({key for values in rows.values() for key in values})
    token_to_log = cell["token_to_log"]
    rng = random.Random(args.seed)
    summaries: dict[str, dict[str, float]] = {}
    for metric in metrics:
        log_values: dict[str, list[float]] = defaultdict(list)
        token_values = []
        for token, values in rows.items():
            if metric in values:
                value = values[metric]
                token_values.append(value)
                log_values[token_to_log[token]].append(value)
        log_means = [float(np.mean(values)) for values in log_values.values()]
        if not log_means:
            continue
        bootstrap = [
            float(np.mean([rng.choice(log_means) for _ in log_means]))
            for _ in range(args.bootstrap_repetitions)
        ]
        summaries[metric] = {
            "token_mean": float(np.mean(token_values)),
            "log_cluster_mean": float(np.mean(log_means)),
            "ci95_low": float(np.percentile(bootstrap, 2.5)),
            "ci95_high": float(np.percentile(bootstrap, 97.5)),
            "valid_tokens": len(token_values),
            "log_clusters": len(log_means),
        }
    payload = {
        "schema": "selector_bench.drive_cl_pdm_cell_summary.v1",
        "protocol_content_sha256": manifest["content_sha256"],
        "stage_index": receipt["stage_index"],
        "stage_name": receipt["stage_name"],
        "split": receipt["split"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "csv": str(args.csv.resolve()),
        "csv_sha256": sha256(args.csv),
        "expected_tokens": len(expected),
        "valid_tokens": len(rows),
        "invalid_tokens": len(invalid),
        "bootstrap_unit": "complete_log",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "metrics": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

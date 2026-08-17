#!/usr/bin/env python3
"""Paired, log-clustered comparisons for Drive-CL PDM audit cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from selector_bench.continual.navsim_protocol import session_id_from_log
from selector_bench.continual.statistics import (
    holm_adjusted_pvalues,
    paired_session_bootstrap,
    read_pdm_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=CSV")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"candidate CSV does not exist: {path}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=candidate, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in (args.baseline_csv, args.receipt):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.bootstrap_repetitions <= 0:
        parser.error("bootstrap repetitions must be positive")
    names = [name for name, _ in args.candidate]
    if len(names) != len(set(names)):
        parser.error("candidate names must be unique")
    return args


def main() -> None:
    args = parse_args()
    receipt = json.loads(args.receipt.read_text())
    manifest_path = Path(receipt["protocol_manifest"])
    manifest = json.loads(manifest_path.read_text())
    stage = next(
        item for item in manifest["stages"] if item["stage_index"] == receipt["stage_index"]
    )
    cell = stage["splits"][receipt["split"]]
    expected = set(cell["tokens"])
    token_to_log = cell["token_to_log"]
    baseline = read_pdm_rows(args.baseline_csv, expected_tokens=expected)

    comparisons = {}
    for offset, (name, path) in enumerate(args.candidate):
        rows = read_pdm_rows(path, expected_tokens=expected)
        metrics = sorted(
            set.intersection(
                *(set(baseline[token]) & set(rows[token]) for token in sorted(expected))
            )
        )
        comparisons[name] = {
            "candidate_csv": str(path.resolve()),
            "candidate_csv_sha256": sha256(path),
            "candidate_minus_baseline": paired_session_bootstrap(
                baseline,
                rows,
                token_to_log,
                metrics=metrics,
                repetitions=args.bootstrap_repetitions,
                seed=args.seed + offset,
            ),
        }

    family = {
        (name, metric): values["bootstrap_two_sided_pvalue"]
        for name, comparison in comparisons.items()
        for metric, values in comparison["candidate_minus_baseline"].items()
    }
    for (name, metric), value in holm_adjusted_pvalues(family).items():
        comparisons[name]["candidate_minus_baseline"][metric][
            "family_holm_adjusted_pvalue"
        ] = value

    payload = {
        "schema": "selector_bench.drive_cl_paired_pdm_comparison.v2",
        "protocol_content_sha256": manifest["content_sha256"],
        "stage_index": receipt["stage_index"],
        "stage_name": receipt["stage_name"],
        "split": receipt["split"],
        "expected_tokens": len(expected),
        "session_clusters": len(
            {session_id_from_log(value) for value in token_to_log.values()}
        ),
        "multiplicity_family": "all_candidate_by_metric_comparisons_in_this_report",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "baseline_csv": str(args.baseline_csv.resolve()),
        "baseline_csv_sha256": sha256(args.baseline_csv),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

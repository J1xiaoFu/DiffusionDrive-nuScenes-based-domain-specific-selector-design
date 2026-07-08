#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.utils.io import ensure_dir, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.selection_dir).glob("*.json")):
        artifact = read_json(path)
        selection = artifact["selection"]
        rows.append(
            {
                "file": path.name,
                "baseline": selection["baseline"],
                "score_source": selection["score_source"],
                "budget_ratio": selection["budget_ratio"],
                "selected_count": selection["selected_count"],
                "selection_hash": selection["selection_hash"],
                "domain_counts": selection["domain_counts"],
            }
        )

    output = Path(args.output)
    ensure_dir(output.parent)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "baseline",
                "score_source",
                "budget_ratio",
                "selected_count",
                "selection_hash",
                "domain_counts",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage1_heuristic import build_stage1_selector_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="stage1_lightweight")
    args = parser.parse_args()

    artifact = build_stage1_selector_checkpoint(
        feature_dir=args.features,
        output_path=args.output,
        run_id=args.run_id,
    )
    summary = artifact["metadata"]["summary"]
    print(f"Wrote selector checkpoint: {args.output}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

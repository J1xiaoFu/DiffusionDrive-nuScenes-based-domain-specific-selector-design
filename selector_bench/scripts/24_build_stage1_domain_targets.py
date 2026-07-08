#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage1_domain_targets import build_stage1_domain_targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build config-driven domain/cluster-specific Stage 1 target scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--labels-output", default=None)
    args = parser.parse_args()

    result = build_stage1_domain_targets(
        config_path=args.config,
        output_path=args.output,
        labels_output_path=args.labels_output,
    )
    metadata = result["metadata"]["metadata"]
    print(f"Wrote Stage 1 target scores: {result['metadata']['target_scores_path']}")
    print(metadata["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

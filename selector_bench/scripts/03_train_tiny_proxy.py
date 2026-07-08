#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.models.tiny_proxy_planner import train_tiny_proxy_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--run-id", default="tiny_proxy_seed_lightweight")
    args = parser.parse_args()

    artifact = train_tiny_proxy_seed(
        feature_dir=args.features,
        output_path=args.output,
        ridge_alpha=args.ridge_alpha,
        run_id=args.run_id,
    )
    print(f"Wrote tiny proxy seed: {args.output}")
    print(artifact["metadata"]["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage2_lightweight import train_selector_stage2_lightweight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ema-alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default="stage2_lightweight")
    args = parser.parse_args()

    artifact = train_selector_stage2_lightweight(
        feature_dir=args.features,
        proxy_path=args.proxy,
        init_selector_path=args.init,
        output_path=args.output,
        iterations=args.iterations,
        budget=args.budget,
        lr=args.lr,
        ema_alpha=args.ema_alpha,
        seed=args.seed,
        run_id=args.run_id,
    )
    print(f"Wrote Stage2 selector checkpoint: {args.output}")
    print(artifact["metadata"]["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.gpu_smoke import run_gpu_smoke_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default=str(ROOT / "reports" / "gpu_smoke_report.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    artifact = run_gpu_smoke_training(
        feature_dir=args.features,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=args.device,
    )
    print(f"Wrote GPU smoke report: {args.output}")
    print(artifact["metadata"]["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

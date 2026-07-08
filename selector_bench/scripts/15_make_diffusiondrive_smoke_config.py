#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.integrations.diffusiondrive.smoke_config import make_smoke_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(
            ROOT.parent
            / "DiffusionDrive"
            / "projects"
            / "configs"
            / "diffusiondrive_configs"
            / "diffusiondrive_small_stage2.py"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "diffusiondrive_configs" / "diffusiondrive_small_stage2_smoke_1gpu.py"),
    )
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()

    output = make_smoke_config(
        source_config=args.source,
        output_config=args.output,
        max_iters=args.max_iters,
        workers_per_gpu=args.workers_per_gpu,
    )
    print(f"Wrote DiffusionDrive smoke config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

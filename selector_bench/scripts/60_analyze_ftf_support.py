#!/usr/bin/env python3
"""Compute explanatory support coverage, distribution shift and conflict."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--radius-quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from selector_bench.continual.support_analysis import analyze_support

    with np.load(args.arrays, allow_pickle=False) as arrays:
        required = {"old_features", "current_features", "old_gradient", "current_gradient"}
        if not required.issubset(arrays.files):
            raise SystemExit(f"NPZ must contain {sorted(required)}")
        analysis = analyze_support(
            arrays["old_features"],
            arrays["current_features"],
            arrays["old_gradient"],
            arrays["current_gradient"],
            k=args.k,
            radius_quantile=args.radius_quantile,
        )
    payload = {
        "schema": "selector_bench.ftf_opd_support_analysis.v1",
        "role": "post-hoc explanation only; never used by the functional trigger",
        **asdict(analysis),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

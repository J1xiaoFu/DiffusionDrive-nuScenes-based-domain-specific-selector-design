#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.domain_clustering import fit_stage1_domain_clusters


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit deterministic NumPy k-means domain labels for Stage 1 target building.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels-output", default=None)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--profile-output", default=None)
    args = parser.parse_args()

    result = fit_stage1_domain_clusters(
        config_path=args.config,
        labels_output=args.labels_output,
        model_output=args.model_output,
        profile_output=args.profile_output,
    )
    model = result["cluster_model"]["cluster_model"]
    print(
        {
            "num_clusters": model["num_clusters"],
            "iterations": model["iterations"],
            "cluster_counts": model["cluster_counts"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

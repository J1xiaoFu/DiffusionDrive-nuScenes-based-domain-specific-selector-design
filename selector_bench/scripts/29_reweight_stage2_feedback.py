#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage2_neural import reweight_episode_record
from selector_bench.utils.io import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a Stage 2 feedback JSONL with rewards recomputed under a new collision weight."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rho-box", type=float, required=True)
    parser.add_argument("--rho-obj", type=float, default=None)
    parser.add_argument("--episode-id-suffix", default="")
    parser.add_argument("--notes-suffix", default="")
    args = parser.parse_args()

    rows = [
        reweight_episode_record(
            row,
            rho_box=args.rho_box,
            rho_obj=args.rho_obj,
            episode_id_suffix=args.episode_id_suffix,
            notes_suffix=args.notes_suffix,
        )
        for row in read_jsonl(args.input_jsonl)
    ]
    write_jsonl(args.output_jsonl, rows)
    rewards = [float(row["reward"]) for row in rows]
    print(f"Wrote reweighted Stage 2 feedback JSONL: {args.output_jsonl}")
    print(
        {
            "rows": len(rows),
            "rho_box": args.rho_box,
            "rho_obj": rows[0]["reward_definition"]["rho_obj"] if rows else args.rho_obj,
            "reward_min": min(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

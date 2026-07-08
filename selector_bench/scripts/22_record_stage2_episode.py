#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage2_neural import append_episode_record, build_episode_record


def _metrics(raw: str) -> dict[str, float]:
    data = json.loads(raw)
    return {key: float(value) for key, value in data.items()}


def _path_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    data = json.loads(raw)
    return {str(key): str(value) for key, value in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append one future Stage 2 selector/random feedback episode to an episode.jsonl buffer."
    )
    parser.add_argument("--episode-jsonl", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--selector-selection", required=True)
    parser.add_argument("--random-selection", required=True)
    parser.add_argument("--selector-metrics-json", required=True, help='Example: {"l2":4.0,"obj_box_col":0.2,"obj_col":0.0}')
    parser.add_argument("--random-metrics-json", required=True)
    parser.add_argument("--rho-box", type=float, default=0.05)
    parser.add_argument("--rho-obj", type=float, default=0.05)
    parser.add_argument("--train-work-dirs-json", default=None)
    parser.add_argument("--eval-work-dirs-json", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    record = build_episode_record(
        episode_id=args.episode_id,
        selector_selection_path=args.selector_selection,
        random_selection_path=args.random_selection,
        selector_metrics=_metrics(args.selector_metrics_json),
        random_metrics=_metrics(args.random_metrics_json),
        rho_box=args.rho_box,
        rho_obj=args.rho_obj,
        train_work_dirs=_path_map(args.train_work_dirs_json),
        eval_work_dirs=_path_map(args.eval_work_dirs_json),
        notes=args.notes,
    )
    append_episode_record(args.episode_jsonl, record)
    print(f"Appended Stage 2 episode: {args.episode_jsonl}")
    print({"episode_id": record["episode_id"], "reward": record["reward"], "cost": record["cost"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

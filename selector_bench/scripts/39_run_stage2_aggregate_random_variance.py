#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from selector_bench.core.feature_store import FeatureStore
from selector_bench.training.stage2_high_frequency import (
    _ensure_random_selection,
    _run_diffusiondrive_one,
    export_diffusiondrive_info_subset,
    normalize_eval_metrics,
    write_subset_train_config,
)
from selector_bench.training.stage2_neural import build_episode_record, reward_cost
from selector_bench.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate random-seed variance for a fixed aggregate selector selection. "
            "The selector arm is read from an existing aggregate real-eval report and "
            "is not retrained."
        )
    )
    parser.add_argument(
        "--selector-report",
        default=str(
            ROOT
            / "artifacts"
            / "stage2_aggregate_selected_real_eval"
            / "stage2_loss_duel_aggregate320_real_eval_lr3e5_100iter_20260707_report.json"
        ),
    )
    parser.add_argument("--result-index", type=int, default=0)
    parser.add_argument(
        "--features",
        default=str(ROOT / "artifacts" / "features" / "trainval_partial_teacher_100iter_train_20260706"),
    )
    parser.add_argument(
        "--source-info",
        default=str(
            REPO_ROOT
            / "DiffusionDrive"
            / "data"
            / "infos"
            / "trainval_partial_20260706"
            / "nuscenes_partial_1600_400_infos_train.pkl"
        ),
    )
    parser.add_argument("--diffusiondrive-root", default=str(REPO_ROOT / "DiffusionDrive"))
    parser.add_argument(
        "--base-config",
        default=str(
            ROOT
            / "artifacts"
            / "diffusiondrive_configs"
            / "diffusiondrive_small_stage2_trainval_partial_1600_400_real_anchor_100iter.py"
        ),
    )
    parser.add_argument(
        "--planner-init-checkpoint",
        default=str(REPO_ROOT / "DiffusionDrive" / "ckpts" / "sparsedrive_stage1.pth"),
    )
    parser.add_argument("--planner-lr", type=float, default=3e-5)
    parser.add_argument("--planner-max-iters", type=int, default=100)
    parser.add_argument("--random-seeds", default="7200,7201,7202")
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--rho-box", type=float, default=0.0)
    parser.add_argument("--rho-obj", type=float, default=0.0)
    parser.add_argument("--diffusiondrive-python", default=None)
    parser.add_argument("--wrapper-python", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "stage2_aggregate_random_variance"),
    )
    parser.add_argument("--run-id", default="stage2_loss_duel_r40_random_variance_lr3e5_100iter_20260708")
    parser.add_argument(
        "--subset-info-dir",
        default=str(REPO_ROOT / "DiffusionDrive" / "data" / "infos" / "trainval_partial_selector_subsets"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "artifacts" / "diffusiondrive_configs"),
    )
    parser.add_argument("--work-dir-root", default=str(ROOT / "work_dirs"))
    parser.add_argument("--episode-jsonl", default=None)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_random_variance(args)
    report_path = Path(args.output_dir) / f"{args.run_id}_report.json"
    print(f"Wrote aggregate random-variance report: {report_path}")
    print(report["summary"])
    return 0


def run_random_variance(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_dir(args.output_dir)
    selection_root = ensure_dir(output_root / "selections")
    sidecar_root = ensure_dir(output_root / "subset_metadata")
    subset_root = ensure_dir(args.subset_info_dir)
    config_root = ensure_dir(args.config_dir)
    work_root = ensure_dir(args.work_dir_root)
    episode_jsonl = Path(args.episode_jsonl) if args.episode_jsonl else output_root / f"{args.run_id}_episodes.jsonl"
    if args.restart and episode_jsonl.exists():
        write_jsonl(episode_jsonl, [])

    selector_report = read_json(args.selector_report)
    selector_result = selector_report["results"][args.result_index]
    selector_selection_path = Path(selector_result["selector_selection"])
    selector_metrics = normalize_eval_metrics(selector_result["selector_metrics"])
    selector_tokens = set(read_json(selector_selection_path)["selection"]["selected_tokens"])
    selection_count = int(selector_result["selected_count"])
    store = FeatureStore.from_feature_dir(args.features, cache=True)

    rows: list[dict[str, Any]] = []
    for seed in _parse_seeds(args.random_seeds):
        name = f"{args.run_id}_seed{seed}_random{selection_count}"
        paths = _paths(
            name=name,
            selection_root=selection_root,
            sidecar_root=sidecar_root,
            subset_root=subset_root,
            config_root=config_root,
            work_root=work_root,
            max_iters=args.planner_max_iters,
        )
        random_selection = _ensure_random_selection(
            store=store,
            output_path=paths["random_selection"],
            run_id=name,
            selection_count=selection_count,
            split=args.split,
            seed=seed,
            exclude_tokens=selector_tokens,
            feature_dir=args.features,
        )
        _ensure_random_subset_and_config(
            source_info=args.source_info,
            random_selection=paths["random_selection"],
            random_info=paths["random_info"],
            random_sidecar=paths["random_sidecar"],
            base_config=args.base_config,
            random_config=paths["random_config"],
            diffusiondrive_root=Path(args.diffusiondrive_root),
            planner_init_checkpoint=args.planner_init_checkpoint,
            planner_lr=args.planner_lr,
            planner_max_iters=args.planner_max_iters,
        )
        random_metrics = normalize_eval_metrics(
            _run_diffusiondrive_one(
                config=paths["random_config"],
                train_work_dir=paths["random_work_dir"],
                eval_work_dir=paths["random_eval_work_dir"],
                diffusiondrive_root=Path(args.diffusiondrive_root),
                train_seed=args.train_seed,
                eval_seed=args.eval_seed,
                diffusiondrive_python=args.diffusiondrive_python,
                wrapper_python=args.wrapper_python,
            )
        )
        episode_id = f"{args.run_id}_seed{seed}"
        record = build_episode_record(
            episode_id=episode_id,
            selector_selection_path=selector_selection_path,
            random_selection_path=paths["random_selection"],
            selector_metrics=selector_metrics,
            random_metrics=random_metrics,
            rho_box=args.rho_box,
            rho_obj=args.rho_obj,
            train_work_dirs={
                "selector": str(selector_result.get("selector_work_dir", "")),
                "random_same_budget": str(paths["random_work_dir"]),
            },
            eval_work_dirs={
                "selector": str(selector_result.get("selector_eval_work_dir", "")),
                "random_same_budget": str(paths["random_eval_work_dir"]),
            },
            notes=(
                "Random-seed variance check for a fixed aggregate selector arm. "
                "Selector metrics are reused from the aggregate real-eval report."
            ),
        )
        _append_or_replace_episode(episode_jsonl, record)
        random_tokens = set(random_selection["selection"]["selected_tokens"])
        rows.append(
            {
                "seed": seed,
                "selected_count": selection_count,
                "random_selection": str(paths["random_selection"]),
                "random_hash": random_selection["selection"]["selection_hash"],
                "random_info": str(paths["random_info"]),
                "random_config": str(paths["random_config"]),
                "random_work_dir": str(paths["random_work_dir"]),
                "random_eval_work_dir": str(paths["random_eval_work_dir"]),
                "selector_random_overlap_count": len(selector_tokens & random_tokens),
                "random_metrics": random_metrics,
                "selector_cost": reward_cost(selector_metrics, args.rho_box, args.rho_obj),
                "random_cost": reward_cost(random_metrics, args.rho_box, args.rho_obj),
                "reward": float(record["reward"]),
            }
        )

    report = {
        "schema": "selector_bench.stage2_aggregate_random_variance.v0",
        "run_id": args.run_id,
        "episode_jsonl": str(episode_jsonl),
        "selector_report": str(args.selector_report),
        "selector_selection": str(selector_selection_path),
        "selector_metrics": selector_metrics,
        "config": {
            "planner_init_checkpoint": str(args.planner_init_checkpoint),
            "planner_lr": args.planner_lr,
            "planner_max_iters": args.planner_max_iters,
            "random_seeds": _parse_seeds(args.random_seeds),
            "split": args.split,
            "train_seed": args.train_seed,
            "eval_seed": args.eval_seed,
            "rho_box": args.rho_box,
            "rho_obj": args.rho_obj,
        },
        "random_results": rows,
        "summary": _summary(selector_metrics, rows),
    }
    write_json(output_root / f"{args.run_id}_report.json", report)
    return report


def _parse_seeds(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("random-seeds cannot be empty.")
    return list(dict.fromkeys(values))


def _paths(
    name: str,
    selection_root: Path,
    sidecar_root: Path,
    subset_root: Path,
    config_root: Path,
    work_root: Path,
    max_iters: int,
) -> dict[str, Path]:
    return {
        "random_selection": selection_root / f"{name}.json",
        "random_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{name}.pkl",
        "random_sidecar": sidecar_root / f"{name}_subset.json",
        "random_config": config_root / f"dd_{name}.py",
        "random_work_dir": work_root / f"{name}_subset_{max_iters}iter",
        "random_eval_work_dir": work_root / f"{name}_subset_{max_iters}iter_eval_standalone",
    }


def _ensure_random_subset_and_config(
    source_info: str | Path,
    random_selection: Path,
    random_info: Path,
    random_sidecar: Path,
    base_config: str | Path,
    random_config: Path,
    diffusiondrive_root: Path,
    planner_init_checkpoint: str | Path | None,
    planner_lr: float | None,
    planner_max_iters: int | None,
) -> None:
    if not random_info.exists():
        export_diffusiondrive_info_subset(source_info, random_selection, random_info, random_sidecar)
    if not random_config.exists():
        write_subset_train_config(
            base_config,
            random_config,
            random_info,
            diffusiondrive_root,
            planner_init_checkpoint=planner_init_checkpoint,
            planner_lr=planner_lr,
            planner_max_iters=planner_max_iters,
        )


def _append_or_replace_episode(path: Path, record: dict[str, Any]) -> None:
    rows = read_jsonl(path) if path.exists() else []
    rows = [row for row in rows if row.get("episode_id") != record["episode_id"]]
    rows.append(record)
    write_jsonl(path, rows)


def _summary(selector_metrics: dict[str, float], rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_l2 = [float(row["random_metrics"]["l2"]) for row in rows]
    selector_l2 = float(selector_metrics["l2"])
    rewards = [float(row["reward"]) for row in rows]
    if not rows:
        return {"status": "empty", "completed_random_seeds": 0}
    return {
        "status": "complete",
        "completed_random_seeds": len(rows),
        "selector_l2": selector_l2,
        "random_l2_mean": float(statistics.mean(random_l2)),
        "random_l2_stdev": float(statistics.stdev(random_l2)) if len(random_l2) > 1 else 0.0,
        "random_l2_min": float(min(random_l2)),
        "random_l2_max": float(max(random_l2)),
        "reward_mean": float(statistics.mean(rewards)),
        "reward_min": float(min(rewards)),
        "reward_max": float(max(rewards)),
        "selector_better_count": sum(1 for value in random_l2 if selector_l2 < value),
    }


if __name__ == "__main__":
    raise SystemExit(main())

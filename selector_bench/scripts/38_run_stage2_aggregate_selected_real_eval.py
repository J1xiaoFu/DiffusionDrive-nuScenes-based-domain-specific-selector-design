#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.training.stage2_high_frequency import (
    _ensure_random_selection,
    _run_diffusiondrive_one,
    export_diffusiondrive_info_subset,
    normalize_eval_metrics,
    write_subset_train_config,
)
from selector_bench.training.stage2_neural import build_episode_record, reward_cost
from selector_bench.utils.hashing import hash_jsonable
from selector_bench.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cumulative selector tokens from loss-duel episodes against "
            "a same-size random baseline with real DiffusionDrive train/eval."
        )
    )
    parser.add_argument(
        "--loss-episodes",
        default=str(
            ROOT
            / "artifacts"
            / "stage2_loss_duel_inprocess"
            / "stage2_loss_duel_inprocess_stage1_lr3e5_r40x8_train_test_20260707_episodes.jsonl"
        ),
    )
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
    parser.add_argument(
        "--round-cutoffs",
        default="40",
        help="Comma-separated cumulative loss-duel round cutoffs, e.g. 10,20,30,40.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--random-seed", type=int, default=7100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--rho-box", type=float, default=0.0)
    parser.add_argument("--rho-obj", type=float, default=0.0)
    parser.add_argument("--diffusiondrive-python", default=None)
    parser.add_argument("--wrapper-python", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "stage2_aggregate_selected_real_eval"),
    )
    parser.add_argument("--run-id", default="stage2_loss_duel_aggregate320_real_eval_20260707")
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
    parser.add_argument(
        "--allow-random-overlap",
        action="store_true",
        help="Allow aggregate random baseline to overlap selector tokens. Default is disjoint.",
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_aggregate_real_eval(args)
    report_path = Path(args.output_dir) / f"{args.run_id}_report.json"
    print(f"Wrote aggregate real-eval report: {report_path}")
    print(report["summary"])
    return 0


def run_aggregate_real_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_dir(args.output_dir)
    selection_root = ensure_dir(output_root / "selections")
    sidecar_root = ensure_dir(output_root / "subset_metadata")
    subset_root = ensure_dir(args.subset_info_dir)
    config_root = ensure_dir(args.config_dir)
    work_root = ensure_dir(args.work_dir_root)
    episode_jsonl = Path(args.episode_jsonl) if args.episode_jsonl else output_root / f"{args.run_id}_episodes.jsonl"
    if args.restart and episode_jsonl.exists():
        write_jsonl(episode_jsonl, [])

    store = FeatureStore.from_feature_dir(args.features, cache=True)
    rows = _loss_episode_rows(args.loss_episodes)
    cutoffs = _parse_cutoffs(args.round_cutoffs, max_round=len(rows))
    completed: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        selector_tokens = _cumulative_selector_tokens(rows[:cutoff])
        name = f"{args.run_id}_r{cutoff:02d}_{len(selector_tokens)}"
        paths = _paths(
            name=name,
            selection_root=selection_root,
            sidecar_root=sidecar_root,
            subset_root=subset_root,
            config_root=config_root,
            work_root=work_root,
            max_iters=args.planner_max_iters,
        )
        selector_selection = _ensure_aggregate_selection(
            output_path=paths["selector_selection"],
            run_id=f"{name}_selector",
            tokens=selector_tokens,
            store=store,
            split=args.split,
            source=args.loss_episodes,
            cutoff=cutoff,
        )
        random_exclusions = set() if args.allow_random_overlap else set(selector_tokens)
        random_selection = _ensure_random_selection(
            store=store,
            output_path=paths["random_selection"],
            run_id=f"{name}_random",
            selection_count=len(selector_tokens),
            split=args.split,
            seed=args.random_seed + cutoff,
            exclude_tokens=random_exclusions,
            feature_dir=args.features,
        )
        _ensure_subset_and_config(
            source_info=args.source_info,
            selector_selection=paths["selector_selection"],
            random_selection=paths["random_selection"],
            selector_info=paths["selector_info"],
            random_info=paths["random_info"],
            selector_sidecar=paths["selector_sidecar"],
            random_sidecar=paths["random_sidecar"],
            base_config=args.base_config,
            selector_config=paths["selector_config"],
            random_config=paths["random_config"],
            diffusiondrive_root=Path(args.diffusiondrive_root),
            planner_init_checkpoint=args.planner_init_checkpoint,
            planner_lr=args.planner_lr,
            planner_max_iters=args.planner_max_iters,
        )
        selector_metrics = normalize_eval_metrics(
            _run_diffusiondrive_one(
                config=paths["selector_config"],
                train_work_dir=paths["selector_work_dir"],
                eval_work_dir=paths["selector_eval_work_dir"],
                diffusiondrive_root=Path(args.diffusiondrive_root),
                train_seed=args.train_seed,
                eval_seed=args.eval_seed,
                diffusiondrive_python=args.diffusiondrive_python,
                wrapper_python=args.wrapper_python,
            )
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
        episode_id = f"{args.run_id}_r{cutoff:02d}_{len(selector_tokens)}"
        record = build_episode_record(
            episode_id=episode_id,
            selector_selection_path=paths["selector_selection"],
            random_selection_path=paths["random_selection"],
            selector_metrics=selector_metrics,
            random_metrics=random_metrics,
            rho_box=args.rho_box,
            rho_obj=args.rho_obj,
            train_work_dirs={
                "selector": str(paths["selector_work_dir"]),
                "random_same_budget": str(paths["random_work_dir"]),
            },
            eval_work_dirs={
                "selector": str(paths["selector_eval_work_dir"]),
                "random_same_budget": str(paths["random_eval_work_dir"]),
            },
            notes=(
                f"Cumulative aggregate real evaluation through loss-duel round {cutoff}; "
                f"{len(selector_tokens)} selector tokens vs same-size random baseline."
            ),
        )
        _append_or_replace_episode(episode_jsonl, record)
        selector_random_overlap = sorted(
            set(selector_selection["selection"]["selected_tokens"])
            & set(random_selection["selection"]["selected_tokens"])
        )
        completed.append(
            {
                "cutoff_round": cutoff,
                "selected_count": len(selector_tokens),
                "selector_selection": str(paths["selector_selection"]),
                "random_selection": str(paths["random_selection"]),
                "selector_hash": selector_selection["selection"]["selection_hash"],
                "random_hash": random_selection["selection"]["selection_hash"],
                "selector_info": str(paths["selector_info"]),
                "random_info": str(paths["random_info"]),
                "selector_config": str(paths["selector_config"]),
                "random_config": str(paths["random_config"]),
                "selector_work_dir": str(paths["selector_work_dir"]),
                "random_work_dir": str(paths["random_work_dir"]),
                "selector_eval_work_dir": str(paths["selector_eval_work_dir"]),
                "random_eval_work_dir": str(paths["random_eval_work_dir"]),
                "selector_metrics": selector_metrics,
                "random_metrics": random_metrics,
                "selector_cost": reward_cost(selector_metrics, args.rho_box, args.rho_obj),
                "random_cost": reward_cost(random_metrics, args.rho_box, args.rho_obj),
                "reward": float(record["reward"]),
                "selector_random_overlap_count": len(selector_random_overlap),
                "selector_random_overlap_tokens": selector_random_overlap,
            }
        )

    report = {
        "schema": "selector_bench.stage2_aggregate_selected_real_eval.v0",
        "run_id": args.run_id,
        "episode_jsonl": str(episode_jsonl),
        "config": {
            "loss_episodes": str(args.loss_episodes),
            "round_cutoffs": cutoffs,
            "planner_init_checkpoint": str(args.planner_init_checkpoint),
            "planner_lr": args.planner_lr,
            "planner_max_iters": args.planner_max_iters,
            "split": args.split,
            "random_seed": args.random_seed,
            "train_seed": args.train_seed,
            "eval_seed": args.eval_seed,
            "rho_box": args.rho_box,
            "rho_obj": args.rho_obj,
            "allow_random_overlap": bool(args.allow_random_overlap),
        },
        "results": completed,
        "summary": _summary(completed),
    }
    write_json(output_root / f"{args.run_id}_report.json", report)
    return report


def _loss_episode_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    rows = sorted(rows, key=lambda row: _round_from_episode_id(str(row.get("episode_id", ""))))
    if not rows:
        raise ValueError(f"No loss-duel episodes found in {path}")
    return rows


def _round_from_episode_id(episode_id: str) -> int:
    match = re.search(r"_r(\d+)$", episode_id)
    if not match:
        match = re.search(r"_r(\d+)(?:_|$)", episode_id)
    if not match:
        raise ValueError(f"Cannot parse round from episode_id: {episode_id}")
    return int(match.group(1))


def _parse_cutoffs(text: str, max_round: int) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("round-cutoffs cannot be empty.")
    for value in values:
        if value < 1 or value > max_round:
            raise ValueError(f"Invalid cutoff {value}; expected 1..{max_round}.")
    return sorted(dict.fromkeys(values))


def _cumulative_selector_tokens(rows: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for row in rows:
        selection_path = row["selector_selection"]["path"]
        selection = read_json(selection_path)["selection"]["selected_tokens"]
        for token in selection:
            token = str(token)
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def _ensure_aggregate_selection(
    output_path: Path,
    run_id: str,
    tokens: list[str],
    store: FeatureStore,
    split: str,
    source: str | Path,
    cutoff: int,
) -> dict[str, Any]:
    if output_path.exists():
        return read_json(output_path)
    missing = [token for token in tokens if token not in set(store.manifest.tokens(split=split))]
    if missing:
        raise ValueError(f"{len(missing)} aggregate tokens are not in split {split}: {missing[:5]}")
    domain_counts: dict[int, int] = {}
    for token in tokens:
        domain_id = int(store.manifest.get(token).domain_id)
        domain_counts[domain_id] = domain_counts.get(domain_id, 0) + 1
    selection = {
        "baseline": "stage2_loss_duel_cumulative_selector",
        "score_source": "selector_bench.stage2_loss_duel_inprocess",
        "budget_ratio": float(len(tokens)),
        "budget_mode": "exact_count",
        "budget_count": len(tokens),
        "split": split,
        "selected_count": len(tokens),
        "selected_tokens": tokens,
        "selection_hash": hash_jsonable(tokens),
        "domain_budget": {str(key): value for key, value in sorted(domain_counts.items())},
        "domain_counts": {str(key): value for key, value in sorted(domain_counts.items())},
        "exclusions": {"excluded_token_count": 0},
        "score_summary": {},
        "aggregate_source": {
            "loss_episode_jsonl": str(source),
            "cutoff_round": cutoff,
            "source_round_count": cutoff,
            "source_selection_count_per_round": len(tokens) // max(cutoff, 1),
        },
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="selection",
            run_id=run_id,
            config={
                "baseline": "stage2_loss_duel_cumulative_selector",
                "cutoff_round": cutoff,
                "selected_count": len(tokens),
            },
            input_paths=[source],
            summary={
                "selected_count": len(tokens),
                "selection_hash": selection["selection_hash"],
            },
        ),
        "selection": selection,
    }
    write_json(output_path, artifact)
    return artifact


def _paths(
    name: str,
    selection_root: Path,
    sidecar_root: Path,
    subset_root: Path,
    config_root: Path,
    work_root: Path,
    max_iters: int,
) -> dict[str, Path]:
    selector_name = f"{name}_selector"
    random_name = f"{name}_random"
    return {
        "selector_selection": selection_root / f"{selector_name}.json",
        "random_selection": selection_root / f"{random_name}.json",
        "selector_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{selector_name}.pkl",
        "random_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{random_name}.pkl",
        "selector_sidecar": sidecar_root / f"{selector_name}_subset.json",
        "random_sidecar": sidecar_root / f"{random_name}_subset.json",
        "selector_config": config_root / f"dd_{selector_name}.py",
        "random_config": config_root / f"dd_{random_name}.py",
        "selector_work_dir": work_root / f"{selector_name}_subset_{max_iters}iter",
        "random_work_dir": work_root / f"{random_name}_subset_{max_iters}iter",
        "selector_eval_work_dir": work_root / f"{selector_name}_subset_{max_iters}iter_eval_standalone",
        "random_eval_work_dir": work_root / f"{random_name}_subset_{max_iters}iter_eval_standalone",
    }


def _ensure_subset_and_config(
    source_info: str | Path,
    selector_selection: Path,
    random_selection: Path,
    selector_info: Path,
    random_info: Path,
    selector_sidecar: Path,
    random_sidecar: Path,
    base_config: str | Path,
    selector_config: Path,
    random_config: Path,
    diffusiondrive_root: Path,
    planner_init_checkpoint: str | Path | None,
    planner_lr: float | None,
    planner_max_iters: int | None,
) -> None:
    if not selector_info.exists():
        export_diffusiondrive_info_subset(source_info, selector_selection, selector_info, selector_sidecar)
    if not random_info.exists():
        export_diffusiondrive_info_subset(source_info, random_selection, random_info, random_sidecar)
    if not selector_config.exists():
        write_subset_train_config(
            base_config,
            selector_config,
            selector_info,
            diffusiondrive_root,
            planner_init_checkpoint=planner_init_checkpoint,
            planner_lr=planner_lr,
            planner_max_iters=planner_max_iters,
        )
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


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"status": "empty", "completed_cutoffs": 0}
    best = min(results, key=lambda item: float(item["selector_cost"]))
    return {
        "status": "complete",
        "completed_cutoffs": len(results),
        "cutoffs": [int(item["cutoff_round"]) for item in results],
        "best_selector_cost_cutoff": int(best["cutoff_round"]),
        "last_reward": float(results[-1]["reward"]),
        "last_selector_l2": results[-1]["selector_metrics"].get("l2"),
        "last_random_l2": results[-1]["random_metrics"].get("l2"),
    }


if __name__ == "__main__":
    raise SystemExit(main())

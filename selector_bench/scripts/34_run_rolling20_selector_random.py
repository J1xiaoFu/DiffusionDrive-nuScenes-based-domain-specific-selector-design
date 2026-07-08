#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from selector_bench.core.baseline_adapter import DomainStratifiedAdapter
from selector_bench.core.budget_allocator import budget_count
from selector_bench.core.feature_store import FeatureStore
from selector_bench.core.score_provider import RandomScoreProvider
from selector_bench.core.selection_runner import run_selection
from selector_bench.training.stage2_high_frequency import (
    export_diffusiondrive_info_subset,
    normalize_eval_metrics,
    parse_eval_metrics,
    write_subset_train_config,
)
from selector_bench.training.stage2_high_frequency import _run_diffusiondrive_one
from selector_bench.training.stage2_neural import (
    append_episode_record,
    build_episode_record,
    reward_cost,
    sample_gumbel_topk_selection,
    train_neural_selector_stage2_feedback,
)
from selector_bench.utils.io import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three rolling 20% selector-vs-random comparisons. Random is "
            "sampled from the same current candidate pool as selector; only "
            "selector tokens are removed before the next round."
        )
    )
    parser.add_argument(
        "--features",
        default=str(ROOT / "artifacts" / "features" / "trainval_partial_teacher_100iter_train_20260706"),
    )
    parser.add_argument(
        "--init-selector",
        default=str(
            ROOT
            / "artifacts"
            / "selectors"
            / "stage2_feedback_partial_trainval_neural_v7_tiny_rho_box100_normfix_20260706.json"
        ),
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
    parser.add_argument(
        "--diffusiondrive-root",
        default=str(REPO_ROOT / "DiffusionDrive"),
    )
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
        default=str(REPO_ROOT / "DiffusionDrive" / "ckpt" / "diffusiondrive_nusc_stage2.pth"),
    )
    parser.add_argument("--planner-lr", type=float, default=1e-6)
    parser.add_argument("--planner-max-iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument(
        "--selection-count",
        type=int,
        default=None,
        help="Exact count per round. Defaults to ceil(budget * initial train count).",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--sample-seed", type=int, default=4100)
    parser.add_argument("--random-seed", type=int, default=5100)
    parser.add_argument("--selector-seed", type=int, default=6100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--selector-epochs", type=int, default=40)
    parser.add_argument("--selector-lr", type=float, default=0.01)
    parser.add_argument("--selector-l2", type=float, default=1e-4)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--target-clip", type=float, default=2.0)
    parser.add_argument("--min-abs-reward", type=float, default=0.0)
    parser.add_argument("--negative-token-penalty", type=float, default=0.0)
    parser.add_argument("--negative-random-bonus", type=float, default=0.0)
    parser.add_argument("--negative-reward-max", type=float, default=-1e-12)
    parser.add_argument("--rho-box", type=float, default=100.0)
    parser.add_argument("--rho-obj", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "rolling20_selector_random"),
    )
    parser.add_argument("--run-id", default="rolling20_selector_random_lr1e6_20260707")
    parser.add_argument(
        "--subset-info-dir",
        default=str(REPO_ROOT / "DiffusionDrive" / "data" / "infos" / "trainval_partial_selector_subsets"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "artifacts" / "diffusiondrive_configs"),
    )
    parser.add_argument("--work-dir-root", default=str(ROOT / "work_dirs"))
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--episode-jsonl", default=None)
    parser.add_argument("--diffusiondrive-python", default=None)
    parser.add_argument("--wrapper-python", default=None)
    parser.add_argument("--skip-fulltrain", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_rolling20_selector_random(args)
    report_path = Path(args.output_dir) / f"{args.run_id}_report.json"
    print(f"Wrote rolling selector/random report: {report_path}")
    print(report["summary"])
    return 0


def run_rolling20_selector_random(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_dir(args.output_dir)
    selection_root = ensure_dir(output_root / "selections")
    selector_root = ensure_dir(output_root / "selectors")
    sidecar_root = ensure_dir(output_root / "subset_metadata")
    subset_root = ensure_dir(args.subset_info_dir)
    config_root = ensure_dir(args.config_dir)
    work_root = ensure_dir(args.work_dir_root)
    dd_root = Path(args.diffusiondrive_root)
    state_path = Path(args.state_json) if args.state_json else output_root / f"{args.run_id}_state.json"
    episode_jsonl = Path(args.episode_jsonl) if args.episode_jsonl else output_root / f"{args.run_id}_episodes.jsonl"

    store = FeatureStore.from_feature_dir(args.features, cache=True)
    train_records = list(store.manifest.filter(split=args.split))
    initial_count = len(train_records)
    if initial_count < 1:
        raise ValueError(f"No records found for split {args.split!r}.")
    selection_count = args.selection_count or budget_count(initial_count, args.budget)
    if selection_count < 1:
        raise ValueError("selection_count must be positive.")
    if args.rounds * selection_count > initial_count:
        raise ValueError(
            f"rounds * selection_count exceeds candidate count: "
            f"{args.rounds} * {selection_count} > {initial_count}"
        )

    state = _load_or_init_state(
        state_path=state_path,
        run_id=args.run_id,
        init_selector_path=Path(args.init_selector),
        restart=args.restart,
    )
    selector_used_tokens = set(state["selector_used_tokens"])
    current_selector = Path(state["current_selector"])
    completed_rounds = int(state["completed_rounds"])
    round_records = list(state.get("rounds", []))

    _validate_paths(args)
    _write_state(
        state_path=state_path,
        run_id=args.run_id,
        current_selector=current_selector,
        selector_used_tokens=selector_used_tokens,
        completed_rounds=completed_rounds,
        target_rounds=args.rounds,
        selection_count=selection_count,
        rounds=round_records,
        status="running" if completed_rounds < args.rounds else "complete",
    )

    for round_idx in range(completed_rounds + 1, args.rounds + 1):
        paths = _round_paths(
            run_id=args.run_id,
            round_idx=round_idx,
            selection_count=selection_count,
            selection_root=selection_root,
            selector_root=selector_root,
            subset_root=subset_root,
            sidecar_root=sidecar_root,
            config_root=config_root,
            work_root=work_root,
        )
        candidate_exclusions = set(selector_used_tokens)
        selector_selection = _ensure_selector_selection(
            feature_dir=args.features,
            selector_checkpoint=current_selector,
            output_path=paths["selector_selection"],
            run_id=f"{args.run_id}_r{round_idx:02d}_selector{selection_count}",
            selection_count=selection_count,
            split=args.split,
            temperature=args.temperature,
            seed=args.sample_seed + round_idx - 1,
            exclude_tokens=candidate_exclusions,
        )
        random_selection = _ensure_random_selection(
            store=store,
            feature_dir=args.features,
            output_path=paths["random_selection"],
            run_id=f"{args.run_id}_r{round_idx:02d}_random{selection_count}",
            selection_count=selection_count,
            split=args.split,
            seed=args.random_seed + round_idx - 1,
            exclude_tokens=candidate_exclusions,
        )

        selector_tokens = list(selector_selection["selection"]["selected_tokens"])
        random_tokens = list(random_selection["selection"]["selected_tokens"])
        overlap = sorted(set(selector_tokens) & set(random_tokens))

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
            diffusiondrive_root=dd_root,
            planner_init_checkpoint=args.planner_init_checkpoint,
            planner_lr=args.planner_lr,
            planner_max_iters=args.planner_max_iters,
        )

        selector_metrics = _run_diffusiondrive_one(
            config=paths["selector_config"],
            train_work_dir=paths["selector_work_dir"],
            eval_work_dir=paths["selector_eval_work_dir"],
            diffusiondrive_root=dd_root,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
            diffusiondrive_python=args.diffusiondrive_python,
            wrapper_python=args.wrapper_python,
        )
        random_metrics = _run_diffusiondrive_one(
            config=paths["random_config"],
            train_work_dir=paths["random_work_dir"],
            eval_work_dir=paths["random_eval_work_dir"],
            diffusiondrive_root=dd_root,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
            diffusiondrive_python=args.diffusiondrive_python,
            wrapper_python=args.wrapper_python,
        )

        episode_id = f"{args.run_id}_r{round_idx:02d}"
        record = build_episode_record(
            episode_id=episode_id,
            selector_selection_path=paths["selector_selection"],
            random_selection_path=paths["random_selection"],
            selector_metrics=normalize_eval_metrics(selector_metrics),
            random_metrics=normalize_eval_metrics(random_metrics),
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
                "Rolling 20% selector-vs-random round. Both arms sample from "
                "the same selector-remaining candidate pool; only selector "
                "tokens are removed before the next round."
            ),
        )
        if not _episode_exists(episode_jsonl, episode_id):
            append_episode_record(episode_jsonl, record)

        if not paths["updated_selector"].exists():
            train_neural_selector_stage2_feedback(
                feature_dir=args.features,
                init_selector_path=current_selector,
                episode_jsonl=episode_jsonl,
                output_path=paths["updated_selector"],
                run_id=f"{args.run_id}_r{round_idx:02d}_selector_update",
                epochs=args.selector_epochs,
                lr=args.selector_lr,
                l2=args.selector_l2,
                reward_scale=args.reward_scale,
                target_clip=args.target_clip,
                min_abs_reward=args.min_abs_reward,
                negative_token_penalty=args.negative_token_penalty,
                negative_random_bonus=args.negative_random_bonus,
                negative_reward_max=args.negative_reward_max,
                seed=args.selector_seed + round_idx - 1,
            )

        current_selector = paths["updated_selector"]
        selector_used_tokens.update(selector_tokens)
        completed_rounds = round_idx
        round_records = [
            item for item in round_records if int(item.get("round", -1)) != round_idx
        ]
        round_records.append(
            {
                "round": round_idx,
                "status": "updated",
                "candidate_pool_count": initial_count - (completed_rounds - 1) * selection_count,
                "selector_selection": str(paths["selector_selection"]),
                "random_selection": str(paths["random_selection"]),
                "selector_info": str(paths["selector_info"]),
                "random_info": str(paths["random_info"]),
                "selector_config": str(paths["selector_config"]),
                "random_config": str(paths["random_config"]),
                "selector_work_dir": str(paths["selector_work_dir"]),
                "random_work_dir": str(paths["random_work_dir"]),
                "selector_eval_work_dir": str(paths["selector_eval_work_dir"]),
                "random_eval_work_dir": str(paths["random_eval_work_dir"]),
                "updated_selector": str(current_selector),
                "selector_hash": selector_selection["selection"]["selection_hash"],
                "random_hash": random_selection["selection"]["selection_hash"],
                "selector_count": len(selector_tokens),
                "random_count": len(random_tokens),
                "same_round_overlap_count": len(overlap),
                "same_round_overlap_tokens": overlap,
                "selector_metrics": selector_metrics,
                "random_metrics": random_metrics,
                "selector_cost": reward_cost(selector_metrics, rho_box=args.rho_box, rho_obj=args.rho_obj),
                "random_cost": reward_cost(random_metrics, rho_box=args.rho_box, rho_obj=args.rho_obj),
                "reward": float(record["reward"]),
            }
        )
        _write_state(
            state_path=state_path,
            run_id=args.run_id,
            current_selector=current_selector,
            selector_used_tokens=selector_used_tokens,
            completed_rounds=completed_rounds,
            target_rounds=args.rounds,
            selection_count=selection_count,
            rounds=round_records,
            status="complete" if completed_rounds >= args.rounds else "running",
        )

    fulltrain_record = None
    if not args.skip_fulltrain:
        fulltrain_record = _run_fulltrain_baseline(
            args=args,
            run_id=args.run_id,
            config_root=config_root,
            work_root=work_root,
            diffusiondrive_root=dd_root,
        )

    report = _build_report(
        args=args,
        state_path=state_path,
        episode_jsonl=episode_jsonl,
        current_selector=current_selector,
        initial_count=initial_count,
        selection_count=selection_count,
        selector_used_tokens=selector_used_tokens,
        round_records=round_records,
        fulltrain_record=fulltrain_record,
    )
    write_json(output_root / f"{args.run_id}_report.json", report)
    return report


def _validate_paths(args: argparse.Namespace) -> None:
    for name in [
        "features",
        "init_selector",
        "source_info",
        "diffusiondrive_root",
        "base_config",
        "planner_init_checkpoint",
    ]:
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")


def _load_or_init_state(
    state_path: Path,
    run_id: str,
    init_selector_path: Path,
    restart: bool,
) -> dict[str, Any]:
    if state_path.exists() and not restart:
        return read_json(state_path)
    return {
        "schema": "selector_bench.rolling20_selector_random_state.v0",
        "run_id": run_id,
        "status": "running",
        "current_selector": str(init_selector_path),
        "completed_rounds": 0,
        "selector_used_tokens": [],
        "rounds": [],
    }


def _write_state(
    state_path: Path,
    run_id: str,
    current_selector: Path,
    selector_used_tokens: set[str],
    completed_rounds: int,
    target_rounds: int,
    selection_count: int,
    rounds: list[dict[str, Any]],
    status: str,
) -> None:
    write_json(
        state_path,
        {
            "schema": "selector_bench.rolling20_selector_random_state.v0",
            "run_id": run_id,
            "status": status,
            "current_selector": str(current_selector),
            "completed_rounds": completed_rounds,
            "target_rounds": target_rounds,
            "selection_count": selection_count,
            "selector_used_tokens": sorted(selector_used_tokens),
            "rounds": rounds,
        },
    )


def _round_paths(
    run_id: str,
    round_idx: int,
    selection_count: int,
    selection_root: Path,
    selector_root: Path,
    subset_root: Path,
    sidecar_root: Path,
    config_root: Path,
    work_root: Path,
) -> dict[str, Path]:
    stem = f"{run_id}_r{round_idx:02d}"
    selector_name = f"{stem}_selector{selection_count}"
    random_name = f"{stem}_random{selection_count}"
    return {
        "selector_selection": selection_root / f"{selector_name}.json",
        "random_selection": selection_root / f"{random_name}.json",
        "selector_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{selector_name}.pkl",
        "random_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{random_name}.pkl",
        "selector_sidecar": sidecar_root / f"{selector_name}_subset.json",
        "random_sidecar": sidecar_root / f"{random_name}_subset.json",
        "selector_config": config_root / f"dd_{selector_name}.py",
        "random_config": config_root / f"dd_{random_name}.py",
        "selector_work_dir": work_root / f"{selector_name}_subset_100iter",
        "random_work_dir": work_root / f"{random_name}_subset_100iter",
        "selector_eval_work_dir": work_root / f"{selector_name}_subset_100iter_eval_standalone",
        "random_eval_work_dir": work_root / f"{random_name}_subset_100iter_eval_standalone",
        "updated_selector": selector_root / f"{stem}_updated_selector.json",
    }


def _ensure_selector_selection(
    feature_dir: str | Path,
    selector_checkpoint: str | Path,
    output_path: Path,
    run_id: str,
    selection_count: int,
    split: str,
    temperature: float,
    seed: int,
    exclude_tokens: set[str],
) -> dict[str, Any]:
    if output_path.exists():
        return read_json(output_path)
    return sample_gumbel_topk_selection(
        feature_dir=feature_dir,
        selector_checkpoint=selector_checkpoint,
        output_path=output_path,
        run_id=run_id,
        budget=float(selection_count),
        split=split,
        temperature=temperature,
        seed=seed,
        exclude_tokens=exclude_tokens,
    )


def _ensure_random_selection(
    store: FeatureStore,
    feature_dir: str | Path,
    output_path: Path,
    run_id: str,
    selection_count: int,
    split: str,
    seed: int,
    exclude_tokens: set[str],
) -> dict[str, Any]:
    if output_path.exists():
        return read_json(output_path)
    return run_selection(
        manifest=store.manifest,
        adapter=DomainStratifiedAdapter(),
        score_provider=RandomScoreProvider(seed=seed),
        budget=float(selection_count),
        output_path=output_path,
        run_id=run_id,
        split=split,
        config={
            "features": str(feature_dir),
            "baseline": "random",
            "score": "random",
            "budget": float(selection_count),
            "budget_count": selection_count,
            "seed": seed,
            "candidate_pool": "selector_remaining_tokens",
            "excluded_token_count": len(exclude_tokens),
        },
        input_paths=[Path(feature_dir) / "manifest.jsonl"],
        exclude_tokens=exclude_tokens,
    )


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
    planner_init_checkpoint: str | Path,
    planner_lr: float,
    planner_max_iters: int,
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


def _episode_exists(path: Path, episode_id: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if f'"episode_id": "{episode_id}"' in line:
            return True
    return False


def _run_fulltrain_baseline(
    args: argparse.Namespace,
    run_id: str,
    config_root: Path,
    work_root: Path,
    diffusiondrive_root: Path,
) -> dict[str, Any]:
    full_name = f"{run_id}_fulltrain_lr{_format_lr(args.planner_lr)}"
    config = config_root / f"dd_{full_name}.py"
    work_dir = work_root / f"{full_name}_100iter"
    eval_work_dir = work_root / f"{full_name}_100iter_eval_standalone"
    if not config.exists():
        write_subset_train_config(
            args.base_config,
            config,
            args.source_info,
            diffusiondrive_root,
            planner_init_checkpoint=args.planner_init_checkpoint,
            planner_lr=args.planner_lr,
            planner_max_iters=args.planner_max_iters,
        )
    metrics = _run_diffusiondrive_one(
        config=config,
        train_work_dir=work_dir,
        eval_work_dir=eval_work_dir,
        diffusiondrive_root=diffusiondrive_root,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        diffusiondrive_python=args.diffusiondrive_python,
        wrapper_python=args.wrapper_python,
    )
    metrics = normalize_eval_metrics(metrics)
    return {
        "status": "evaluated",
        "config": str(config),
        "train_work_dir": str(work_dir),
        "eval_work_dir": str(eval_work_dir),
        "metrics": metrics,
        "cost": reward_cost(metrics, rho_box=args.rho_box, rho_obj=args.rho_obj),
    }


def _build_report(
    args: argparse.Namespace,
    state_path: Path,
    episode_jsonl: Path,
    current_selector: Path,
    initial_count: int,
    selection_count: int,
    selector_used_tokens: set[str],
    round_records: list[dict[str, Any]],
    fulltrain_record: dict[str, Any] | None,
) -> dict[str, Any]:
    random_unique_tokens: set[str] = set()
    for record in round_records:
        random_path = Path(record["random_selection"])
        if random_path.exists():
            random_unique_tokens.update(read_json(random_path)["selection"]["selected_tokens"])

    report = {
        "schema": "selector_bench.rolling20_selector_random_report.v0",
        "run_id": args.run_id,
        "state_json": str(state_path),
        "episode_jsonl": str(episode_jsonl),
        "current_selector": str(current_selector),
        "config": {
            "features": str(args.features),
            "init_selector": str(args.init_selector),
            "source_info": str(args.source_info),
            "base_config": str(args.base_config),
            "planner_init_checkpoint": str(args.planner_init_checkpoint),
            "planner_lr": args.planner_lr,
            "planner_max_iters": args.planner_max_iters,
            "rounds": args.rounds,
            "budget": args.budget,
            "selection_count": selection_count,
            "split": args.split,
            "temperature": args.temperature,
            "sample_seed": args.sample_seed,
            "random_seed": args.random_seed,
            "selector_seed": args.selector_seed,
            "rho_box": args.rho_box,
            "rho_obj": args.rho_obj,
            "candidate_pool_policy": (
                "Selector and random sample from the same current pool; only "
                "selector selections are removed before the next round."
            ),
        },
        "summary": {
            "status": "complete",
            "initial_candidate_count": initial_count,
            "selection_count_per_round": selection_count,
            "rounds": len(round_records),
            "selector_unique_count": len(selector_used_tokens),
            "selector_unique_ratio": len(selector_used_tokens) / initial_count,
            "random_unique_count": len(random_unique_tokens),
            "random_unique_ratio": len(random_unique_tokens) / initial_count,
        },
        "rounds": sorted(round_records, key=lambda item: int(item["round"])),
        "fulltrain_baseline": fulltrain_record,
    }
    return report


def _format_lr(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


if __name__ == "__main__":
    raise SystemExit(main())

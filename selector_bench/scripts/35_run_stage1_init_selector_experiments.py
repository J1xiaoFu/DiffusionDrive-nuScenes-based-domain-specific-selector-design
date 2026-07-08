#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from selector_bench.core.budget_allocator import budget_count
from selector_bench.core.feature_store import FeatureStore
from selector_bench.training.stage2_high_frequency import (
    _checkpoint_from_status,
    _run_diffusiondrive_one,
    export_diffusiondrive_info_subset,
    normalize_eval_metrics,
    parse_eval_metrics,
    write_subset_train_config,
)
from selector_bench.training.stage2_neural import reward_cost, sample_gumbel_topk_selection
from selector_bench.utils.io import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SparseDrive stage-1 initialized DiffusionDrive experiments: "
            "direct eval, full-partial LR sweep, and three rolling 20% selector "
            "fine-tuning stages."
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
        "--base-config",
        default=str(
            ROOT
            / "artifacts"
            / "diffusiondrive_configs"
            / "diffusiondrive_small_stage2_trainval_partial_1600_400_real_anchor_100iter.py"
        ),
    )
    parser.add_argument("--diffusiondrive-root", default=str(REPO_ROOT / "DiffusionDrive"))
    parser.add_argument(
        "--stage1-checkpoint",
        default=str(REPO_ROOT / "DiffusionDrive" / "ckpts" / "sparsedrive_stage1.pth"),
    )
    parser.add_argument("--lr-sweep", default="3e-4,1e-4,3e-5,1e-5")
    parser.add_argument("--planner-max-iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--selection-count", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--sample-seed", type=int, default=7300)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--rho-box", type=float, default=100.0)
    parser.add_argument("--rho-obj", type=float, default=0.05)
    parser.add_argument("--run-id", default="stage1_init_partial100_selector3_20260707")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "stage1_init_selector_experiments"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "artifacts" / "diffusiondrive_configs"),
    )
    parser.add_argument(
        "--subset-info-dir",
        default=str(REPO_ROOT / "DiffusionDrive" / "data" / "infos" / "trainval_partial_selector_subsets"),
    )
    parser.add_argument("--work-dir-root", default=str(ROOT / "work_dirs"))
    parser.add_argument("--diffusiondrive-python", default=None)
    parser.add_argument("--wrapper-python", default=None)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    report_path = Path(args.output_dir) / f"{args.run_id}_report.json"
    print(f"Wrote report: {report_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_inputs(args)
    output_root = ensure_dir(args.output_dir)
    config_root = ensure_dir(args.config_dir)
    work_root = ensure_dir(args.work_dir_root)
    subset_root = ensure_dir(args.subset_info_dir)
    selection_root = ensure_dir(output_root / "selections")
    sidecar_root = ensure_dir(output_root / "subset_metadata")
    dd_root = Path(args.diffusiondrive_root)
    report_path = output_root / f"{args.run_id}_report.json"

    store = FeatureStore.from_feature_dir(args.features, cache=True)
    train_records = list(store.manifest.filter(split=args.split))
    selection_count = args.selection_count or budget_count(len(train_records), args.budget)
    lrs = _parse_lrs(args.lr_sweep)

    report: dict[str, Any] = (
        read_json(report_path)
        if report_path.exists() and not args.restart
        else {
            "schema": "selector_bench.stage1_init_selector_experiments.v0",
            "run_id": args.run_id,
            "config": {
                "features": str(args.features),
                "init_selector": str(args.init_selector),
                "source_info": str(args.source_info),
                "base_config": str(args.base_config),
                "stage1_checkpoint": str(args.stage1_checkpoint),
                "lr_sweep": lrs,
                "planner_max_iters": args.planner_max_iters,
                "rounds": args.rounds,
                "budget": args.budget,
                "selection_count": selection_count,
                "rho_box": args.rho_box,
                "rho_obj": args.rho_obj,
            },
            "stage1_eval": None,
            "full_lr_sweep": [],
            "selected_lr": None,
            "selector_stages": [],
            "summary": {},
        }
    )

    if report.get("stage1_eval") is None:
        report["stage1_eval"] = _run_stage1_eval(args, work_root)
        _persist(report_path, report)

    completed_lrs = {float(item["lr"]) for item in report.get("full_lr_sweep", [])}
    for lr in lrs:
        if lr in completed_lrs:
            continue
        record = _run_full_lr(args, config_root, work_root, dd_root, lr)
        report["full_lr_sweep"].append(record)
        _persist(report_path, report)

    if report.get("selected_lr") is None:
        best = min(
            report["full_lr_sweep"],
            key=lambda item: (float(item["cost"]), float(item["metrics"]["l2"])),
        )
        report["selected_lr"] = float(best["lr"])
        _persist(report_path, report)

    completed_stages = {int(item["stage"]) for item in report.get("selector_stages", [])}
    used_tokens: set[str] = set()
    previous_checkpoint = Path(args.stage1_checkpoint)
    for item in sorted(report.get("selector_stages", []), key=lambda row: int(row["stage"])):
        selection_path = Path(item["selection"])
        if selection_path.exists():
            used_tokens.update(read_json(selection_path)["selection"]["selected_tokens"])
        status_path = Path(item["train_work_dir"]) / "run_status.json"
        if status_path.exists():
            previous_checkpoint = _checkpoint_from_status(status_path)

    for stage_idx in range(1, args.rounds + 1):
        if stage_idx in completed_stages:
            continue
        stage_record = _run_selector_stage(
            args=args,
            config_root=config_root,
            work_root=work_root,
            subset_root=subset_root,
            selection_root=selection_root,
            sidecar_root=sidecar_root,
            dd_root=dd_root,
            stage_idx=stage_idx,
            selection_count=selection_count,
            lr=float(report["selected_lr"]),
            init_checkpoint=previous_checkpoint,
            used_tokens=used_tokens,
        )
        report["selector_stages"].append(stage_record)
        used_tokens.update(read_json(stage_record["selection"])["selection"]["selected_tokens"])
        previous_checkpoint = _checkpoint_from_status(Path(stage_record["train_work_dir"]) / "run_status.json")
        _persist(report_path, report)

    report["summary"] = _build_summary(report)
    _persist(report_path, report)
    _write_markdown_report(output_root / f"{args.run_id}_report.md", report)
    return report


def _run_stage1_eval(args: argparse.Namespace, work_root: Path) -> dict[str, Any]:
    work_dir = work_root / f"{args.run_id}_stage1_checkpoint_eval"
    status_json = work_dir / "eval_status.json"
    if not _successful_eval_status(status_json):
        wrapper = str(args.wrapper_python or sys.executable)
        cmd = [
            wrapper,
            str(ROOT / "scripts" / "18_eval_diffusiondrive_checkpoint.py"),
            "--diffusiondrive-root",
            str(args.diffusiondrive_root),
            "--config",
            str(args.base_config),
            "--checkpoint",
            str(args.stage1_checkpoint),
            "--work-dir",
            str(work_dir),
            "--seed",
            str(args.eval_seed),
            "--status-json",
            str(status_json),
        ]
        if args.diffusiondrive_python is not None:
            cmd.extend(["--python", str(args.diffusiondrive_python)])
        _run_checked(cmd)
    metrics = parse_eval_metrics(work_dir / "outer_eval.log", status_json)
    metrics = normalize_eval_metrics(metrics)
    metrics["cost"] = reward_cost(metrics, rho_box=args.rho_box, rho_obj=args.rho_obj)
    return {
        "checkpoint": str(args.stage1_checkpoint),
        "eval_work_dir": str(work_dir),
        "metrics": metrics,
        "cost": float(metrics["cost"]),
    }


def _run_full_lr(
    args: argparse.Namespace,
    config_root: Path,
    work_root: Path,
    dd_root: Path,
    lr: float,
) -> dict[str, Any]:
    name = f"{args.run_id}_full_lr{_format_lr(lr)}"
    config = config_root / f"dd_{name}.py"
    work_dir = work_root / f"{name}_{args.planner_max_iters}iter"
    eval_work_dir = work_root / f"{name}_{args.planner_max_iters}iter_eval_standalone"
    if not config.exists():
        write_subset_train_config(
            args.base_config,
            config,
            args.source_info,
            dd_root,
            planner_init_checkpoint=args.stage1_checkpoint,
            planner_lr=lr,
            planner_max_iters=args.planner_max_iters,
        )
    metrics = _run_diffusiondrive_one(
        config=config,
        train_work_dir=work_dir,
        eval_work_dir=eval_work_dir,
        diffusiondrive_root=dd_root,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        diffusiondrive_python=args.diffusiondrive_python,
        wrapper_python=args.wrapper_python,
    )
    metrics = normalize_eval_metrics(metrics)
    cost = reward_cost(metrics, rho_box=args.rho_box, rho_obj=args.rho_obj)
    return {
        "lr": lr,
        "config": str(config),
        "train_work_dir": str(work_dir),
        "eval_work_dir": str(eval_work_dir),
        "metrics": metrics,
        "cost": float(cost),
    }


def _run_selector_stage(
    args: argparse.Namespace,
    config_root: Path,
    work_root: Path,
    subset_root: Path,
    selection_root: Path,
    sidecar_root: Path,
    dd_root: Path,
    stage_idx: int,
    selection_count: int,
    lr: float,
    init_checkpoint: Path,
    used_tokens: set[str],
) -> dict[str, Any]:
    name = f"{args.run_id}_selector_stage{stage_idx:02d}_{selection_count}"
    selection_path = selection_root / f"{name}.json"
    info_path = subset_root / f"nuscenes_partial_1600_400_infos_train_{name}.pkl"
    sidecar_path = sidecar_root / f"{name}_subset.json"
    config = config_root / f"dd_{name}.py"
    work_dir = work_root / f"{name}_{args.planner_max_iters}iter"
    eval_work_dir = work_root / f"{name}_{args.planner_max_iters}iter_eval_standalone"

    if not selection_path.exists():
        sample_gumbel_topk_selection(
            feature_dir=args.features,
            selector_checkpoint=args.init_selector,
            output_path=selection_path,
            run_id=name,
            budget=float(selection_count),
            split=args.split,
            temperature=args.temperature,
            seed=args.sample_seed + stage_idx - 1,
            exclude_tokens=used_tokens,
        )
    if not info_path.exists():
        export_diffusiondrive_info_subset(args.source_info, selection_path, info_path, sidecar_path)
    if not config.exists():
        write_subset_train_config(
            args.base_config,
            config,
            info_path,
            dd_root,
            planner_init_checkpoint=init_checkpoint,
            planner_lr=lr,
            planner_max_iters=args.planner_max_iters,
        )

    metrics = _run_diffusiondrive_one(
        config=config,
        train_work_dir=work_dir,
        eval_work_dir=eval_work_dir,
        diffusiondrive_root=dd_root,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        diffusiondrive_python=args.diffusiondrive_python,
        wrapper_python=args.wrapper_python,
    )
    metrics = normalize_eval_metrics(metrics)
    selection = read_json(selection_path)["selection"]
    cost = reward_cost(metrics, rho_box=args.rho_box, rho_obj=args.rho_obj)
    return {
        "stage": stage_idx,
        "init_checkpoint": str(init_checkpoint),
        "selection": str(selection_path),
        "selection_hash": selection["selection_hash"],
        "selected_count": int(selection["selected_count"]),
        "subset_info": str(info_path),
        "config": str(config),
        "train_work_dir": str(work_dir),
        "eval_work_dir": str(eval_work_dir),
        "metrics": metrics,
        "cost": float(cost),
    }


def _build_summary(report: dict[str, Any]) -> dict[str, Any]:
    full = sorted(report["full_lr_sweep"], key=lambda item: (float(item["cost"]), float(item["metrics"]["l2"])))
    stages = sorted(report["selector_stages"], key=lambda item: int(item["stage"]))
    return {
        "stage1_eval_l2": report["stage1_eval"]["metrics"].get("l2"),
        "stage1_eval_cost": report["stage1_eval"].get("cost"),
        "selected_lr": report.get("selected_lr"),
        "best_full_l2": full[0]["metrics"].get("l2") if full else None,
        "best_full_cost": full[0].get("cost") if full else None,
        "selector_stage_l2": [stage["metrics"].get("l2") for stage in stages],
        "selector_stage_cost": [stage.get("cost") for stage in stages],
        "completed_selector_stages": len(stages),
    }


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {report['run_id']}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2),
        "```",
        "",
        "## Full LR Sweep",
        "",
        "| lr | l2 | obj_box_col | obj_col | cost | train s | eval s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["full_lr_sweep"]:
        metrics = item["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["lr"]),
                    _fmt(metrics.get("l2")),
                    _fmt(metrics.get("obj_box_col")),
                    _fmt(metrics.get("obj_col")),
                    _fmt(item.get("cost")),
                    _fmt(metrics.get("train_runtime_s")),
                    _fmt(metrics.get("eval_runtime_s")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selector Stages",
            "",
            "| stage | selected | l2 | obj_box_col | obj_col | cost | selection hash |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for stage in sorted(report["selector_stages"], key=lambda item: int(item["stage"])):
        metrics = stage["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(stage["stage"]),
                    str(stage["selected_count"]),
                    _fmt(metrics.get("l2")),
                    _fmt(metrics.get("obj_box_col")),
                    _fmt(metrics.get("obj_col")),
                    _fmt(stage.get("cost")),
                    str(stage["selection_hash"])[:12],
                ]
            )
            + " |"
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_inputs(args: argparse.Namespace) -> None:
    for name in [
        "features",
        "init_selector",
        "source_info",
        "base_config",
        "diffusiondrive_root",
        "stage1_checkpoint",
    ]:
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.rounds < 1:
        raise ValueError("rounds must be positive.")
    if args.planner_max_iters < 1:
        raise ValueError("planner-max-iters must be positive.")


def _successful_eval_status(path: Path) -> bool:
    if not path.exists():
        return False
    status = read_json(path)
    return int(status.get("exit_code", 1)) == 0


def _run_checked(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd}")


def _parse_lrs(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("lr-sweep must contain at least one learning rate.")
    return values


def _format_lr(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _persist(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)


if __name__ == "__main__":
    raise SystemExit(main())

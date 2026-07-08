#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from selector_bench.training.inprocess_loss_worker import InProcessDiffusionDriveLossWorker
from selector_bench.training.stage2_high_frequency import run_stage2_high_frequency_update
from selector_bench.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run high-frequency selector/random duels with one persistent "
            "in-process DiffusionDrive train-loss worker."
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
    parser.add_argument("--planner-max-iters", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--selection-count", type=int, default=8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--sample-seed", type=int, default=11100)
    parser.add_argument("--random-seed", type=int, default=12100)
    parser.add_argument("--selector-seed", type=int, default=13100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--selector-epochs", type=int, default=40)
    parser.add_argument("--selector-lr", type=float, default=0.01)
    parser.add_argument("--selector-l2", type=float, default=1e-4)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--target-clip", type=float, default=2.0)
    parser.add_argument("--min-abs-reward", type=float, default=0.0)
    parser.add_argument("--negative-token-penalty", type=float, default=0.0)
    parser.add_argument("--negative-random-bonus", type=float, default=0.0)
    parser.add_argument("--negative-reward-max", type=float, default=-1e-12)
    parser.add_argument("--loss-key", default="loss")
    parser.add_argument(
        "--loss-utility-mode",
        choices=["drop", "relative_drop", "final", "best"],
        default="drop",
    )
    parser.add_argument("--loss-tail-window", type=int, default=3)
    parser.add_argument("--workers-per-gpu", type=int, default=0)
    parser.add_argument("--base-state-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--grad-clip-max-norm", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "stage2_loss_duel_inprocess"),
    )
    parser.add_argument("--run-id", default="stage2_loss_duel_inprocess_stage1_lr3e5_20260707")
    parser.add_argument(
        "--subset-info-dir",
        default=str(REPO_ROOT / "DiffusionDrive" / "data" / "infos" / "trainval_partial_selector_subsets"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(ROOT / "artifacts" / "diffusiondrive_configs"),
    )
    parser.add_argument("--work-dir-root", default=str(ROOT / "work_dirs"))
    parser.add_argument("--inprocess-work-dir", default=None)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--episode-jsonl", default=None)
    parser.add_argument(
        "--remove-random-tokens",
        action="store_true",
        help="Also remove random-arm tokens from later candidates. Default removes selector tokens only.",
    )
    parser.add_argument(
        "--allow-pair-overlap",
        action="store_true",
        help="Allow selector and random arms in the same duel to overlap.",
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_jsonl = (
        Path(args.episode_jsonl)
        if args.episode_jsonl
        else Path(args.output_dir) / f"{args.run_id}_episodes.jsonl"
    )
    worker_box: dict[str, InProcessDiffusionDriveLossWorker] = {}

    def get_worker() -> InProcessDiffusionDriveLossWorker:
        worker = worker_box.get("worker")
        if worker is None:
            worker = InProcessDiffusionDriveLossWorker(
                diffusiondrive_root=args.diffusiondrive_root,
                base_config=args.base_config,
                source_info=args.source_info,
                planner_init_checkpoint=args.planner_init_checkpoint,
                planner_lr=args.planner_lr,
                max_iters=args.planner_max_iters,
                work_dir=args.inprocess_work_dir
                or str(Path(args.output_dir) / "inprocess_work_dirs" / args.run_id),
                seed=args.train_seed,
                samples_per_gpu=1,
                workers_per_gpu=args.workers_per_gpu,
                base_state_device=args.base_state_device,
                grad_clip_max_norm=args.grad_clip_max_norm,
                loss_key=args.loss_key,
                tail_window=args.loss_tail_window,
                utility_mode=args.loss_utility_mode,
            )
            worker_box["worker"] = worker
        return worker

    def metric_provider(round_idx: int, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        worker = get_worker()
        return worker.run_pair(
            selector_selection_path=context["selector_selection"],
            random_selection_path=context["random_selection"],
            round_idx=round_idx,
        )

    try:
        report = run_stage2_high_frequency_update(
            feature_dir=args.features,
            init_selector_path=args.init_selector,
            episode_jsonl=episode_jsonl,
            output_dir=args.output_dir,
            source_info=args.source_info,
            diffusiondrive_root=args.diffusiondrive_root,
            base_config=args.base_config,
            run_id=args.run_id,
            rounds=args.rounds,
            selection_count=args.selection_count,
            split=args.split,
            temperature=args.temperature,
            sample_seed=args.sample_seed,
            random_seed=args.random_seed,
            selector_seed=args.selector_seed,
            train_seed=args.train_seed,
            selector_epochs=args.selector_epochs,
            selector_lr=args.selector_lr,
            selector_l2=args.selector_l2,
            reward_scale=args.reward_scale,
            target_clip=args.target_clip,
            min_abs_reward=args.min_abs_reward,
            negative_token_penalty=args.negative_token_penalty,
            negative_random_bonus=args.negative_random_bonus,
            negative_reward_max=args.negative_reward_max,
            rho_box=0.0,
            rho_obj=0.0,
            subset_info_dir=args.subset_info_dir,
            config_dir=args.config_dir,
            work_dir_root=args.work_dir_root,
            state_json=args.state_json,
            execute_diffusiondrive=False,
            metric_provider=metric_provider,
            planner_init_checkpoint=args.planner_init_checkpoint,
            planner_lr=args.planner_lr,
            planner_max_iters=args.planner_max_iters,
            disjoint_pair=not args.allow_pair_overlap,
            remove_random_tokens=args.remove_random_tokens,
            restart=args.restart,
        )
    finally:
        worker = worker_box.get("worker")
        if worker is not None:
            worker.close()

    report["config"].update(
        {
            "feedback_source": "diffusiondrive_inprocess_train_loss_proxy",
            "loss_key": args.loss_key,
            "loss_utility_mode": args.loss_utility_mode,
            "loss_tail_window": args.loss_tail_window,
            "base_state_device": args.base_state_device,
            "workers_per_gpu": args.workers_per_gpu,
            "planner_reset_policy": "reset_to_base_checkpoint_before_each_arm",
            "preserve_planner_updates_across_duels": False,
        }
    )
    write_json(Path(args.output_dir) / f"{args.run_id}_report.json", report)
    print(f"Wrote in-process loss-duel report: {Path(args.output_dir) / (args.run_id + '_report.json')}")
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

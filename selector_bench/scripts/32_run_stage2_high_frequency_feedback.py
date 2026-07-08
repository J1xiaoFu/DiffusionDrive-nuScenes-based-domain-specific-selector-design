#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage2_high_frequency import run_stage2_high_frequency_update


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or prepare a Stage 2 feedback update: "
            "323 selector samples vs 323 random samples per update by default, "
            "with conservative planner fine-tuning from the official checkpoint."
        )
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--init-selector", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "stage2_high_frequency"))
    parser.add_argument("--run-id", default="stage2_high_frequency")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--selection-count", type=int, default=323)
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
    parser.add_argument("--replay-episodes", default=None)
    parser.add_argument("--replay-penalty", type=float, default=0.0)
    parser.add_argument("--replay-reward-max", type=float, default=-1e-12)
    parser.add_argument("--rho-box", type=float, default=0.05)
    parser.add_argument("--rho-obj", type=float, default=0.05)
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
            / "diffusiondrive_small_stage2_trainval_partial_stage1_cluster_domain_neural_v0_normfix_gumbel20_100iter.py"
        ),
    )
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
    parser.add_argument(
        "--planner-init-checkpoint",
        default=str(REPO_ROOT / "DiffusionDrive" / "ckpt" / "diffusiondrive_nusc_stage2.pth"),
        help="Fixed DiffusionDrive checkpoint used to initialize every selector/random subset train.",
    )
    parser.add_argument("--planner-lr", type=float, default=1e-6, help="Planner fine-tuning learning rate override.")
    parser.add_argument("--planner-max-iters", type=int, default=100, help="Planner fine-tuning iteration budget override.")
    parser.add_argument("--diffusiondrive-python", default=None)
    parser.add_argument("--wrapper-python", default=None)
    parser.add_argument("--execute-diffusiondrive", action="store_true")
    parser.add_argument(
        "--allow-pair-overlap",
        action="store_true",
        help="Allow selector and random selections in the same micro-round to overlap. Default is disjoint.",
    )
    parser.add_argument("--disjoint-pair", action="store_true", help="Deprecated: disjoint pairs are now the default.")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    report = run_stage2_high_frequency_update(
        feature_dir=args.features,
        init_selector_path=args.init_selector,
        episode_jsonl=args.episodes,
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
        eval_seed=args.eval_seed,
        selector_epochs=args.selector_epochs,
        selector_lr=args.selector_lr,
        selector_l2=args.selector_l2,
        reward_scale=args.reward_scale,
        target_clip=args.target_clip,
        min_abs_reward=args.min_abs_reward,
        negative_token_penalty=args.negative_token_penalty,
        negative_random_bonus=args.negative_random_bonus,
        negative_reward_max=args.negative_reward_max,
        replay_episode_jsonl=args.replay_episodes,
        replay_penalty=args.replay_penalty,
        replay_reward_max=args.replay_reward_max,
        rho_box=args.rho_box,
        rho_obj=args.rho_obj,
        subset_info_dir=args.subset_info_dir,
        config_dir=args.config_dir,
        work_dir_root=args.work_dir_root,
        state_json=args.state_json,
        execute_diffusiondrive=args.execute_diffusiondrive,
        diffusiondrive_python=args.diffusiondrive_python,
        wrapper_python=args.wrapper_python,
        planner_init_checkpoint=args.planner_init_checkpoint,
        planner_lr=args.planner_lr,
        planner_max_iters=args.planner_max_iters,
        disjoint_pair=not args.allow_pair_overlap,
        restart=args.restart,
    )
    print(f"Wrote high-frequency Stage 2 report: {Path(args.output_dir) / (args.run_id + '_report.json')}")
    print(report["summary"])
    if report["summary"]["status"] == "prepared_round_needs_feedback":
        print("Prepared the next micro-round. Re-run with --execute-diffusiondrive to train/eval and update automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

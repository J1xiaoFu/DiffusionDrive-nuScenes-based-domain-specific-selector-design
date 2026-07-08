#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.training.stage2_neural import (
    sample_gumbel_topk_selection,
    train_neural_selector_stage1_imitation,
    write_episode_schema,
)


def _load_token_file(path: str | None) -> set[str]:
    if not path:
        return set()
    tokens: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.add(token)
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a MATES-inspired neural selector v0 by Stage 1 imitation and optionally sample a domain-budgeted subset."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--teacher-selector", default=None)
    parser.add_argument("--target-scores", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="stage2_neural_stage1_imitation")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-domain-bias", action="store_true")
    parser.add_argument("--domain-residual-head", action="store_true")
    parser.add_argument("--selection-output", default=None)
    parser.add_argument("--selection-run-id", default="stage2_neural_gumbel_topk")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument(
        "--budget-count",
        type=int,
        default=None,
        help="Exact sample count. Overrides --budget when set.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--exclude-tokens-file",
        default=None,
        help="Optional newline-delimited sample tokens to exclude before budget allocation.",
    )
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--replay-episodes", default=None)
    parser.add_argument("--replay-penalty", type=float, default=0.0)
    parser.add_argument("--replay-reward-max", type=float, default=-1e-12)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--episode-schema-output", default=None)
    args = parser.parse_args()

    if args.teacher_selector is None and args.target_scores is None:
        parser.error("one of --teacher-selector or --target-scores is required")

    artifact = train_neural_selector_stage1_imitation(
        feature_dir=args.features,
        teacher_selector_path=args.teacher_selector,
        output_path=args.output,
        target_score_path=args.target_scores,
        run_id=args.run_id,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        use_domain_bias=not args.no_domain_bias,
        use_domain_residual_head=args.domain_residual_head,
    )
    print(f"Wrote neural selector checkpoint: {args.output}")
    print(artifact["metadata"]["summary"])
    print(artifact["metadata"]["metrics"])

    if args.selection_output:
        selection = sample_gumbel_topk_selection(
            feature_dir=args.features,
            selector_checkpoint=args.output,
            output_path=args.selection_output,
            run_id=args.selection_run_id,
            budget=float(args.budget_count) if args.budget_count is not None else args.budget,
            split=args.split,
            temperature=args.temperature,
            seed=args.sample_seed,
            replay_episode_jsonl=args.replay_episodes,
            replay_penalty=args.replay_penalty,
            replay_reward_max=args.replay_reward_max,
            exclude_tokens=_load_token_file(args.exclude_tokens_file),
        )
        payload = selection["selection"]
        print(f"Wrote neural Gumbel-Top-k selection: {args.selection_output}")
        print(
            {
                "selected_count": payload["selected_count"],
                "selection_hash": payload["selection_hash"][:12],
                "domain_counts": payload["domain_counts"],
            }
        )

    if args.episode_schema_output:
        write_episode_schema(args.episode_schema_output)
        print(f"Wrote Stage 2 episode schema: {args.episode_schema_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

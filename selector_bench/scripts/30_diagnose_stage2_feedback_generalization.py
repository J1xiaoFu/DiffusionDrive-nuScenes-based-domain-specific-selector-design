#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.core.feature_store import FeatureStore
from selector_bench.training.stage2_neural import (
    score_neural_selector,
    train_neural_selector_stage2_feedback,
)
from selector_bench.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _selection_tokens(path: str | Path, manifest_tokens: set[str]) -> list[str]:
    artifact = read_json(path)
    return [
        token
        for token in artifact["selection"]["selected_tokens"]
        if token in manifest_tokens
    ]


def _score_heldout(
    feature_store: FeatureStore,
    checkpoint_path: str | Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    manifest_tokens = set(feature_store.manifest.tokens(split="train"))
    selector_tokens = _selection_tokens(row["selector_selection"]["path"], manifest_tokens)
    random_tokens = _selection_tokens(row["random_same_budget_selection"]["path"], manifest_tokens)
    if not selector_tokens:
        raise ValueError(f"Held-out selector selection has no matching train tokens: {row['episode_id']}")
    if not random_tokens:
        raise ValueError(f"Held-out random selection has no matching train tokens: {row['episode_id']}")

    selector_scores = score_neural_selector(feature_store, checkpoint_path, selector_tokens)
    random_scores = score_neural_selector(feature_store, checkpoint_path, random_tokens)
    selector_mean = float(np.mean(selector_scores))
    random_mean = float(np.mean(random_scores))
    predicted_gap = selector_mean - random_mean
    reward = float(row["reward"])
    reward_sign = _sign(reward)
    gap_sign = _sign(predicted_gap)
    signed_margin = predicted_gap * reward_sign if reward_sign else 0.0
    return {
        "episode_id": str(row["episode_id"]),
        "reward": reward,
        "reward_sign": reward_sign,
        "selector_hash12": str(row["selector_selection"]["selection_hash"])[:12],
        "random_hash12": str(row["random_same_budget_selection"]["selection_hash"])[:12],
        "selector_score_mean": selector_mean,
        "random_score_mean": random_mean,
        "predicted_gap_selector_minus_random": predicted_gap,
        "predicted_gap_sign": gap_sign,
        "signed_margin": signed_margin,
        "preference_correct": bool(reward_sign != 0 and gap_sign == reward_sign),
        "selector_count": len(selector_tokens),
        "random_count": len(random_tokens),
    }


def build_leave_one_out_report(
    feature_dir: str | Path,
    init_selector_path: str | Path,
    episode_jsonl: str | Path,
    work_dir: str | Path,
    run_id: str,
    epochs: int,
    lr: float,
    l2: float,
    reward_scale: float | None,
    target_clip: float,
    min_abs_reward: float,
    seed: int,
    negative_token_penalty: float = 0.0,
    negative_random_bonus: float = 0.0,
    negative_reward_max: float = -1e-12,
) -> dict[str, Any]:
    rows = read_jsonl(episode_jsonl)
    if len(rows) < 3:
        raise ValueError("At least three feedback rows are required for leave-one-out diagnostics.")
    for row in rows:
        if row.get("schema") != "selector_bench.stage2_episode.v0":
            raise ValueError(f"Unexpected episode schema: {row.get('schema')}")

    target_dir = ensure_dir(work_dir)
    feature_store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    fold_reports: list[dict[str, Any]] = []
    for idx, heldout in enumerate(rows):
        fold_dir = ensure_dir(target_dir / f"fold_{idx:02d}")
        train_jsonl = fold_dir / "episodes_train.jsonl"
        checkpoint = fold_dir / "selector.json"
        train_rows = [row for row_idx, row in enumerate(rows) if row_idx != idx]
        write_jsonl(train_jsonl, train_rows)
        artifact = train_neural_selector_stage2_feedback(
            feature_dir=feature_dir,
            init_selector_path=init_selector_path,
            episode_jsonl=train_jsonl,
            output_path=checkpoint,
            run_id=f"{run_id}_loo_{idx:02d}",
            epochs=epochs,
            lr=lr,
            l2=l2,
            reward_scale=reward_scale,
            target_clip=target_clip,
            min_abs_reward=min_abs_reward,
            negative_token_penalty=negative_token_penalty,
            negative_random_bonus=negative_random_bonus,
            negative_reward_max=negative_reward_max,
            seed=seed + idx,
        )
        heldout_scores = _score_heldout(feature_store, checkpoint, heldout)
        fold_reports.append(
            {
                "fold": idx,
                "heldout": heldout_scores,
                "train_episode_count": len(train_rows),
                "train_episode_jsonl": str(train_jsonl),
                "checkpoint": str(checkpoint),
                "train_metrics": artifact["metadata"]["metrics"],
            }
        )

    nonzero = [fold for fold in fold_reports if fold["heldout"]["reward_sign"] != 0]
    correct = [fold for fold in nonzero if fold["heldout"]["preference_correct"]]
    margins = [float(fold["heldout"]["signed_margin"]) for fold in nonzero]
    positive = [fold for fold in nonzero if fold["heldout"]["reward_sign"] > 0]
    negative = [fold for fold in nonzero if fold["heldout"]["reward_sign"] < 0]

    def _accuracy(folds: list[dict[str, Any]]) -> float | None:
        if not folds:
            return None
        return float(sum(1 for fold in folds if fold["heldout"]["preference_correct"]) / len(folds))

    summary = {
        "episode_count": len(rows),
        "fold_count": len(fold_reports),
        "nonzero_reward_fold_count": len(nonzero),
        "preference_accuracy": _accuracy(nonzero),
        "positive_reward_fold_count": len(positive),
        "positive_preference_accuracy": _accuracy(positive),
        "negative_reward_fold_count": len(negative),
        "negative_preference_accuracy": _accuracy(negative),
        "mean_signed_margin": float(np.mean(margins)) if margins else None,
        "median_signed_margin": float(np.median(margins)) if margins else None,
        "min_signed_margin": float(np.min(margins)) if margins else None,
        "max_signed_margin": float(np.max(margins)) if margins else None,
        "num_wrong_sign": len(nonzero) - len(correct),
    }
    return {
        "schema": "selector_bench.stage2_feedback_generalization.v0",
        "diagnostic": "leave_one_episode_out",
        "feature_dir": str(feature_dir),
        "init_selector_path": str(init_selector_path),
        "episode_jsonl": str(episode_jsonl),
        "work_dir": str(target_dir),
        "run_id": run_id,
        "training_config": {
            "epochs": epochs,
            "lr": lr,
            "l2": l2,
            "reward_scale": reward_scale,
            "target_clip": target_clip,
            "min_abs_reward": min_abs_reward,
            "negative_token_penalty": negative_token_penalty,
            "negative_random_bonus": negative_random_bonus,
            "negative_reward_max": negative_reward_max,
            "seed": seed,
        },
        "summary": summary,
        "folds": fold_reports,
    }


def _fmt(value: Any, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{precision}f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    summary = report["summary"]
    lines = [
        "# Stage 2 Feedback Generalization Diagnostic",
        "",
        f"Diagnostic: `{report['diagnostic']}`",
        f"Episode buffer: `{report['episode_jsonl']}`",
        f"Init selector: `{report['init_selector_path']}`",
        f"Fold work dir: `{report['work_dir']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| folds | {summary['fold_count']} |",
        f"| preference accuracy | {_fmt(summary['preference_accuracy'])} |",
        f"| positive reward accuracy | {_fmt(summary['positive_preference_accuracy'])} |",
        f"| negative reward accuracy | {_fmt(summary['negative_preference_accuracy'])} |",
        f"| wrong sign folds | {summary['num_wrong_sign']} |",
        f"| mean signed margin | {_fmt(summary['mean_signed_margin'])} |",
        f"| min signed margin | {_fmt(summary['min_signed_margin'])} |",
        "",
        "## Folds",
        "",
        "| fold | heldout | reward | selector | random | predicted gap | signed margin | correct | checkpoint |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for fold in report["folds"]:
        heldout = fold["heldout"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(fold["fold"]),
                    str(heldout["episode_id"]),
                    _fmt(float(heldout["reward"])),
                    str(heldout["selector_hash12"]),
                    str(heldout["random_hash12"]),
                    _fmt(float(heldout["predicted_gap_selector_minus_random"])),
                    _fmt(float(heldout["signed_margin"])),
                    str(heldout["preference_correct"]),
                    f"`{fold['checkpoint']}`",
                ]
            )
            + " |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Stage 2 feedback selector generalization with leave-one-episode-out folds."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--init-selector", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--run-id", default="stage2_feedback_generalization")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--target-clip", type=float, default=2.0)
    parser.add_argument("--min-abs-reward", type=float, default=0.0)
    parser.add_argument("--negative-token-penalty", type=float, default=0.0)
    parser.add_argument("--negative-random-bonus", type=float, default=0.0)
    parser.add_argument("--negative-reward-max", type=float, default=-1e-12)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    report = build_leave_one_out_report(
        feature_dir=args.features,
        init_selector_path=args.init_selector,
        episode_jsonl=args.episodes,
        work_dir=args.work_dir,
        run_id=args.run_id,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        reward_scale=args.reward_scale,
        target_clip=args.target_clip,
        min_abs_reward=args.min_abs_reward,
        seed=args.seed,
        negative_token_penalty=args.negative_token_penalty,
        negative_random_bonus=args.negative_random_bonus,
        negative_reward_max=args.negative_reward_max,
    )
    write_json(args.output_json, report)
    write_markdown(report, args.output_md)
    print(f"Wrote Stage 2 generalization JSON: {args.output_json}")
    print(f"Wrote Stage 2 generalization Markdown: {args.output_md}")
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

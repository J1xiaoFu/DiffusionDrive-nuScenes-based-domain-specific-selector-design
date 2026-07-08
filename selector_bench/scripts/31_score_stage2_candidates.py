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
from selector_bench.training.stage2_neural import score_neural_selector
from selector_bench.utils.io import ensure_dir, read_json, read_jsonl, write_json


def _selection_tokens(path: str | Path, manifest_tokens: set[str]) -> list[str]:
    artifact = read_json(path)
    return [
        token
        for token in artifact["selection"]["selected_tokens"]
        if token in manifest_tokens
    ]


def _fmt(value: Any, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{precision}f}"
    return str(value)


def _dedupe_candidates(candidate_report: dict[str, Any], max_candidates: int | None) -> list[dict[str, Any]]:
    items = candidate_report.get("items", [])
    ranked = sorted(items, key=lambda item: int(item.get("rank", 10**9)))
    by_hash: dict[str, dict[str, Any]] = {}
    for item in ranked:
        selection_hash = str(item["selection_hash"])
        if selection_hash in by_hash:
            by_hash[selection_hash].setdefault("aliases", []).append(
                {
                    "rank": int(item.get("rank", 0)),
                    "path": str(item["path"]),
                    "seed": int(item.get("seed", -1)),
                    "replay_penalty": float(item.get("replay_penalty", 0.0)),
                }
            )
            continue
        kept = dict(item)
        kept["aliases"] = []
        by_hash[selection_hash] = kept
    candidates = sorted(by_hash.values(), key=lambda item: int(item.get("rank", 10**9)))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def _episode_reference_sets(rows: list[dict[str, Any]], manifest_tokens: set[str]) -> dict[str, Any]:
    negative_tokens: set[str] = set()
    positive_tokens: set[str] = set()
    historical: list[dict[str, Any]] = []
    for row in rows:
        reward = float(row["reward"])
        selector_tokens = _selection_tokens(row["selector_selection"]["path"], manifest_tokens)
        random_tokens = _selection_tokens(row["random_same_budget_selection"]["path"], manifest_tokens)
        if reward < 0:
            negative_tokens.update(selector_tokens)
        elif reward > 0:
            positive_tokens.update(selector_tokens)
        historical.append(
            {
                "episode_id": str(row["episode_id"]),
                "reward": reward,
                "reward_sign": 1 if reward > 0 else -1 if reward < 0 else 0,
                "selector_hash12": str(row["selector_selection"]["selection_hash"])[:12],
                "random_hash12": str(row["random_same_budget_selection"]["selection_hash"])[:12],
                "selector_tokens": selector_tokens,
                "random_tokens": random_tokens,
            }
        )
    return {
        "negative_tokens": negative_tokens,
        "positive_tokens": positive_tokens,
        "historical": historical,
    }


def _mean_for_tokens(scores_by_token: dict[str, float], tokens: list[str]) -> float:
    values = [scores_by_token[token] for token in tokens if token in scores_by_token]
    if not values:
        return 0.0
    return float(np.mean(values))


def _bootstrap_mean_margin(
    values: np.ndarray,
    samples: int,
    seed: int,
    threshold: float,
    quantile: float,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive.")
    if values.size == 0:
        raise ValueError("bootstrap values cannot be empty.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("bootstrap quantile must be in [0, 1].")

    rng = np.random.default_rng(seed)
    means = np.empty((samples,), dtype=np.float64)
    for idx in range(samples):
        draw = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(draw))
    return {
        "samples": int(samples),
        "seed": int(seed),
        "threshold": float(threshold),
        "quantile": float(quantile),
        "mean": float(np.mean(means)),
        "std": float(np.std(means)),
        "q_low": float(np.quantile(means, quantile)),
        "median": float(np.quantile(means, 0.5)),
        "q_high": float(np.quantile(means, 1.0 - quantile)),
        "pass_rate": float(np.mean(means >= threshold)),
    }


def build_candidate_gate_report(
    feature_dir: str | Path,
    episode_jsonl: str | Path,
    fold_report_json: str | Path,
    candidate_report_json: str | Path,
    max_candidates: int | None = None,
    max_negative_overlap: int = 0,
    min_random_margin: float = 0.0,
    min_positive_margin: float | None = None,
    max_fold_std: float | None = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260706,
    bootstrap_quantile: float = 0.05,
    min_bootstrap_pass_rate: float | None = None,
) -> dict[str, Any]:
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    if min_bootstrap_pass_rate is not None and bootstrap_samples <= 0:
        raise ValueError("min_bootstrap_pass_rate requires bootstrap_samples > 0.")
    if min_bootstrap_pass_rate is not None and not 0.0 <= min_bootstrap_pass_rate <= 1.0:
        raise ValueError("min_bootstrap_pass_rate must be in [0, 1].")

    candidate_report = read_json(candidate_report_json)
    fold_report = read_json(fold_report_json)
    rows = read_jsonl(episode_jsonl)
    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    manifest_tokens = set(store.manifest.tokens(split="train"))
    candidates = _dedupe_candidates(candidate_report, max_candidates=max_candidates)
    if not candidates:
        raise ValueError(f"No candidates found in {candidate_report_json}")

    refs = _episode_reference_sets(rows, manifest_tokens)
    candidate_tokens: dict[str, list[str]] = {
        str(candidate["selection_hash"]): _selection_tokens(candidate["path"], manifest_tokens)
        for candidate in candidates
    }
    all_tokens = sorted(
        {
            token
            for tokens in candidate_tokens.values()
            for token in tokens
        }
        | {
            token
            for row in refs["historical"]
            for token in [*row["selector_tokens"], *row["random_tokens"]]
        }
    )

    fold_rows: list[dict[str, Any]] = []
    for fold in fold_report["folds"]:
        checkpoint = fold["checkpoint"]
        scores = score_neural_selector(store, checkpoint, all_tokens)
        scores_by_token = {token: float(score) for token, score in zip(all_tokens, scores)}
        random_means = [_mean_for_tokens(scores_by_token, row["random_tokens"]) for row in refs["historical"]]
        positive_means = [
            _mean_for_tokens(scores_by_token, row["selector_tokens"])
            for row in refs["historical"]
            if row["reward_sign"] > 0
        ]
        negative_means = [
            _mean_for_tokens(scores_by_token, row["selector_tokens"])
            for row in refs["historical"]
            if row["reward_sign"] < 0
        ]
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "checkpoint": str(checkpoint),
                "random_mean": float(np.mean(random_means)) if random_means else 0.0,
                "positive_selector_mean": float(np.mean(positive_means)) if positive_means else None,
                "negative_selector_mean": float(np.mean(negative_means)) if negative_means else None,
                "scores_by_token": scores_by_token,
            }
        )

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        selection_hash = str(candidate["selection_hash"])
        tokens = candidate_tokens[selection_hash]
        token_set = set(tokens)
        fold_scores = []
        random_margins = []
        positive_margins = []
        negative_margins = []
        for fold in fold_rows:
            mean_score = _mean_for_tokens(fold["scores_by_token"], tokens)
            fold_scores.append(mean_score)
            random_margins.append(mean_score - float(fold["random_mean"]))
            if fold["positive_selector_mean"] is not None:
                positive_margins.append(mean_score - float(fold["positive_selector_mean"]))
            if fold["negative_selector_mean"] is not None:
                negative_margins.append(mean_score - float(fold["negative_selector_mean"]))

        negative_overlap = len(token_set & refs["negative_tokens"])
        positive_overlap = len(token_set & refs["positive_tokens"])
        fold_scores_array = np.asarray(fold_scores, dtype=np.float64)
        random_margins_array = np.asarray(random_margins, dtype=np.float64)
        positive_margins_array = np.asarray(positive_margins, dtype=np.float64)
        negative_margins_array = np.asarray(negative_margins, dtype=np.float64)
        score_gate = bool(float(np.mean(random_margins_array)) >= min_random_margin)
        positive_gate = (
            True
            if min_positive_margin is None
            else bool(float(np.mean(positive_margins_array)) >= min_positive_margin)
        )
        stability_gate = True if max_fold_std is None else bool(float(np.std(fold_scores_array)) <= max_fold_std)
        bootstrap_random_margin = (
            _bootstrap_mean_margin(
                random_margins_array,
                samples=bootstrap_samples,
                seed=bootstrap_seed + int(candidate.get("rank", 0)),
                threshold=min_random_margin,
                quantile=bootstrap_quantile,
            )
            if bootstrap_samples > 0
            else None
        )
        bootstrap_gate = (
            True
            if min_bootstrap_pass_rate is None
            else bool(
                bootstrap_random_margin is not None
                and float(bootstrap_random_margin["pass_rate"]) >= min_bootstrap_pass_rate
            )
        )
        replay_gate = negative_overlap <= max_negative_overlap
        gates = {
            "negative_overlap": replay_gate,
            "random_margin": score_gate,
            "positive_margin": positive_gate,
            "fold_stability": stability_gate,
            "bootstrap_random_margin": bootstrap_gate,
        }
        verdict = "pass" if all(gates.values()) else "reject"
        scored_candidates.append(
            {
                "rank": int(candidate.get("rank", 0)),
                "selection_hash": selection_hash,
                "hash12": str(candidate.get("hash12", selection_hash[:12])),
                "path": str(candidate["path"]),
                "seed": int(candidate.get("seed", -1)),
                "replay_penalty": float(candidate.get("replay_penalty", 0.0)),
                "selected_count": len(tokens),
                "negative_overlap": negative_overlap,
                "positive_overlap": positive_overlap,
                "candidate_report_negative_overlap": int(candidate.get("negative_overlap", negative_overlap)),
                "v4_overlap": int(candidate.get("v4_overlap", 0)),
                "v5_overlap": int(candidate.get("v5_overlap", 0)),
                "v6_overlap": int(candidate.get("v6_overlap", 0)),
                "original_score_mean": float(candidate.get("score_mean", 0.0)),
                "fold_score_mean": float(np.mean(fold_scores_array)),
                "fold_score_std": float(np.std(fold_scores_array)),
                "fold_score_min": float(np.min(fold_scores_array)),
                "fold_score_max": float(np.max(fold_scores_array)),
                "candidate_minus_random_mean": float(np.mean(random_margins_array)),
                "candidate_minus_random_min": float(np.min(random_margins_array)),
                "candidate_minus_random_by_fold": random_margins_array.astype(float).tolist(),
                "candidate_minus_positive_mean": float(np.mean(positive_margins_array)) if positive_margins else None,
                "candidate_minus_negative_mean": float(np.mean(negative_margins_array)) if negative_margins else None,
                "bootstrap_random_margin": bootstrap_random_margin,
                "gates": gates,
                "verdict": verdict,
                "aliases": candidate.get("aliases", []),
            }
        )

    ranked = sorted(
        scored_candidates,
        key=lambda item: (
            0 if item["verdict"] == "pass" else 1,
            -float(
                item["bootstrap_random_margin"]["q_low"]
                if item.get("bootstrap_random_margin") is not None
                else float("-inf")
            ),
            -float(item["candidate_minus_random_mean"]),
            int(item["negative_overlap"]),
            -float(item["fold_score_mean"]),
            int(item["rank"]),
        ),
    )
    for gate_rank, item in enumerate(ranked, start=1):
        item["gate_rank"] = gate_rank

    passed = [item for item in ranked if item["verdict"] == "pass"]
    summary = {
        "candidate_count": len(scored_candidates),
        "pass_count": len(passed),
        "best_hash12": ranked[0]["hash12"],
        "best_verdict": ranked[0]["verdict"],
        "best_candidate_minus_random_mean": ranked[0]["candidate_minus_random_mean"],
        "best_negative_overlap": ranked[0]["negative_overlap"],
        "best_bootstrap_pass_rate": (
            ranked[0]["bootstrap_random_margin"]["pass_rate"]
            if ranked[0].get("bootstrap_random_margin") is not None
            else None
        ),
        "best_bootstrap_q_low": (
            ranked[0]["bootstrap_random_margin"]["q_low"]
            if ranked[0].get("bootstrap_random_margin") is not None
            else None
        ),
        "canonical_hash12": str(candidate_report.get("best", {}).get("hash12", "")),
    }
    return {
        "schema": "selector_bench.stage2_candidate_gate.v0",
        "feature_dir": str(feature_dir),
        "episode_jsonl": str(episode_jsonl),
        "fold_report_json": str(fold_report_json),
        "candidate_report_json": str(candidate_report_json),
        "gate_config": {
            "max_candidates": max_candidates,
            "max_negative_overlap": max_negative_overlap,
            "min_random_margin": min_random_margin,
            "min_positive_margin": min_positive_margin,
            "max_fold_std": max_fold_std,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_quantile": bootstrap_quantile,
            "min_bootstrap_pass_rate": min_bootstrap_pass_rate,
        },
        "reference_summary": {
            "episode_count": len(rows),
            "fold_count": len(fold_rows),
            "negative_token_count": len(refs["negative_tokens"]),
            "positive_token_count": len(refs["positive_tokens"]),
        },
        "summary": summary,
        "candidates": ranked,
    }


def write_markdown(report: dict[str, Any], path: str | Path, limit: int = 40) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    summary = report["summary"]
    lines = [
        "# Stage 2 Candidate Gate",
        "",
        f"Candidate report: `{report['candidate_report_json']}`",
        f"Fold report: `{report['fold_report_json']}`",
        f"Episode buffer: `{report['episode_jsonl']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| candidates | {summary['candidate_count']} |",
        f"| pass count | {summary['pass_count']} |",
        f"| best hash | `{summary['best_hash12']}` |",
        f"| best verdict | {summary['best_verdict']} |",
        f"| best minus random | {_fmt(summary['best_candidate_minus_random_mean'])} |",
        f"| best negative overlap | {summary['best_negative_overlap']} |",
        f"| best bootstrap pass rate | {_fmt(summary['best_bootstrap_pass_rate'])} |",
        f"| best bootstrap q_low | {_fmt(summary['best_bootstrap_q_low'])} |",
        f"| canonical hash | `{summary['canonical_hash12']}` |",
        "",
        "## Ranked Candidates",
        "",
        "| gate_rank | verdict | hash | orig_rank | seed | penalty | neg_overlap | fold_score | score_std | minus_random | boot_pass | boot_q05 | minus_positive | v4/v5/v6 | path |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report["candidates"][:limit]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["gate_rank"]),
                    str(item["verdict"]),
                    f"`{item['hash12']}`",
                    str(item["rank"]),
                    str(item["seed"]),
                    _fmt(float(item["replay_penalty"]), precision=3),
                    str(item["negative_overlap"]),
                    _fmt(float(item["fold_score_mean"])),
                    _fmt(float(item["fold_score_std"])),
                    _fmt(float(item["candidate_minus_random_mean"])),
                    _fmt(
                        item["bootstrap_random_margin"]["pass_rate"]
                        if item.get("bootstrap_random_margin") is not None
                        else None
                    ),
                    _fmt(
                        item["bootstrap_random_margin"]["q_low"]
                        if item.get("bootstrap_random_margin") is not None
                        else None
                    ),
                    _fmt(item["candidate_minus_positive_mean"]),
                    f"{item['v4_overlap']}/{item['v5_overlap']}/{item['v6_overlap']}",
                    f"`{item['path']}`",
                ]
            )
            + " |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Stage 2 candidates with a fold-ensemble offline gate.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--fold-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-negative-overlap", type=int, default=0)
    parser.add_argument("--min-random-margin", type=float, default=0.0)
    parser.add_argument("--min-positive-margin", type=float, default=None)
    parser.add_argument("--max-fold-std", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260706)
    parser.add_argument("--bootstrap-quantile", type=float, default=0.05)
    parser.add_argument("--min-bootstrap-pass-rate", type=float, default=None)
    parser.add_argument("--markdown-limit", type=int, default=40)
    args = parser.parse_args()

    report = build_candidate_gate_report(
        feature_dir=args.features,
        episode_jsonl=args.episodes,
        fold_report_json=args.fold_report,
        candidate_report_json=args.candidate_report,
        max_candidates=args.max_candidates,
        max_negative_overlap=args.max_negative_overlap,
        min_random_margin=args.min_random_margin,
        min_positive_margin=args.min_positive_margin,
        max_fold_std=args.max_fold_std,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_quantile=args.bootstrap_quantile,
        min_bootstrap_pass_rate=args.min_bootstrap_pass_rate,
    )
    write_json(args.output_json, report)
    write_markdown(report, args.output_md, limit=args.markdown_limit)
    print(f"Wrote Stage 2 candidate gate JSON: {args.output_json}")
    print(f"Wrote Stage 2 candidate gate Markdown: {args.output_md}")
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.core.baseline_adapter import (
    DomainStratifiedAdapter,
    FullDataAdapter,
    MosaicLiteAdapter,
)
from selector_bench.core.feature_store import FeatureStore
from selector_bench.core.score_provider import (
    MosaicLiteOriginalScoreProvider,
    OurSelectorScoreProvider,
    RandomScoreProvider,
    TeacherLossScoreProvider,
    TeacherUncertaintyScoreProvider,
)
from selector_bench.core.selection_runner import run_selection


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


def build_adapter(name: str):
    if name == "mosaic":
        return MosaicLiteAdapter()
    if name in {"random", "loss_mining", "uncertainty"}:
        return DomainStratifiedAdapter()
    if name == "full_data":
        return FullDataAdapter()
    raise ValueError(f"Unknown baseline: {name}")


def build_score_provider(score: str, baseline: str, store: FeatureStore, selector: str | None, seed: int):
    manifest = store.manifest
    if score == "original":
        if baseline == "mosaic":
            return MosaicLiteOriginalScoreProvider(manifest)
        if baseline == "loss_mining":
            return TeacherLossScoreProvider(manifest)
        if baseline == "uncertainty":
            return TeacherUncertaintyScoreProvider(manifest)
        if baseline == "random":
            return RandomScoreProvider(seed=seed)
        if baseline == "full_data":
            return RandomScoreProvider(seed=seed)
    if score == "random":
        return RandomScoreProvider(seed=seed)
    if score == "loss":
        return TeacherLossScoreProvider(manifest)
    if score == "uncertainty":
        return TeacherUncertaintyScoreProvider(manifest)
    if score == "ours":
        if not selector:
            raise ValueError("--selector is required when --score ours.")
        return OurSelectorScoreProvider(manifest, store, selector)
    raise ValueError(f"Unsupported score '{score}' for baseline '{baseline}'.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--baseline", required=True, choices=["random", "mosaic", "loss_mining", "uncertainty", "full_data"])
    parser.add_argument("--score", required=True, choices=["original", "random", "loss", "uncertainty", "ours"])
    parser.add_argument("--selector", default=None)
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    store = FeatureStore.from_feature_dir(args.features, cache=True)
    adapter = build_adapter(args.baseline)
    provider = build_score_provider(args.score, args.baseline, store, args.selector, args.seed)
    budget = float(args.budget_count) if args.budget_count is not None else float(args.budget)
    exclude_tokens = _load_token_file(args.exclude_tokens_file)
    run_id = args.run_id or f"{args.baseline}_{args.score}_b{budget}_s{args.seed}"

    artifact = run_selection(
        manifest=store.manifest,
        adapter=adapter,
        score_provider=provider,
        budget=budget,
        output_path=args.output,
        run_id=run_id,
        split=args.split,
        config={
            "features": args.features,
            "baseline": args.baseline,
            "score": args.score,
            "selector": args.selector,
            "budget": budget,
            "budget_count": args.budget_count,
            "seed": args.seed,
            "exclude_tokens_file": args.exclude_tokens_file,
            "excluded_token_count": len(exclude_tokens),
        },
        input_paths=[Path(args.features) / "manifest.jsonl"],
        cwd=ROOT.parent,
        exclude_tokens=exclude_tokens,
    )
    selection = artifact["selection"]
    print(f"Wrote selection artifact: {args.output}")
    print(
        {
            "selected_count": selection["selected_count"],
            "selection_hash": selection["selection_hash"][:12],
            "domain_counts": selection["domain_counts"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

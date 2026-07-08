from __future__ import annotations

from pathlib import Path
from typing import Any

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.baseline_adapter import BaselineAdapter, SelectionResult
from selector_bench.core.manifest import Manifest
from selector_bench.core.score_provider import ScoreProvider
from selector_bench.utils.hashing import hash_jsonable
from selector_bench.utils.io import write_json


def selection_to_payload(result: SelectionResult) -> dict[str, Any]:
    return {
        "baseline": result.baseline,
        "score_source": result.score_source,
        "budget_ratio": result.budget_ratio,
        "budget_mode": result.budget_mode,
        "budget_count": result.budget_count,
        "split": result.split,
        "selected_count": len(result.selected_tokens),
        "selected_tokens": result.selected_tokens,
        "selection_hash": hash_jsonable(result.selected_tokens),
        "domain_budget": {str(k): v for k, v in result.domain_budget.items()},
        "domain_counts": {str(k): v for k, v in result.domain_counts.items()},
        "exclusions": {
            "excluded_token_count": result.excluded_token_count,
        },
        "score_summary": result.score_summary,
    }


def run_selection(
    manifest: Manifest,
    adapter: BaselineAdapter,
    score_provider: ScoreProvider,
    budget: float,
    output_path: str | Path,
    run_id: str,
    split: str = "train",
    config: dict[str, Any] | None = None,
    input_paths: list[str | Path] | None = None,
    cwd: str | Path | None = None,
    exclude_tokens: set[str] | None = None,
) -> dict[str, Any]:
    result = adapter.select(
        manifest,
        score_provider,
        budget=budget,
        split=split,
        exclude_tokens=exclude_tokens,
    )
    payload = selection_to_payload(result)
    metadata = make_artifact_metadata(
        artifact_type="selection",
        run_id=run_id,
        config=config or {},
        input_paths=input_paths,
        cwd=cwd,
        summary={
            "selected_count": payload["selected_count"],
            "selection_hash": payload["selection_hash"],
        },
    )
    artifact = {"metadata": metadata, "selection": payload}
    write_json(output_path, artifact)
    return artifact

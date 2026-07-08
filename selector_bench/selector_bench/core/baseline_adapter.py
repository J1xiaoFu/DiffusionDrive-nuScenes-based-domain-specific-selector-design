from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from selector_bench.core.budget_allocator import budget_count, proportional_domain_budget
from selector_bench.core.manifest import Manifest, ManifestRecord
from selector_bench.core.score_provider import MosaicLiteOriginalScoreProvider, ScoreProvider


@dataclass
class SelectionResult:
    baseline: str
    score_source: str
    budget_ratio: float
    budget_mode: str
    budget_count: int
    split: str
    selected_tokens: list[str]
    domain_budget: dict[int, int]
    domain_counts: dict[int, int]
    score_summary: dict[str, Any]
    excluded_token_count: int = 0


class BaselineAdapter:
    name = "base"

    def allocate_budget(self, records: list[ManifestRecord], total_budget: int) -> dict[int, int]:
        return proportional_domain_budget(records, total_budget)

    def select(
        self,
        manifest: Manifest,
        score_provider: ScoreProvider,
        budget: float,
        split: str = "train",
        exclude_tokens: set[str] | None = None,
    ) -> SelectionResult:
        excluded = exclude_tokens or set()
        records = [
            record for record in manifest.filter(split=split)
            if record.sample_token not in excluded
        ]
        total_budget = budget_count(len(records), budget)
        domain_budget = self.allocate_budget(records, total_budget)
        selected: list[str] = []
        selected_scores: list[float] = []
        all_scores: list[float] = []

        by_domain: dict[int, list[ManifestRecord]] = {}
        for record in records:
            by_domain.setdefault(record.domain_id, []).append(record)
        for domain_id, domain_records in sorted(by_domain.items()):
            k = domain_budget.get(domain_id, 0)
            if k <= 0:
                continue
            tokens = [record.sample_token for record in domain_records]
            scores = score_provider.score(tokens, domain_id=domain_id)
            all_scores.extend(float(score) for score in scores)
            ranked = sorted(zip(tokens, scores), key=lambda item: (-float(item[1]), item[0]))
            chosen = ranked[:k]
            selected.extend(token for token, _ in chosen)
            selected_scores.extend(float(score) for _, score in chosen)

        domain_counts = {
            domain_id: sum(1 for token in selected if manifest.get(token).domain_id == domain_id)
            for domain_id in sorted(by_domain)
        }
        return SelectionResult(
            baseline=self.name,
            score_source=score_provider.name,
            budget_ratio=budget,
            budget_mode="ratio" if budget <= 1 else "exact_count",
            budget_count=total_budget,
            split=split,
            selected_tokens=selected,
            domain_budget=domain_budget,
            domain_counts=domain_counts,
            score_summary=self._score_summary(all_scores, selected_scores),
            excluded_token_count=len(excluded),
        )

    @staticmethod
    def _score_summary(all_scores: list[float], selected_scores: list[float]) -> dict[str, Any]:
        def mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "all_score_mean": mean(all_scores),
            "selected_score_mean": mean(selected_scores),
            "selected_score_min": float(min(selected_scores)) if selected_scores else 0.0,
            "selected_score_max": float(max(selected_scores)) if selected_scores else 0.0,
        }


class DomainStratifiedAdapter(BaselineAdapter):
    name = "domain_stratified"


class MosaicLiteAdapter(BaselineAdapter):
    name = "mosaic_lite"

    def allocate_budget(self, records: list[ManifestRecord], total_budget: int) -> dict[int, int]:
        domain_values: dict[int, list[float]] = {}
        manifest = Manifest.from_records(records)
        score_provider = MosaicLiteOriginalScoreProvider(manifest)
        for record in records:
            score = float(score_provider.score([record.sample_token], record.domain_id)[0])
            domain_values.setdefault(record.domain_id, []).append(score)
        weights = {
            domain_id: len(values) * (0.50 + float(np.mean(values)))
            for domain_id, values in domain_values.items()
        }
        return proportional_domain_budget(records, total_budget, weights=weights)


class FullDataAdapter(BaselineAdapter):
    name = "full_data"

    def select(
        self,
        manifest: Manifest,
        score_provider: ScoreProvider,
        budget: float,
        split: str = "train",
        exclude_tokens: set[str] | None = None,
    ) -> SelectionResult:
        excluded = exclude_tokens or set()
        records = [
            record for record in manifest.filter(split=split)
            if record.sample_token not in excluded
        ]
        selected = [record.sample_token for record in records]
        domain_counts = {
            domain_id: sum(1 for record in records if record.domain_id == domain_id)
            for domain_id in sorted({record.domain_id for record in records})
        }
        return SelectionResult(
            baseline=self.name,
            score_source=score_provider.name,
            budget_ratio=1.0,
            budget_mode="full_data",
            budget_count=len(selected),
            split=split,
            selected_tokens=selected,
            domain_budget=domain_counts,
            domain_counts=domain_counts,
            score_summary={},
            excluded_token_count=len(excluded),
        )

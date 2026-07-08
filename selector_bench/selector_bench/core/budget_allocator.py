from __future__ import annotations

import math
from collections import Counter

from selector_bench.core.manifest import ManifestRecord


def budget_count(num_items: int, budget: float) -> int:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if budget <= 1:
        return max(1, int(math.ceil(num_items * budget)))
    return min(num_items, int(budget))


def proportional_domain_budget(
    records: list[ManifestRecord],
    total_budget: int,
    weights: dict[int, float] | None = None,
) -> dict[int, int]:
    counts = Counter(record.domain_id for record in records)
    if not counts:
        return {}

    if weights is None:
        weights = {domain_id: float(count) for domain_id, count in counts.items()}
    else:
        weights = {
            domain_id: max(0.0, float(weights.get(domain_id, 0.0))) for domain_id in counts
        }

    if sum(weights.values()) <= 0:
        weights = {domain_id: float(count) for domain_id, count in counts.items()}

    raw = {
        domain_id: total_budget * weights[domain_id] / sum(weights.values())
        for domain_id in counts
    }
    budgets = {
        domain_id: min(counts[domain_id], int(math.floor(value)))
        for domain_id, value in raw.items()
    }

    remaining = total_budget - sum(budgets.values())
    order = sorted(
        counts,
        key=lambda domain_id: (raw[domain_id] - math.floor(raw[domain_id]), counts[domain_id]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for domain_id in order:
            if remaining <= 0:
                break
            if budgets[domain_id] < counts[domain_id]:
                budgets[domain_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return dict(sorted(budgets.items()))

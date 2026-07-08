#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.utils.io import ensure_dir, read_jsonl, write_json


def reward_cost(metrics: dict[str, float], rho_box: float, rho_obj: float) -> float:
    return (
        float(metrics["l2"])
        + rho_box * float(metrics.get("obj_box_col", 0.0))
        + rho_obj * float(metrics.get("obj_col", 0.0))
    )


def reward(
    selector: dict[str, float],
    random_same_budget: dict[str, float],
    rho_box: float,
    rho_obj: float,
) -> float:
    return reward_cost(random_same_budget, rho_box, rho_obj) - reward_cost(selector, rho_box, rho_obj)


def break_even_rho_box(
    selector: dict[str, float],
    random_same_budget: dict[str, float],
    rho_obj: float,
) -> float | None:
    delta_box = float(selector.get("obj_box_col", 0.0)) - float(random_same_budget.get("obj_box_col", 0.0))
    if abs(delta_box) < 1e-12:
        return None
    numerator = (
        float(random_same_budget["l2"])
        - float(selector["l2"])
        + rho_obj * (float(random_same_budget.get("obj_col", 0.0)) - float(selector.get("obj_col", 0.0)))
    )
    value = numerator / delta_box
    if value < 0.0 or not math.isfinite(value):
        return None
    return value


def _hash12(row: dict[str, Any]) -> str:
    return str(row["selector_selection"]["selection_hash"])[:12]


def _fmt_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def build_report(
    episode_jsonl: str | Path,
    rho_boxes: list[float],
    rho_obj: float,
    decision_rho_box: float,
) -> dict[str, Any]:
    rows = read_jsonl(episode_jsonl)
    if not rows:
        raise ValueError(f"No feedback episodes found in {episode_jsonl}")

    episode_reports: list[dict[str, Any]] = []
    for row in rows:
        if row.get("schema") != "selector_bench.stage2_episode.v0":
            raise ValueError(f"Unexpected episode schema: {row.get('schema')}")
        selector = {key: float(value) for key, value in row["metrics"]["selector"].items()}
        random_same_budget = {
            key: float(value) for key, value in row["metrics"]["random_same_budget"].items()
        }
        reward_by_rho = {
            str(rho_box): reward(selector, random_same_budget, rho_box=rho_box, rho_obj=rho_obj)
            for rho_box in rho_boxes
        }
        current_rho_box = float(row["reward_definition"]["rho_box"])
        episode_reports.append(
            {
                "episode_id": row["episode_id"],
                "selector_hash12": _hash12(row),
                "random_hash12": str(row["random_same_budget_selection"]["selection_hash"])[:12],
                "current_rho_box": current_rho_box,
                "current_reward": float(row["reward"]),
                "delta_l2_selector_minus_random": selector["l2"] - random_same_budget["l2"],
                "delta_obj_box_col_fraction_selector_minus_random": selector.get("obj_box_col", 0.0)
                - random_same_budget.get("obj_box_col", 0.0),
                "delta_obj_box_col_percentage_points_selector_minus_random": 100.0
                * (selector.get("obj_box_col", 0.0) - random_same_budget.get("obj_box_col", 0.0)),
                "delta_obj_col_fraction_selector_minus_random": selector.get("obj_col", 0.0)
                - random_same_budget.get("obj_col", 0.0),
                "break_even_rho_box": break_even_rho_box(
                    selector,
                    random_same_budget,
                    rho_obj=rho_obj,
                ),
                "reward_by_rho_box": reward_by_rho,
            }
        )

    return {
        "source_episode_jsonl": str(episode_jsonl),
        "episode_count": len(rows),
        "collision_units": "fractions, not percentages",
        "rho_box_interpretation": "One +1 percentage point obj_box_col gap changes cost by 0.01*rho_box.",
        "rho_obj": rho_obj,
        "rho_boxes": rho_boxes,
        "decision": {
            "comparable_reward_rho_box": 0.05,
            "future_collision_aware_candidate_rho_box": decision_rho_box,
            "rationale": (
                "Keep the comparable feedback buffer on rho_box=0.05 for direct history tracking. "
                "Use separate reweighted buffers or reports for collision-aware rho_box values so "
                "reward scales are not silently mixed."
            ),
        },
        "episodes": episode_reports,
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    rho_boxes = [float(value) for value in report["rho_boxes"]]
    headers = [
        "episode",
        "selector",
        "dL2 sel-rand",
        "dBox pp",
        "break-even rho_box",
        *[f"R@{rho:g}" for rho in rho_boxes],
    ]
    lines = [
        "# Stage 2 Collision-Weight Sensitivity",
        "",
        f"Source: `{report['source_episode_jsonl']}`",
        "",
        "Collision metrics are stored as fractions. A +1 percentage point box-collision gap changes "
        "the scalar cost by `0.01 * rho_box`.",
        "",
        "Decision: keep the comparable feedback buffer on `rho_box=0.05` for direct history "
        "tracking. Use separate reweighted buffers or reports for collision-aware `rho_box` "
        "values so reward scales are not silently mixed.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for episode in report["episodes"]:
        reward_values = [
            _fmt_float(float(episode["reward_by_rho_box"][str(rho)]), precision=6)
            for rho in rho_boxes
        ]
        values = [
            str(episode["episode_id"]),
            str(episode["selector_hash12"]),
            _fmt_float(float(episode["delta_l2_selector_minus_random"]), precision=6),
            _fmt_float(
                float(episode["delta_obj_box_col_percentage_points_selector_minus_random"]),
                precision=3,
            ),
            _fmt_float(episode["break_even_rho_box"], precision=3),
            *reward_values,
        ]
        lines.append("| " + " | ".join(values) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report Stage 2 reward sensitivity to box-collision penalty."
    )
    parser.add_argument("--episode-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--rho-boxes",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 10.0, 50.0, 100.0, 250.0, 500.0],
    )
    parser.add_argument("--rho-obj", type=float, default=0.05)
    parser.add_argument("--decision-rho-box", type=float, default=100.0)
    args = parser.parse_args()

    report = build_report(
        episode_jsonl=args.episode_jsonl,
        rho_boxes=args.rho_boxes,
        rho_obj=args.rho_obj,
        decision_rho_box=args.decision_rho_box,
    )
    write_json(args.output_json, report)
    write_markdown(report, args.output_md)
    print(f"Wrote collision sensitivity JSON: {args.output_json}")
    print(f"Wrote collision sensitivity Markdown: {args.output_md}")
    for episode in report["episodes"]:
        print(
            {
                "episode_id": episode["episode_id"],
                "selector": episode["selector_hash12"],
                "delta_box_pp": round(
                    episode["delta_obj_box_col_percentage_points_selector_minus_random"],
                    3,
                ),
                "break_even_rho_box": (
                    None
                    if episode["break_even_rho_box"] is None
                    else round(episode["break_even_rho_box"], 3)
                ),
                "reward_at_decision_rho": round(
                    episode["reward_by_rho_box"][str(args.decision_rho_box)],
                    6,
                ),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

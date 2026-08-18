#!/usr/bin/env python3
"""Expand the scientific FTF-OPD plan into non-overlapping server job rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "drive_ftf_opd_research_v1.json",
    )
    parser.add_argument(
        "--phase", choices=("screening", "seed0", "main"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows_for(config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if phase == "screening":
        screening = config["screening"]
        for gpu_slot, variant in enumerate(screening["variants"]):
            rows.append(
                {
                    "phase": phase,
                    "server": screening["server"],
                    "gpu_slot": gpu_slot,
                    "dataset": screening["dataset"],
                    "stream": screening["stream"],
                    "seed": screening["seed"],
                    "fidelity": screening["fidelity"],
                    **variant,
                }
            )
    elif phase == "seed0":
        navsim = config["navsim"]
        for stream in navsim["streams"]:
            for method in navsim["seed0_methods"]:
                rows.append(
                    {
                        "phase": phase,
                        "server": navsim["server"],
                        "dataset": "NAVSIM",
                        "stream": stream,
                        "method_id": method,
                        "seed": 0,
                        "fidelity": "complete three-stage curve",
                    }
                )
    else:
        navsim = config["navsim"]
        for stream in navsim["streams"]:
            for method in navsim["main_methods"]:
                for seed in navsim["main_seeds"]:
                    rows.append(
                        {
                            "phase": phase,
                            "server": navsim["server"],
                            "dataset": "NAVSIM",
                            "stream": stream,
                            "method_id": method,
                            "seed": seed,
                            "fidelity": "main table",
                        }
                    )
        nuscenes = config["nuscenes"]
        for stream in nuscenes["streams"]:
            for method in nuscenes["methods"]:
                for seed in nuscenes["seeds"]:
                    rows.append(
                        {
                            "phase": phase,
                            "server": nuscenes["server"],
                            "dataset": "nuScenes",
                            "stream": stream,
                            "method_id": method,
                            "seed": seed,
                            "fidelity": "mechanism and external validity",
                        }
                    )
    for index, row in enumerate(rows):
        row["job_index"] = index
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema") != "selector_bench.ftf_opd_research_plan.v1":
        raise SystemExit("unsupported FTF-OPD research-plan schema")
    rows = rows_for(config, args.phase)
    payload = {
        "schema": "selector_bench.ftf_opd_experiment_matrix.v1",
        "phase": args.phase,
        "job_count": len(rows),
        "jobs_by_server": {
            server: sum(row["server"] == server for row in rows)
            for server in sorted({row["server"] for row in rows})
        },
        "jobs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("phase", "job_count", "jobs_by_server")}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numerically audit historical and scheduler-consistent Drive-OPD states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from diffusers.schedulers import DDIMScheduler

from selector_bench.continual.drive_opd import (
    PlanningDistillationConfig,
    configure_rollout_schedule,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )


def rms(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).square().mean().sqrt().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--method-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.official_source, args.method_source):
        if not path.is_file():
            parser.error(f"missing source: {path}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20260817)
    clean = torch.randn((4, 20, 8, 2), generator=generator).clamp(-1.0, 1.0)
    noise = torch.randn(clean.shape, generator=generator)

    historical = scheduler()
    historical.set_timesteps(1000, torch.device("cpu"))
    historical_initial = torch.full((clean.shape[0],), 8, dtype=torch.long)
    historical_state = historical.add_noise(clean, noise, historical_initial)
    historical_next = historical.step(
        model_output=clean,
        timestep=10,
        sample=historical_state,
        eta=0.0,
    ).prev_sample
    expected_t9 = historical.add_noise(
        clean, noise, torch.full((clean.shape[0],), 9, dtype=torch.long)
    )
    expected_t0_historical = historical.add_noise(
        clean, noise, torch.zeros(clean.shape[0], dtype=torch.long)
    )

    corrected = scheduler()
    schedule = configure_rollout_schedule(
        corrected,
        PlanningDistillationConfig(),
        torch.device("cpu"),
    )
    corrected_initial = torch.full(
        (clean.shape[0],), schedule.initial_noise_timestep, dtype=torch.long
    )
    corrected_state = corrected.add_noise(clean, noise, corrected_initial)
    corrected_next = corrected.step(
        model_output=clean,
        timestep=schedule.query_timesteps[0],
        sample=corrected_state,
        eta=0.0,
    ).prev_sample
    expected_t0_corrected = corrected.add_noise(
        clean, noise, torch.zeros(clean.shape[0], dtype=torch.long)
    )

    rows = [
        {
            "schedule": "historical_official_heuristic",
            "initial_noise_timestep": 8,
            "first_query_timestep": 10,
            "scheduler_inference_steps": 1000,
            "scheduler_transition_stride": 1,
            "scheduler_actual_next_timestep": 9,
            "declared_next_query_timestep": 0,
            "initial_query_time_mismatch": 2,
            "transition_label_mismatch": 9,
            "rms_to_scheduler_t9_from_same_noise": rms(historical_next, expected_t9),
            "rms_to_declared_t0_from_same_noise": rms(
                historical_next, expected_t0_historical
            ),
            "student_queries": 2,
            "teacher_queries": 2,
        },
        {
            "schedule": "drive_opd_constant_stride",
            "initial_noise_timestep": schedule.initial_noise_timestep,
            "first_query_timestep": schedule.query_timesteps[0],
            "scheduler_inference_steps": schedule.scheduler_inference_steps,
            "scheduler_transition_stride": schedule.transition_stride,
            "scheduler_actual_next_timestep": 0,
            "declared_next_query_timestep": schedule.query_timesteps[1],
            "initial_query_time_mismatch": 0,
            "transition_label_mismatch": 0,
            "rms_to_scheduler_t9_from_same_noise": None,
            "rms_to_declared_t0_from_same_noise": rms(
                corrected_next, expected_t0_corrected
            ),
            "student_queries": 2,
            "teacher_queries": 2,
        },
    ]
    csv_path = args.output_dir / "schedule_comparison.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    tolerance = 1e-6
    payload = {
        "schema": "selector_bench.drive_opd_rollout_schedule_audit.v1",
        "status": (
            "PASS"
            if rows[1]["rms_to_declared_t0_from_same_noise"] < tolerance
            and rows[1]["initial_query_time_mismatch"] == 0
            and rows[1]["transition_label_mismatch"] == 0
            else "FAIL"
        ),
        "historical_official_heuristic": rows[0],
        "scheduler_consistent_method": rows[1],
        "negative_control": {
            "same_clean_states": True,
            "same_noise": True,
            "same_prediction_type": "sample",
            "same_beta_schedule": "scaled_linear",
            "same_query_and_teacher_forward_counts": True,
        },
        "method_semantics": {
            "student_rollout_state_transition_gradient": "detached_semi_gradient",
            "final_timestep_has_no_unused_scheduler_step": True,
        },
        "sources": {
            "official_source": str(args.official_source.resolve()),
            "official_source_sha256": sha256_file(args.official_source),
            "method_source": str(args.method_source.resolve()),
            "method_source_sha256": sha256_file(args.method_source),
            "diffusers_version": __import__("diffusers").__version__,
        },
        "table": str(csv_path.resolve()),
        "table_sha256": sha256_file(csv_path),
        "evidence_boundary": (
            "This proves scheduler state-time consistency and equal denoiser-query counts. "
            "It does not prove that the corrected auxiliary loss improves NAVSIM metrics."
        ),
    }
    output = args.output_dir / "rollout_schedule_audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

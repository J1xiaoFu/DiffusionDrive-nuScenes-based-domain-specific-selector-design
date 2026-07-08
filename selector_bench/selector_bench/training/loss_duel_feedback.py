from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.training.stage2_high_frequency import (
    _run_checked,
    _successful_train_status,
)
from selector_bench.utils.io import read_json


def parse_train_loss_log(
    work_dir: str | Path,
    loss_key: str = "loss",
    tail_window: int = 3,
) -> dict[str, Any]:
    """Parse MMCV JSON train logs and summarize one scalar loss series."""

    if tail_window < 1:
        raise ValueError("tail_window must be positive.")
    root = Path(work_dir)
    log_paths = sorted(root.glob("*.log.json"))
    if not log_paths:
        raise FileNotFoundError(f"No *.log.json files found in {root}")

    rows: list[dict[str, Any]] = []
    for path in log_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if row.get("mode") != "train":
                continue
            value = row.get(loss_key)
            if isinstance(value, (int, float)):
                rows.append(row)
    if not rows:
        raise ValueError(f"No train rows with numeric {loss_key!r} found in {root}")

    losses = np.asarray([float(row[loss_key]) for row in rows], dtype=np.float64)
    iters = [int(row.get("iter", idx + 1)) for idx, row in enumerate(rows)]
    window = min(int(tail_window), len(losses))
    first_mean = float(np.mean(losses[:window]))
    tail_mean = float(np.mean(losses[-window:]))
    best = float(np.min(losses))
    delta = first_mean - tail_mean
    relative_delta = delta / max(abs(first_mean), 1e-6)
    return {
        "log_path": str(log_paths[-1]),
        "loss_key": loss_key,
        "num_points": int(len(losses)),
        "iters": iters,
        "losses": losses.astype(float).tolist(),
        "first_mean": first_mean,
        "tail_mean": tail_mean,
        "best": best,
        "delta": float(delta),
        "relative_delta": float(relative_delta),
    }


def loss_summary_to_metric(summary: dict[str, Any], mode: str) -> float:
    """Return a lower-is-better scalar suitable for stage2 reward_cost l2."""

    if mode == "final":
        return float(summary["tail_mean"])
    if mode == "best":
        return float(summary["best"])
    if mode == "drop":
        return -float(summary["delta"])
    if mode == "relative_drop":
        return -float(summary["relative_delta"])
    raise ValueError(f"Unsupported loss utility mode: {mode}")


def run_train_loss_proxy(
    config: str | Path,
    work_dir: str | Path,
    diffusiondrive_root: str | Path,
    seed: int = 0,
    diffusiondrive_python: str | Path | None = None,
    wrapper_python: str | Path | None = None,
    loss_key: str = "loss",
    tail_window: int = 3,
    utility_mode: str = "drop",
) -> dict[str, float | str | list[float] | list[int]]:
    """Run train-only DiffusionDrive feedback and convert its log loss to metrics."""

    train_work_dir = Path(work_dir)
    train_status = train_work_dir / "run_status.json"
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    wrapper = str(wrapper_python or sys.executable)
    if not _successful_train_status(train_status):
        cmd = [
            wrapper,
            str(scripts_root / "20_train_diffusiondrive_checkpoint.py"),
            "--diffusiondrive-root",
            str(diffusiondrive_root),
            "--config",
            str(config),
            "--work-dir",
            str(train_work_dir),
            "--seed",
            str(seed),
            "--no-validate",
            "--status-json",
            str(train_status),
        ]
        if diffusiondrive_python is not None:
            cmd.extend(["--python", str(diffusiondrive_python)])
        _run_checked(cmd)

    summary = parse_train_loss_log(train_work_dir, loss_key=loss_key, tail_window=tail_window)
    status = read_json(train_status)
    metric_value = loss_summary_to_metric(summary, mode=utility_mode)
    return {
        "l2": float(metric_value),
        "obj_box_col": 0.0,
        "obj_col": 0.0,
        "loss_key": str(loss_key),
        "loss_utility_mode": str(utility_mode),
        "loss_first_mean": float(summary["first_mean"]),
        "loss_tail_mean": float(summary["tail_mean"]),
        "loss_best": float(summary["best"]),
        "loss_delta": float(summary["delta"]),
        "loss_relative_delta": float(summary["relative_delta"]),
        "loss_num_points": int(summary["num_points"]),
        "loss_iters": summary["iters"],
        "loss_values": summary["losses"],
        "train_runtime_s": float(status.get("runtime_seconds", 0.0)),
        "runtime_s": float(status.get("runtime_seconds", 0.0)),
        "peak_gpu_mb": float(status.get("peak_gpu_memory_nvidia_smi_mb") or 0.0),
    }


def run_loss_proxy_pair(
    context: dict[str, Any],
    diffusiondrive_root: str | Path,
    train_seed: int = 0,
    diffusiondrive_python: str | Path | None = None,
    wrapper_python: str | Path | None = None,
    loss_key: str = "loss",
    tail_window: int = 3,
    utility_mode: str = "drop",
) -> tuple[dict[str, Any], dict[str, Any]]:
    selector = run_train_loss_proxy(
        config=context["selector_config"],
        work_dir=context["selector_work_dir"],
        diffusiondrive_root=diffusiondrive_root,
        seed=train_seed,
        diffusiondrive_python=diffusiondrive_python,
        wrapper_python=wrapper_python,
        loss_key=loss_key,
        tail_window=tail_window,
        utility_mode=utility_mode,
    )
    random = run_train_loss_proxy(
        config=context["random_config"],
        work_dir=context["random_work_dir"],
        diffusiondrive_root=diffusiondrive_root,
        seed=train_seed,
        diffusiondrive_python=diffusiondrive_python,
        wrapper_python=wrapper_python,
        loss_key=loss_key,
        tail_window=tail_window,
        utility_mode=utility_mode,
    )
    return selector, random

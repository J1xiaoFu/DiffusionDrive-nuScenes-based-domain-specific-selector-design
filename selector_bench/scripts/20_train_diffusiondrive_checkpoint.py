#!/usr/bin/env python3
"""Run a DiffusionDrive training config with traceable outer-run metadata."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _query_gpu_memory_mb() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line.split()[0]))
        except ValueError:
            continue
    return max(values) if values else None


def _checkpoint_files(work_dir: Path) -> list[str]:
    return [str(path) for path in sorted(work_dir.glob("*.pth"))]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Train DiffusionDrive through tools/train.py.")
    parser.add_argument(
        "--diffusiondrive-root",
        type=Path,
        default=root / "DiffusionDrive",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=root / ".conda" / "diffusiondrive_py39" / "bin" / "python",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="Extra cfg-options passed through to train.py.",
    )
    parser.add_argument("--outer-log", type=Path)
    parser.add_argument("--status-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    outer_log = (args.outer_log or work_dir / "outer_train.log").resolve()
    status_json = (args.status_json or work_dir / "run_status.json").resolve()

    cmd = [
        str(args.python.resolve()),
        "tools/train.py",
        str(args.config.resolve()),
        "--work-dir",
        str(work_dir),
        "--seed",
        str(args.seed),
    ]
    if args.no_validate:
        cmd.append("--no-validate")
    if args.cfg_options:
        cmd.extend(["--cfg-options", *args.cfg_options])

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." + (os.pathsep + current_pythonpath if current_pythonpath else "")

    start = time.monotonic()
    peak_mem = _query_gpu_memory_mb()
    with outer_log.open("w", encoding="utf-8") as log:
        log.write("command=" + repr(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(args.diffusiondrive_root.resolve()),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            mem = _query_gpu_memory_mb()
            if mem is not None:
                peak_mem = mem if peak_mem is None else max(peak_mem, mem)
            time.sleep(max(args.poll_interval, 0.05))
        return_code = process.returncode
        mem = _query_gpu_memory_mb()
        if mem is not None:
            peak_mem = mem if peak_mem is None else max(peak_mem, mem)

    elapsed = time.monotonic() - start
    status: dict[str, Any] = {
        "command": cmd,
        "cwd": str(args.diffusiondrive_root.resolve()),
        "config": str(args.config.resolve()),
        "work_dir": str(work_dir),
        "outer_log": str(outer_log),
        "seed": int(args.seed),
        "no_validate": bool(args.no_validate),
        "exit_code": return_code,
        "runtime_seconds": round(elapsed, 3),
        "peak_gpu_memory_nvidia_smi_mb": peak_mem,
        "checkpoint_files": _checkpoint_files(work_dir),
    }
    status_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())

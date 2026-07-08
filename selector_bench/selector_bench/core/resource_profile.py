from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def disk_report(paths: list[str | Path]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for item in paths:
        path = Path(item)
        existing = path if path.exists() else _existing_parent(path)
        usage = shutil.disk_usage(existing)
        report[str(path)] = {
            "checked_path": str(existing),
            "total_gb": _gb(usage.total),
            "used_gb": _gb(usage.used),
            "free_gb": _gb(usage.free),
            "use_pct": round(100.0 * usage.used / usage.total, 2),
        }
    return report


def python_dependency_report() -> dict[str, Any]:
    modules = [
        "numpy",
        "torch",
        "yaml",
        "mmcv",
        "mmdet",
        "mmdet3d",
        "nuscenes",
        "cv2",
        "pandas",
        "pyarrow",
    ]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "modules": {name: _module_version(name) for name in modules},
    }


def gpu_report() -> dict[str, Any]:
    report: dict[str, Any] = {"nvidia_smi": None, "torch_cuda": None}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        report["nvidia_smi"] = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        report["nvidia_smi"] = {"returncode": 127, "stdout": "", "stderr": "nvidia-smi not found"}

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        report["torch_cuda"] = {
            "torch_version": torch.__version__,
            "cuda_available": cuda_ok,
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if cuda_ok else None,
        }
    except Exception as exc:
        report["torch_cuda"] = {"error": repr(exc)}
    return report


def limited_hardware_recommendations(
    workspace_root: str | Path,
    min_free_gb: float = 60.0,
) -> list[str]:
    usage = shutil.disk_usage(_existing_parent(Path(workspace_root)))
    free_gb = _gb(usage.free)
    recs: list[str] = []
    if free_gb < min_free_gb:
        recs.append(
            f"Free disk is {free_gb:.1f}GB, below reserve {min_free_gb:.1f}GB. "
            "Clean work_dirs/checkpoints before real training."
        )
    recs.append("Do not download full nuScenes trainval on this 500GB disk.")
    recs.append("Start with nuScenes mini or a curated sample list capped at 200-500 samples.")
    recs.append("Keep features as compressed npz and avoid parquet until pyarrow is installed deliberately.")
    recs.append("Use single-GPU smoke configs first: batch_size 1, fp16, 2 dataloader workers.")
    return recs


def build_preflight_report(
    workspace_root: str | Path,
    artifact_root: str | Path,
    min_free_gb: float = 60.0,
) -> dict[str, Any]:
    return {
        "disk": disk_report([workspace_root, artifact_root, "/tmp", "/"]),
        "python": python_dependency_report(),
        "gpu": gpu_report(),
        "recommendations": limited_hardware_recommendations(workspace_root, min_free_gb),
    }


def write_preflight_json(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _gb(value: int) -> float:
    return round(value / (1024**3), 2)


def _module_version(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "version": None}
    try:
        module = __import__(name)
        return {"installed": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"installed": True, "version": None, "import_error": repr(exc)}

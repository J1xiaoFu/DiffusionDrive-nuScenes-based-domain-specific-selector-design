from __future__ import annotations

from pathlib import Path
from typing import Any


def path_size_bytes(path: str | Path) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    if target.is_file():
        return target.stat().st_size
    total = 0
    for item in target.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def build_storage_budget_report(
    workspace_root: str | Path,
    raw_data: str | Path | None = None,
    feature_cache: str | Path | None = None,
    work_dirs: str | Path | None = None,
    projected_raw_gb: float = 0.0,
    projected_feature_gb: float = 0.0,
    projected_work_gb: float = 0.0,
    reserve_free_gb: float = 60.0,
) -> dict[str, Any]:
    import shutil

    workspace = Path(workspace_root)
    usage = shutil.disk_usage(workspace)
    current = {
        "raw_data_gb": _bytes_to_gb(path_size_bytes(raw_data)) if raw_data else 0.0,
        "feature_cache_gb": _bytes_to_gb(path_size_bytes(feature_cache)) if feature_cache else 0.0,
        "work_dirs_gb": _bytes_to_gb(path_size_bytes(work_dirs)) if work_dirs else 0.0,
    }
    projected = {
        "raw_data_gb": projected_raw_gb,
        "feature_cache_gb": projected_feature_gb,
        "work_dirs_gb": projected_work_gb,
    }
    projected_total_gb = sum(current.values()) + sum(projected.values())
    free_after_projection_gb = _bytes_to_gb(usage.free) - sum(projected.values())
    ok = free_after_projection_gb >= reserve_free_gb
    return {
        "workspace_root": str(workspace),
        "disk_total_gb": _bytes_to_gb(usage.total),
        "disk_free_gb": _bytes_to_gb(usage.free),
        "reserve_free_gb": reserve_free_gb,
        "current_usage": current,
        "projected_extra": projected,
        "projected_total_experiment_gb": round(projected_total_gb, 2),
        "free_after_projection_gb": round(free_after_projection_gb, 2),
        "ok": ok,
        "recommendation": _recommendation(ok, free_after_projection_gb, reserve_free_gb),
    }


def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 2)


def _recommendation(ok: bool, free_after: float, reserve: float) -> str:
    if ok:
        return "Storage projection is within the configured reserve."
    return (
        f"Projection leaves {free_after:.1f}GB free, below reserve {reserve:.1f}GB. "
        "Reduce dataset slice, delete old work_dirs, or attach more storage."
    )

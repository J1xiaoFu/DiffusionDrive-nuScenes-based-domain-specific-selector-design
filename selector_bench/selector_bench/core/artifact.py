from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selector_bench.utils.hashing import hash_file, hash_jsonable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.strip()
    except Exception:
        return "unknown"


def input_artifact_hash(paths: list[str | Path] | None) -> str:
    if not paths:
        return "none"
    entries: list[dict[str, str]] = []
    for item in paths:
        path = Path(item)
        if path.is_file():
            entries.append({"path": str(path), "sha256": hash_file(path)})
        else:
            entries.append({"path": str(path), "sha256": "unhashed"})
    return hash_jsonable(entries)


def make_artifact_metadata(
    artifact_type: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    input_paths: list[str | Path] | None = None,
    cwd: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    return {
        "artifact_type": artifact_type,
        "run_id": run_id,
        "config_hash": hash_jsonable(config),
        "git_commit": git_commit(cwd),
        "input_artifact_hash": input_artifact_hash(input_paths),
        "created_at": utc_now(),
        "metrics": metrics or {},
        "summary": summary or {},
    }

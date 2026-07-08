from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


KEY_FILES = [
    "docs/train_eval.md",
    "projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py",
]


def check_diffusiondrive_repo(repo_path: str | Path) -> dict[str, Any]:
    repo = Path(repo_path)
    report: dict[str, Any] = {
        "repo_path": str(repo),
        "exists": repo.exists(),
        "is_git_repo": (repo / ".git").exists(),
        "branch": None,
        "commit": None,
        "key_files": {},
        "recommendations": [],
    }
    if not repo.exists():
        report["recommendations"].append("Clone DiffusionDrive and checkout the nusc branch.")
        return report

    report["branch"] = _git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    report["commit"] = _git(repo, ["rev-parse", "HEAD"])
    for relative in KEY_FILES:
        path = repo / relative
        report["key_files"][relative] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }

    if report["branch"] != "nusc":
        report["recommendations"].append("Checkout the nusc branch before running nuScenes experiments.")
    missing = [name for name, info in report["key_files"].items() if not info["exists"]]
    if missing:
        report["recommendations"].append("Missing key files: " + ", ".join(missing))
    if not report["recommendations"]:
        report["recommendations"].append("DiffusionDrive repo layout looks ready for smoke integration.")
    return report


def _git(repo: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except FileNotFoundError:
        return None

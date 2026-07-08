#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.integrations.diffusiondrive.repo_check import check_diffusiondrive_repo
from selector_bench.utils.io import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", default=str(ROOT / "reports" / "diffusiondrive_repo_check.json"))
    args = parser.parse_args()

    report = check_diffusiondrive_repo(args.repo)
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote DiffusionDrive repo check: {output}")
    for item in report["recommendations"]:
        print(f"- {item}")
    ready = report["exists"] and report["is_git_repo"] and report["branch"] == "nusc"
    ready = ready and all(info["exists"] for info in report["key_files"].values())
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

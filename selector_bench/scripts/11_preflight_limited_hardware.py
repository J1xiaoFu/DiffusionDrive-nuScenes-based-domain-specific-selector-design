#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.core.resource_profile import build_preflight_report, write_preflight_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=str(ROOT.parent))
    parser.add_argument("--artifact-root", default=str(ROOT / "artifacts"))
    parser.add_argument("--min-free-gb", type=float, default=60.0)
    parser.add_argument("--output", default=str(ROOT / "reports" / "preflight_4090d_500gb.json"))
    args = parser.parse_args()

    report = build_preflight_report(
        workspace_root=args.workspace_root,
        artifact_root=args.artifact_root,
        min_free_gb=args.min_free_gb,
    )
    write_preflight_json(args.output, report)
    print(f"Wrote preflight report: {args.output}")
    print("Recommendations:")
    for item in report["recommendations"]:
        print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

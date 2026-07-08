#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.core.storage_budget import build_storage_budget_report
from selector_bench.utils.io import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=str(ROOT.parent))
    parser.add_argument("--raw-data", default=None)
    parser.add_argument("--feature-cache", default=str(ROOT / "artifacts" / "features"))
    parser.add_argument("--work-dirs", default=str(ROOT / "work_dirs"))
    parser.add_argument("--projected-raw-gb", type=float, default=0.0)
    parser.add_argument("--projected-feature-gb", type=float, default=5.0)
    parser.add_argument("--projected-work-gb", type=float, default=20.0)
    parser.add_argument("--reserve-free-gb", type=float, default=60.0)
    parser.add_argument("--output", default=str(ROOT / "reports" / "storage_budget.json"))
    args = parser.parse_args()

    report = build_storage_budget_report(
        workspace_root=args.workspace_root,
        raw_data=args.raw_data,
        feature_cache=args.feature_cache,
        work_dirs=args.work_dirs,
        projected_raw_gb=args.projected_raw_gb,
        projected_feature_gb=args.projected_feature_gb,
        projected_work_gb=args.projected_work_gb,
        reserve_free_gb=args.reserve_free_gb,
    )
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote storage budget report: {output}")
    print(report["recommendation"])
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightweight", action="store_true", help="Skip external dataset checks.")
    parser.add_argument("--diffusiondrive-path", default=None)
    parser.add_argument("--nuscenes-root", default=None)
    parser.add_argument("--feature-dir", default=None)
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    try:
        import numpy as np

        checks.append(("numpy", True, np.__version__))
    except Exception as exc:
        checks.append(("numpy", False, str(exc)))

    if args.feature_dir:
        feature_dir = Path(args.feature_dir)
        checks.append(("feature_dir", feature_dir.exists(), str(feature_dir)))
        checks.append(("manifest.jsonl", (feature_dir / "manifest.jsonl").exists(), str(feature_dir)))

    if not args.lightweight:
        dd_path = Path(args.diffusiondrive_path) if args.diffusiondrive_path else None
        nusc_root = Path(args.nuscenes_root) if args.nuscenes_root else None
        checks.append(("DiffusionDrive repo", bool(dd_path and dd_path.exists()), str(dd_path)))
        checks.append(("nuScenes root", bool(nusc_root and nusc_root.exists()), str(nusc_root)))

    print("Environment check")
    for name, ok, detail in checks:
        status = "OK" if ok else "MISSING"
        print(f"- {name}: {status} ({detail})")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print("Failed checks: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

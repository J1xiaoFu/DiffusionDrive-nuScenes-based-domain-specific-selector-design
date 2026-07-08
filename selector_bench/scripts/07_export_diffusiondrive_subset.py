#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from selector_bench.utils.io import ensure_dir, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    args = parser.parse_args()

    artifact = read_json(args.selection)
    selection = artifact["selection"]
    tokens = selection["selected_tokens"]
    output = Path(args.output)
    ensure_dir(output.parent)
    if args.format == "txt":
        output.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    else:
        output.write_text(
            json.dumps(
                {
                    "source_selection": args.selection,
                    "selected_count": len(tokens),
                    "selection_hash": selection["selection_hash"],
                    "sample_tokens": tokens,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote subset token list: {output}")
    print({"selected_count": len(tokens), "selection_hash": selection["selection_hash"][:12]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

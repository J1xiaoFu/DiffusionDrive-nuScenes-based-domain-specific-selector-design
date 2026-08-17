#!/usr/bin/env python3
"""Build the post-result immutable evidence inventory consumed by global Holm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from selector_bench.continual.claim_protocol import (
    CROSSED_RECEIPT_SCHEMA,
    EVIDENCE_INVENTORY_SCHEMA,
    GLOBAL_FAMILY_SCHEMA,
    comparison_inventory_record,
    load_json,
    require_equal,
    require_within,
    resolve,
    sha256,
    validate_family_registries_and_cells,
    verify_frozen_file,
)
from selector_bench.continual.statistics import StatisticsError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--family-freeze-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite evidence inventory: {args.output}")
    return args


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    family_path = args.family_spec.resolve()
    verify_frozen_file(repository, family_path, args.family_freeze_commit)
    family = load_json(family_path, "global family")
    require_equal(family.get("schema"), GLOBAL_FAMILY_SCHEMA, "global family schema")
    contract = validate_family_registries_and_cells(
        family,
        family_path=family_path,
        repository=repository,
        freeze_commit=args.family_freeze_commit,
    )
    claim_root = resolve(
        family_path.parent, family.get("claim_receipt_root"), "claim receipt root"
    )
    require_within(claim_root, repository, "claim receipt root")
    if not claim_root.is_dir():
        raise StatisticsError("claim receipt root is missing")

    expected = contract["expected_comparisons"]
    discovered: dict[Path, dict[str, object]] = {}
    for path in claim_root.rglob("*"):
        if path.is_symlink():
            raise StatisticsError(f"claim root contains a symlink: {path}")
        if not path.is_file():
            continue
        receipt = load_json(path, "claim receipt")
        require_equal(receipt.get("schema"), CROSSED_RECEIPT_SCHEMA, "claim receipt schema")
        discovered[path.resolve()] = receipt

    expected_paths = {
        resolve(
            repository,
            item["comparison_receipt_repository_path"],
            "expected comparison receipt",
        ): comparison_id
        for comparison_id, item in expected.items()
    }
    records_by_path: dict[Path, dict[str, object]] = {}
    receipt_hashes: set[str] = set()
    raw_fingerprints: set[str] = set()
    for receipt_path, receipt in sorted(
        discovered.items(), key=lambda value: str(value[0])
    ):
        record = comparison_inventory_record(receipt, receipt_path, repository)
        receipt_digest = str(record["comparison_receipt_sha256"])
        if receipt_digest in receipt_hashes:
            raise StatisticsError("evidence inventory contains byte-identical receipts")
        receipt_hashes.add(receipt_digest)
        raw_payload = {side: record[side] for side in ("baseline", "candidate")}
        raw_digest = sha256_bytes(raw_payload)
        if raw_digest in raw_fingerprints:
            raise StatisticsError("evidence inventory reuses the same raw evidence bundle")
        raw_fingerprints.add(raw_digest)
        record["raw_evidence_fingerprint"] = raw_digest
        records_by_path[receipt_path] = record

    if set(discovered) != set(expected_paths):
        raise StatisticsError(
            "claim receipt inventory differs from expected comparisons: "
            f"missing={sorted(str(path) for path in set(expected_paths)-set(discovered))} "
            f"extra={sorted(str(path) for path in set(discovered)-set(expected_paths))}"
        )

    comparison_records = []
    for receipt_path, comparison_id in sorted(expected_paths.items(), key=lambda value: value[1]):
        receipt = discovered[receipt_path]
        require_equal(receipt.get("comparison_id"), comparison_id, "inventory comparison ID")
        registered = expected[comparison_id]
        for field in (
            "comparison_spec_sha256",
            "checkpoint_stage_index",
            "evaluation_stage_index",
            "evaluation_domain",
            "split",
        ):
            require_equal(receipt.get(field), registered.get(field), f"inventory receipt {field}")
        require_equal(
            receipt.get("baseline", {}).get("method_id"),
            registered.get("baseline_method"),
            "inventory baseline method",
        )
        require_equal(
            receipt.get("baseline", {}).get("training_arm"),
            registered.get("baseline_training_arm"),
            "inventory baseline arm",
        )
        require_equal(
            receipt.get("candidate", {}).get("method_id"),
            registered.get("candidate_method"),
            "inventory candidate method",
        )
        require_equal(
            receipt.get("candidate", {}).get("training_arm"),
            registered.get("candidate_training_arm"),
            "inventory candidate arm",
        )
        comparison_records.append(records_by_path[receipt_path])

    payload = {
        "schema": EVIDENCE_INVENTORY_SCHEMA,
        "family_id": family["family_id"],
        "family_spec_repository_path": family_path.relative_to(repository).as_posix(),
        "family_spec_sha256": sha256(family_path),
        "family_freeze_commit": args.family_freeze_commit,
        "dataset_identity": contract["dataset_identity"],
        "method_registry_sha256": contract["method_registry_sha256"],
        "metric_registry_sha256": contract["metric_registry_sha256"],
        "domain_registry_sha256": contract["domain_registry_sha256"],
        "comparisons": comparison_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def sha256_bytes(payload: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


if __name__ == "__main__":
    main()

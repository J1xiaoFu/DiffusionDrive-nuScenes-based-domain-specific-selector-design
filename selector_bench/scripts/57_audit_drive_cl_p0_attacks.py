#!/usr/bin/env python3
"""Run all frozen P0 attack cases independently and emit an output-hashed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ATTACKS = (
    ("P0-01", "seed-reuse"),
    ("P0-02", "csv-swap"),
    ("P0-03", "arm-mismatch"),
    ("P0-04", "unknown-domain"),
    ("P0-05", "suffix-hide"),
    ("P0-06", "duplicate-receipt"),
    ("P0-07", "duplicate-raw-evidence"),
    ("P0-08", "missing-cross-cell"),
    ("P0-09", "row-integrity"),
    ("P0-10", "mutable-protocol"),
    ("P0-11", "incomplete-run"),
    ("P0-12", "hidden-extra-claim"),
    ("P0-13", "noncanonical-final-test-chronology"),
    ("P0-14", "loader-delivery-budget-mismatch"),
    ("P0-15", "non-executable-seed-design"),
    ("P0-16", "strict-raw-git-chronology"),
    ("P0-17", "active-family-test-binding"),
    ("P0-18", "shared-cell-common-baseline"),
    ("P0-19", "finite-seed-design-numerics"),
    ("P0-20", "complete-family-seed-design"),
    ("P0-21", "strict-resource-schema"),
)
REJECTION_MARKER = "DRIVE_CL_P0_REJECTION="


def node_for(identifier: str) -> str:
    suffix = {
        "P0-01": "test_p0_01_seed_reuse_is_rejected_before_claim",
        "P0-02": "test_p0_02_csv_swap_is_rejected_before_claim",
        "P0-03": "test_p0_03_arm_mismatch_is_rejected_before_claim",
        "P0-04": "test_p0_04_unknown_domain_is_rejected_before_claim",
        "P0-05": "test_p0_05_suffix_hide_is_rejected_before_claim_decision",
        "P0-06": "test_p0_06_duplicate_receipt_content_is_rejected",
        "P0-07": "test_p0_07_duplicate_raw_evidence_across_receipts_is_rejected",
        "P0-08": "test_p0_08_missing_cross_cell_is_rejected_at_family_gate",
        "P0-09": "test_p0_09_row_integrity_mutations_are_rejected",
        "P0-10": "test_p0_10_mutable_config_protocol_and_registries_are_rejected",
        "P0-11": "test_p0_11_incomplete_run_budget_is_rejected",
        "P0-12": "test_p0_12_hidden_extra_claim_object_is_rejected",
        "P0-13": "test_p0_13_noncanonical_final_test_chronology_is_rejected",
        "P0-14": "test_p0_14_loader_delivery_budget_mismatch_is_rejected",
        "P0-15": "test_p0_15_seed_design_must_be_exact_executable_replay",
        "P0-16": "test_p0_16_strict_raw_git_chronology_is_required",
        "P0-17": "test_p0_17_test_evidence_is_bound_to_active_family",
        "P0-18": "test_p0_18_shared_cell_requires_two_candidates_one_baseline",
        "P0-19": "test_p0_19_seed_design_rejects_nonfinite_and_boolean_numerics",
        "P0-20": "test_p0_20_seed_design_binds_complete_family_and_exact_inference",
        "P0-21": "test_p0_21_resource_receipts_reject_coercion_before_claim",
    }[identifier]
    return (
        "selector_bench/tests/test_drive_cl_claim_protocol.py::"
        f"DriveCLClaimProtocolTest::{suffix}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_node(
    repo: Path, python: Path, node: str, *, expect_rejections: bool
) -> dict[str, object]:
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        node,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "selector_bench")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    mutation_receipts = []
    for line in process.stdout.splitlines():
        if line.startswith(REJECTION_MARKER):
            mutation_receipts.append(json.loads(line[len(REJECTION_MARKER) :]))
    rejected_before_claim = bool(mutation_receipts) and all(
        item["mutated_cli_exit_code"] != 0
        and item["claim_output_exists_before_rejection_assertion"] is False
        for item in mutation_receipts
    )
    return {
        "command": command,
        "harness_exit_code": process.returncode,
        "output": process.stdout,
        "output_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "mutation_receipts": mutation_receipts,
        "rejected_before_claim_decision": (
            rejected_before_claim if expect_rejections else None
        ),
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    python = args.python.resolve()
    cases = []
    passed = True
    for identifier, mutation in ATTACKS:
        node = node_for(identifier)
        run = run_node(repo, python, node, expect_rejections=True)
        passed = (
            passed
            and run["harness_exit_code"] == 0
            and run["rejected_before_claim_decision"] is True
        )
        cases.append(
            {
                "attack_id": identifier,
                "mutation": mutation,
                "test_node": node,
                **run,
            }
        )
    positive_node = (
        "selector_bench/tests/test_drive_cl_claim_protocol.py::"
        "DriveCLClaimProtocolTest::"
        "test_positive_family_is_filename_and_hypothesis_order_invariant"
    )
    positive = run_node(repo, python, positive_node, expect_rejections=False)
    passed = passed and positive["harness_exit_code"] == 0
    payload = {
        "schema": "selector_bench.drive_cl_p0_attack_audit.v3",
        "status": "PASS" if passed else "FAIL",
        "attack_count": len(cases),
        "attack_cases": cases,
        "positive_family_fixture": positive,
        "test_source": "selector_bench/tests/test_drive_cl_claim_protocol.py",
        "test_source_sha256": sha256(
            repo / "selector_bench/tests/test_drive_cl_claim_protocol.py"
        ),
        "evidence_boundary": (
            "CPU adversarial claim-protocol fixtures only; no cache, official batch, "
            "training, evaluation performance, T0, or GPU work."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

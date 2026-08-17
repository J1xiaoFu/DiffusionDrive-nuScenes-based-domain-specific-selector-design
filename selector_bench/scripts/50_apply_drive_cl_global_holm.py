#!/usr/bin/env python3
"""Apply Holm once to one immutable and content-unique Drive-CL family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from selector_bench.continual.claim_protocol import (
    CROSSED_RECEIPT_SCHEMA,
    EVIDENCE_INVENTORY_SCHEMA,
    GLOBAL_FAMILY_SCHEMA,
    GLOBAL_RESULT_SCHEMA,
    comparison_inventory_record,
    git,
    load_json,
    require_equal,
    require_within,
    resolve,
    sha256,
    shared_sealed_test_access_identity,
    validate_method_arm,
    validate_family_registries_and_cells,
    validate_sealed_test_access_binding,
    frozen_file_bytes,
    validate_seed_design,
    validate_training_resources,
    verify_frozen_file,
)
from selector_bench.continual.statistics import (
    PDM_DEFAULT_METRICS,
    StatisticsError,
    complete_holm_family,
)
from selector_bench.continual.seed_design import finite_integer, finite_real


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--evidence-inventory", type=Path, required=True)
    parser.add_argument("--evidence-freeze-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.family_spec.is_symlink() or not args.family_spec.is_file():
        parser.error(f"missing or symlinked family specification: {args.family_spec}")
    if not (args.repository / ".git").exists():
        parser.error(f"repository is not a Git worktree: {args.repository}")
    return args


def semantic_fingerprint(item: dict[str, Any]) -> tuple[object, ...]:
    return tuple(
        item[field]
        for field in (
            "comparison_id",
            "checkpoint_stage_index",
            "checkpoint_stage_name",
            "evaluation_stage_index",
            "evaluation_stage_name",
            "split",
            "evaluation_domain",
            "metric",
            "direction",
            "baseline_method",
            "baseline_training_arm",
            "candidate_method",
            "candidate_training_arm",
            "sorted_token_sha256",
        )
    )


def comparison_evidence_fingerprint(receipt: dict[str, Any]) -> str:
    evidence: dict[str, Any] = {}
    for side in ("baseline", "candidate"):
        method = receipt.get(side)
        runs = method.get("runs", []) if isinstance(method, dict) else []
        evidence[side] = [
            {
                field: run.get(field)
                for field in (
                    "run_id",
                    "training_result_sha256",
                    "cell_summary_sha256",
                    "evaluator_receipt_sha256",
                    "checkpoint_sha256",
                    "csv_sha256",
                )
            }
            for run in sorted(runs, key=lambda value: int(value.get("training_seed", -1)))
            if isinstance(run, dict)
        ]
    return hashlib.sha256(
        json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_run_evidence(
    receipt: dict[str, Any],
    receipt_path: Path,
    *,
    expected_family: dict[str, Any],
) -> None:
    identities: dict[str, set[str]] = defaultdict(set)
    shared_access_identity: dict[str, Any] | None = None
    for side in ("baseline", "candidate"):
        method = receipt.get(side)
        if not isinstance(method, dict):
            raise StatisticsError(f"{receipt_path} lacks {side} method evidence")
        validate_method_arm(method.get("method_id"), method.get("training_arm"))
        method_access = method.get("sealed_test_access")
        validate_sealed_test_access_binding(
            method_access,
            expected_family=expected_family,
            evaluation_cell_receipt_sha256=str(
                receipt.get("evaluation_cell_receipt_sha256")
            ),
            protocol_content_sha256=str(receipt.get("protocol_content_sha256")),
            evaluation_domain=str(receipt.get("evaluation_domain")),
            split=str(receipt.get("split")),
        )
        projected_access = shared_sealed_test_access_identity(method_access)
        if shared_access_identity is None:
            shared_access_identity = projected_access
        else:
            require_equal(
                projected_access,
                shared_access_identity,
                "baseline/candidate sealed-test access identity",
            )
        runs = method.get("runs")
        if not isinstance(runs, list) or not runs:
            raise StatisticsError(f"{receipt_path} has no {side} runs")
        for run in runs:
            if not isinstance(run, dict):
                raise StatisticsError(f"{receipt_path} has malformed {side} run evidence")
            run_id = str(run.get("run_id", ""))
            if not run_id or run_id in identities["run_id"]:
                raise StatisticsError(f"{receipt_path} reuses or omits a run identity")
            identities["run_id"].add(run_id)
            for prefix in (
                "run_registration",
                "training_protocol",
                "training_result",
                "cell_summary",
                "evaluator_receipt",
                "checkpoint",
                "csv",
            ):
                path_value = run.get(f"{prefix}_path")
                hash_value = run.get(f"{prefix}_sha256")
                path = resolve(receipt_path.parent, path_value, f"{prefix} evidence")
                if not path.is_file():
                    raise StatisticsError(f"missing {prefix} evidence: {path}")
                require_equal(hash_value, sha256(path), f"{prefix} evidence SHA256")
                for label, identity in (
                    (f"{prefix}_path", str(path)),
                    (f"{prefix}_sha256", str(hash_value)),
                ):
                    if identity in identities[label]:
                        raise StatisticsError(
                            f"{receipt_path} reuses {label} across method/seed runs"
                        )
                    identities[label].add(identity)
                if prefix == "evaluator_receipt":
                    evaluator = load_json(path, "comparison evaluator receipt")
                    require_equal(
                        shared_sealed_test_access_identity(
                            evaluator.get("sealed_test_access")
                        ),
                        shared_sealed_test_access_identity(method_access),
                        "comparison/evaluator sealed-test access identity",
                    )
                if prefix == "training_result":
                    training_result = load_json(path, "comparison training result")
                    actual_resources = validate_training_resources(
                        training_result.get("resources")
                    )
                    require_equal(
                        run.get("resources"),
                        actual_resources,
                        "comparison/training-result resources",
                    )


def discover_claim_receipts(claim_root: Path) -> set[Path]:
    observed: set[Path] = set()
    for path in claim_root.rglob("*"):
        if path.is_symlink():
            raise StatisticsError(f"claim receipt root contains a symlink: {path}")
        if not path.is_file():
            continue
        receipt = load_json(path, "claim-root JSON receipt")
        if (
            receipt.get("schema") != CROSSED_RECEIPT_SCHEMA
            or receipt.get("analysis_scope") != "confirmatory"
            or receipt.get("claim_eligibility") != "confirmatory_pending_global_holm"
        ):
            raise StatisticsError(
                f"claim receipt root contains an unregistered/non-claim object: {path}"
            )
        observed.add(path.resolve())
    return observed


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    family_path = args.family_spec.resolve()
    frozen_relative_path = verify_frozen_file(
        repository, family_path, args.freeze_commit
    )
    family = load_json(family_path, "global Holm family")
    require_equal(family.get("schema"), GLOBAL_FAMILY_SCHEMA, "global family schema")
    family_id = family.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise StatisticsError("family_id must be non-empty")
    require_equal(
        tuple(family.get("metric_family", [])), PDM_DEFAULT_METRICS, "global metric family"
    )
    expected_seeds = tuple(
        finite_integer(value, "family expected seed ID")
        for value in family.get("expected_seed_ids", [])
    )
    if len(expected_seeds) < 2 or len(expected_seeds) != len(set(expected_seeds)):
        raise StatisticsError("global family needs at least two unique preregistered seeds")
    alpha = finite_real(family.get("alpha"), "family-wise alpha")
    if not 0.0 < alpha < 1.0:
        raise StatisticsError("family-wise alpha must lie strictly between zero and one")
    family_hash = sha256(family_path)
    family_contract = validate_family_registries_and_cells(
        family,
        family_path=family_path,
        repository=repository,
        freeze_commit=args.freeze_commit,
    )
    access_ledger_path = resolve(
        repository,
        family.get("sealed_test_access_ledger_repository_path"),
        "family sealed-test access ledger",
    )
    access_ledger_repository_path, frozen_access_ledger = frozen_file_bytes(
        repository, access_ledger_path, args.freeze_commit
    )
    require_equal(
        family.get("sealed_test_access_ledger_empty_sha256"),
        hashlib.sha256(frozen_access_ledger).hexdigest(),
        "family empty sealed-test ledger SHA256",
    )
    expected_family = {
        "family_id": family_id,
        "family_spec_repository_path": frozen_relative_path,
        "family_spec_sha256": family_hash,
        "family_freeze_commit": args.freeze_commit,
        "access_ledger_repository_path": access_ledger_repository_path,
        "sealed_test_access_ledger_empty_sha256": hashlib.sha256(
            frozen_access_ledger
        ).hexdigest(),
    }
    evidence_path = args.evidence_inventory.resolve()
    verify_frozen_file(
        repository, evidence_path, args.evidence_freeze_commit
    )
    git(
        repository,
        "merge-base",
        "--is-ancestor",
        args.freeze_commit,
        args.evidence_freeze_commit,
    )
    evidence_inventory = load_json(evidence_path, "frozen evidence inventory")
    require_equal(
        evidence_inventory.get("schema"),
        EVIDENCE_INVENTORY_SCHEMA,
        "evidence inventory schema",
    )
    evidence_bindings = {
        "family_id": family.get("family_id"),
        "family_spec_repository_path": frozen_relative_path,
        "family_spec_sha256": family_hash,
        "family_freeze_commit": args.freeze_commit,
        "dataset_identity": family_contract["dataset_identity"],
        "method_registry_sha256": family_contract["method_registry_sha256"],
        "metric_registry_sha256": family_contract["metric_registry_sha256"],
        "domain_registry_sha256": family_contract["domain_registry_sha256"],
    }
    for field, expected in evidence_bindings.items():
        require_equal(evidence_inventory.get(field), expected, f"evidence inventory {field}")
    seed_design = validate_seed_design(
        family,
        family_path=family_path,
        repository=repository,
        freeze_commit=args.freeze_commit,
        expected_seeds=expected_seeds,
    )
    expected_evidence_scope = (
        "paper_confirmatory"
        if seed_design["paper_claim_eligible"]
        else "synthetic_cpu_fixture_only_no_paper_claim"
    )
    inference_contracts = seed_design.get("comparison_inference_contracts")
    if not isinstance(inference_contracts, list):
        raise StatisticsError("seed design lacks comparison inference contracts")
    inference_by_comparison = {
        item.get("comparison_id"): item
        for item in inference_contracts
        if isinstance(item, dict)
    }
    if len(inference_by_comparison) != len(inference_contracts):
        raise StatisticsError("seed design has malformed comparison inference contracts")
    claim_root = resolve(
        family_path.parent, family.get("claim_receipt_root"), "claim receipt root"
    )
    require_within(claim_root, repository, "claim receipt root")
    if not claim_root.is_dir():
        raise StatisticsError(f"claim receipt root is missing: {claim_root}")
    hypotheses = family.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise StatisticsError("global Holm family must contain hypotheses")

    required_fields = {
        "id",
        "comparison_id",
        "comparison_spec_repository_path",
        "comparison_spec_sha256",
        "comparison_receipt_repository_path",
        "checkpoint_stage_index",
        "checkpoint_stage_name",
        "evaluation_stage_index",
        "evaluation_stage_name",
        "split",
        "evaluation_domain",
        "metric",
        "direction",
        "baseline_method",
        "baseline_training_arm",
        "candidate_method",
        "candidate_training_arm",
        "sorted_token_sha256",
    }
    ids: list[str] = []
    raw: dict[str, float] = {}
    sources: dict[str, dict[str, Any]] = {}
    semantic_coordinates: set[tuple[object, ...]] = set()
    declared_receipts: set[Path] = set()
    declared_metrics: dict[Path, set[str]] = defaultdict(set)
    receipt_cache: dict[Path, dict[str, Any]] = {}
    receipt_hashes: dict[str, Path] = {}
    spec_hashes: dict[str, Path] = {}
    evidence_fingerprints: dict[str, Path] = {}

    for item in hypotheses:
        if not isinstance(item, dict):
            raise StatisticsError("every hypothesis must be an object")
        missing = sorted(required_fields - set(item))
        extra = sorted(set(item) - required_fields)
        if missing or extra:
            raise StatisticsError(
                f"hypothesis semantic fields differ: missing={missing} extra={extra}"
            )
        hypothesis_id = str(item["id"])
        if not hypothesis_id or hypothesis_id in raw:
            raise StatisticsError("hypothesis IDs must be non-empty and unique")
        coordinate = semantic_fingerprint(item)
        if coordinate in semantic_coordinates:
            raise StatisticsError("duplicate semantic hypothesis coordinate")
        semantic_coordinates.add(coordinate)
        metric = str(item["metric"])
        if metric not in PDM_DEFAULT_METRICS:
            raise StatisticsError(f"unregistered hypothesis metric: {metric}")
        require_equal(
            item["direction"],
            "two_sided_candidate_minus_baseline",
            "hypothesis direction",
        )
        validate_method_arm(item["baseline_method"], item["baseline_training_arm"])
        validate_method_arm(item["candidate_method"], item["candidate_training_arm"])
        expected_comparison = family_contract["expected_comparisons"].get(
            item["comparison_id"]
        )
        if expected_comparison is None:
            raise StatisticsError("hypothesis references an unregistered comparison")
        for field in (
            "comparison_spec_repository_path",
            "comparison_spec_sha256",
            "comparison_receipt_repository_path",
            "checkpoint_stage_index",
            "evaluation_stage_index",
            "evaluation_domain",
            "split",
            "baseline_method",
            "baseline_training_arm",
            "candidate_method",
            "candidate_training_arm",
        ):
            require_equal(item.get(field), expected_comparison.get(field), f"hypothesis {field}")

        receipt_path = resolve(
            repository,
            item["comparison_receipt_repository_path"],
            "comparison receipt",
        )
        require_within(receipt_path, claim_root, "comparison receipt")
        receipt = receipt_cache.setdefault(
            receipt_path, load_json(receipt_path, "crossed comparison receipt")
        )
        if receipt_path not in declared_receipts:
            receipt_relative = receipt_path.relative_to(repository).as_posix()
            frozen_receipt = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "cat-file",
                    "-e",
                    f"{args.freeze_commit}:{receipt_relative}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if frozen_receipt.returncode == 0:
                raise StatisticsError(
                    "comparison receipt already existed at the preregistration freeze: "
                    f"{receipt_path}"
                )
            receipt_hash = sha256(receipt_path)
            duplicate_receipt = receipt_hashes.get(receipt_hash)
            if duplicate_receipt is not None:
                raise StatisticsError(
                    f"byte-identical comparison receipts: {duplicate_receipt} and {receipt_path}"
                )
            receipt_hashes[receipt_hash] = receipt_path
            evidence_hash = comparison_evidence_fingerprint(receipt)
            duplicate_evidence = evidence_fingerprints.get(evidence_hash)
            if duplicate_evidence is not None:
                raise StatisticsError(
                    f"duplicate underlying comparison evidence: {duplicate_evidence} and {receipt_path}"
                )
            evidence_fingerprints[evidence_hash] = receipt_path
        require_equal(receipt.get("schema"), CROSSED_RECEIPT_SCHEMA, "comparison receipt schema")
        registration = receipt.get("confirmatory_registration")
        if not isinstance(registration, dict):
            raise StatisticsError("comparison receipt lacks confirmatory registration")
        semantic_checks = {
            "comparison_id": item["comparison_id"],
            "analysis_scope": "confirmatory",
            "claim_eligibility": "confirmatory_pending_global_holm",
            "evidence_scope": expected_evidence_scope,
            "checkpoint_stage_index": item["checkpoint_stage_index"],
            "checkpoint_stage_name": item["checkpoint_stage_name"],
            "evaluation_stage_index": item["evaluation_stage_index"],
            "evaluation_stage_name": item["evaluation_stage_name"],
            "split": item["split"],
            "evaluation_domain": item["evaluation_domain"],
            "sorted_token_sha256": item["sorted_token_sha256"],
            "required_metrics": list(PDM_DEFAULT_METRICS),
            "expected_seed_ids": list(expected_seeds),
            "dataset_identity": family_contract["dataset_identity"],
            "method_registry_sha256": family_contract["method_registry_sha256"],
            "metric_registry_sha256": family_contract["metric_registry_sha256"],
            "domain_registry_sha256": family_contract["domain_registry_sha256"],
        }
        inference_contract = inference_by_comparison.get(item["comparison_id"])
        if not isinstance(inference_contract, dict):
            raise StatisticsError("comparison lacks a seed-design inference contract")
        semantic_checks.update(
            {
                "raw_pvalue_method": inference_contract["raw_pvalue_method"],
                "resampling_convention": inference_contract[
                    "resampling_convention"
                ],
                "metric_seed_convention": inference_contract[
                    "metric_seed_convention"
                ],
                "bootstrap_repetitions": inference_contract[
                    "bootstrap_repetitions"
                ],
                "bootstrap_seed": inference_contract["bootstrap_seed"],
            }
        )
        for field, expected in semantic_checks.items():
            require_equal(receipt.get(field), expected, f"receipt {field}")
        require_equal(receipt["baseline"].get("method_id"), item["baseline_method"], "baseline method")
        require_equal(receipt["baseline"].get("training_arm"), item["baseline_training_arm"], "baseline arm")
        require_equal(receipt["candidate"].get("method_id"), item["candidate_method"], "candidate method")
        require_equal(receipt["candidate"].get("training_arm"), item["candidate_training_arm"], "candidate arm")
        require_equal(registration.get("family_id"), family_id, "registered family ID")
        require_equal(registration.get("family_spec_sha256"), family_hash, "registered family SHA256")
        require_equal(registration.get("family_freeze_commit"), args.freeze_commit, "registered family commit")
        require_equal(
            registration.get("sealed_test_family_identity"),
            expected_family,
            "registered sealed-test family identity",
        )
        if receipt_path not in declared_receipts:
            verify_run_evidence(
                receipt,
                receipt_path,
                expected_family=expected_family,
            )

        spec_path = resolve(
            repository,
            item["comparison_spec_repository_path"],
            "comparison specification",
        )
        verify_frozen_file(repository, spec_path, args.freeze_commit)
        spec_hash = sha256(spec_path)
        require_equal(item["comparison_spec_sha256"], spec_hash, "hypothesis comparison spec SHA256")
        require_equal(receipt.get("comparison_spec_sha256"), spec_hash, "receipt comparison spec SHA256")
        require_equal(
            Path(receipt.get("comparison_spec", "")).resolve(),
            spec_path,
            "receipt comparison spec path",
        )
        if receipt_path not in declared_receipts:
            duplicate_spec = spec_hashes.get(spec_hash)
            if duplicate_spec is not None:
                raise StatisticsError(
                    f"one frozen comparison spec is reused by multiple receipts: {duplicate_spec} and {receipt_path}"
                )
            spec_hashes[spec_hash] = receipt_path

        result_metrics = receipt.get("candidate_minus_baseline")
        if not isinstance(result_metrics, dict) or set(result_metrics) != set(PDM_DEFAULT_METRICS):
            raise StatisticsError("crossed receipt must expose exactly seven registered metrics")
        raw_value = result_metrics.get(metric, {}).get("null_centered_studentized_pvalue")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise StatisticsError(f"missing raw p-value for {hypothesis_id}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise StatisticsError(f"invalid raw p-value for {hypothesis_id}: {value}")
        ids.append(hypothesis_id)
        raw[hypothesis_id] = value
        declared_receipts.add(receipt_path)
        if metric in declared_metrics[receipt_path]:
            raise StatisticsError("duplicate receipt/metric evidence")
        declared_metrics[receipt_path].add(metric)
        sources[hypothesis_id] = {
            "comparison_id": item["comparison_id"],
            "comparison_receipt": str(receipt_path),
            "comparison_receipt_sha256": sha256(receipt_path),
            "comparison_evidence_fingerprint": comparison_evidence_fingerprint(receipt),
            "metric": metric,
            "direction": item["direction"],
        }

    observed_receipts = discover_claim_receipts(claim_root)
    if observed_receipts != declared_receipts:
        raise StatisticsError(
            "claim receipt inventory must exactly equal the frozen family: "
            f"missing={sorted(str(value) for value in declared_receipts-observed_receipts)} "
            f"extra={sorted(str(value) for value in observed_receipts-declared_receipts)}"
        )
    for receipt_path in declared_receipts:
        require_equal(
            declared_metrics[receipt_path],
            set(PDM_DEFAULT_METRICS),
            f"registered p-values for {receipt_path}",
        )

    inventory_comparisons = evidence_inventory.get("comparisons")
    if not isinstance(inventory_comparisons, list) or not inventory_comparisons:
        raise StatisticsError("evidence inventory has no comparison records")
    expected_inventory_records = {
        str(item.get("comparison_id")): item
        for item in inventory_comparisons
        if isinstance(item, dict)
    }
    if len(expected_inventory_records) != len(inventory_comparisons):
        raise StatisticsError("evidence inventory has duplicate/malformed comparison records")
    if set(expected_inventory_records) != set(
        family_contract["expected_comparisons"]
    ):
        raise StatisticsError("evidence inventory comparison IDs are incomplete")
    for receipt_path in declared_receipts:
        receipt = receipt_cache[receipt_path]
        comparison_id = str(receipt.get("comparison_id"))
        actual_record = comparison_inventory_record(receipt, receipt_path, repository)
        registered_record = expected_inventory_records[comparison_id]
        raw_fingerprint = registered_record.get("raw_evidence_fingerprint")
        comparison_without_fingerprint = dict(registered_record)
        comparison_without_fingerprint.pop("raw_evidence_fingerprint", None)
        require_equal(
            comparison_without_fingerprint,
            actual_record,
            f"evidence inventory comparison {comparison_id}",
        )
        require_equal(
            raw_fingerprint,
            hashlib.sha256(
                json.dumps(
                    {side: actual_record[side] for side in ("baseline", "candidate")},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            f"evidence inventory raw fingerprint {comparison_id}",
        )

    results = complete_holm_family(raw, expected_hypothesis_ids=ids, alpha=alpha)
    payload = {
        "schema": GLOBAL_RESULT_SCHEMA,
        "family_id": family_id,
        "family_spec": str(family_path),
        "family_spec_sha256": family_hash,
        "freeze_repository": str(repository),
        "freeze_commit": args.freeze_commit,
        "evidence_inventory": str(evidence_path),
        "evidence_inventory_sha256": sha256(evidence_path),
        "evidence_freeze_commit": args.evidence_freeze_commit,
        "frozen_spec_repository_path": frozen_relative_path,
        "alpha": alpha,
        "hypothesis_count": len(ids),
        "comparison_receipt_count": len(declared_receipts),
        "hypothesis_ids_in_preregistered_order": ids,
        "metric_family": list(PDM_DEFAULT_METRICS),
        "expected_seed_ids": list(expected_seeds),
        "seed_design": seed_design,
        "evidence_scope": expected_evidence_scope,
        "claim_eligibility": (
            "paper_confirmatory_global_holm"
            if seed_design["paper_claim_eligible"]
            else "synthetic_cpu_fixture_only_no_paper_claim"
        ),
        "dataset_identity": family_contract["dataset_identity"],
        "method_registry_sha256": family_contract["method_registry_sha256"],
        "metric_registry_sha256": family_contract["metric_registry_sha256"],
        "domain_registry_sha256": family_contract["domain_registry_sha256"],
        "results": results,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

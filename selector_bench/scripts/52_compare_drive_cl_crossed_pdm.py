#!/usr/bin/env python3
"""Produce a preregistered, identity-bound seed-by-session comparison receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from selector_bench.continual.claim_protocol import (
    CELL_SUMMARY_SCHEMA,
    CROSSED_RECEIPT_SCHEMA,
    CROSSED_SPEC_SCHEMA,
    GLOBAL_FAMILY_SCHEMA,
    frozen_file_bytes,
    load_evaluator_run_contract,
    load_json,
    load_training_run_contract,
    require_equal,
    require_within,
    resolve,
    sha256,
    shared_sealed_test_access_identity,
    validate_method_arm,
    validate_family_registries_and_cells,
    validate_seed_design,
    validate_training_resources,
    verify_frozen_file,
)
from selector_bench.continual.evaluation_contract import load_evaluation_cell_contract
from selector_bench.continual.navsim_protocol import session_id_from_log
from selector_bench.continual.statistics import (
    PDM_DEFAULT_METRICS,
    StatisticsError,
    crossed_seed_session_bootstrap,
    read_pdm_rows,
)
from selector_bench.continual.seed_design import (
    RAW_PVALUE_METHOD,
    RESAMPLING_CONVENTION,
    finite_integer,
)


def token_sha256(tokens: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(tokens)) + "\n").encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-repository", type=Path)
    parser.add_argument("--family-spec", type=Path)
    parser.add_argument("--family-freeze-commit")
    args = parser.parse_args()
    if args.comparison_spec.is_symlink() or not args.comparison_spec.is_file():
        parser.error(f"missing or symlinked comparison specification: {args.comparison_spec}")
    return args


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise StatisticsError(f"{label} fields differ: missing={missing} extra={extra}")


def verify_confirmatory_registration(
    args: argparse.Namespace,
    *,
    spec: dict[str, Any],
    comparison_id: str,
    expected_seeds: tuple[int, ...],
    checkpoint_stage_index: int,
    checkpoint_stage_name: str,
    evaluation_stage_index: int,
    evaluation_stage_name: str,
    evaluation_domain: str,
    split: str,
    sorted_token_sha256: str,
    baseline_method: str,
    baseline_arm: str,
    candidate_method: str,
    candidate_arm: str,
    protocol_path: Path,
) -> dict[str, Any]:
    supplied = (args.family_repository, args.family_spec, args.family_freeze_commit)
    if any(value is None for value in supplied):
        raise StatisticsError(
            "confirmatory comparison requires family repository, spec and full freeze commit"
        )
    repository = args.family_repository.resolve()
    family_path = args.family_spec.resolve()
    freeze_commit = str(args.family_freeze_commit)
    family_repository_path = verify_frozen_file(repository, family_path, freeze_commit)
    comparison_repository_path = verify_frozen_file(
        repository, args.comparison_spec.resolve(), freeze_commit
    )
    family = load_json(family_path, "global Holm family")
    require_equal(family.get("schema"), GLOBAL_FAMILY_SCHEMA, "global family schema")
    require_equal(
        tuple(family.get("metric_family", [])), PDM_DEFAULT_METRICS, "family metric set"
    )
    require_equal(
        tuple(
            finite_integer(value, "family expected seed ID")
            for value in family.get("expected_seed_ids", [])
        ),
        expected_seeds,
        "family seed IDs",
    )
    family_id = family.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise StatisticsError("family_id must be non-empty")
    family_hash = sha256(family_path)
    family_contract = validate_family_registries_and_cells(
        family,
        family_path=family_path,
        repository=repository,
        freeze_commit=freeze_commit,
        expected_protocol_path=protocol_path,
    )
    access_ledger_path = resolve(
        repository,
        family.get("sealed_test_access_ledger_repository_path"),
        "family sealed-test access ledger",
    )
    access_ledger_repository_path, frozen_access_ledger = frozen_file_bytes(
        repository, access_ledger_path, freeze_commit
    )
    require_equal(
        family.get("sealed_test_access_ledger_empty_sha256"),
        hashlib.sha256(frozen_access_ledger).hexdigest(),
        "family empty sealed-test ledger SHA256",
    )
    design = validate_seed_design(
        family,
        family_path=family_path,
        repository=repository,
        freeze_commit=freeze_commit,
        expected_seeds=expected_seeds,
    )
    claim_root = resolve(
        family_path.parent, family.get("claim_receipt_root"), "claim receipt root"
    )
    require_within(claim_root, repository, "claim receipt root")
    output = args.output.absolute().resolve()
    require_within(output, claim_root, "confirmatory comparison output")
    if output.exists():
        raise StatisticsError("confirmatory comparison receipt is append-only and already exists")
    output_relative = output.relative_to(repository).as_posix()
    existed_at_freeze = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{freeze_commit}:{output_relative}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if existed_at_freeze.returncode == 0:
        raise StatisticsError("confirmatory result already existed in the family freeze commit")
    expected_comparison = family_contract["expected_comparisons"].get(comparison_id)
    if expected_comparison is None:
        raise StatisticsError("comparison is absent from the frozen expected-comparison inventory")
    expected_comparison_bindings = {
        "comparison_spec_repository_path": comparison_repository_path,
        "comparison_spec_sha256": sha256(args.comparison_spec),
        "comparison_receipt_repository_path": output_relative,
        "checkpoint_stage_index": checkpoint_stage_index,
        "evaluation_stage_index": evaluation_stage_index,
        "evaluation_domain": evaluation_domain,
        "split": split,
        "baseline_method": baseline_method,
        "baseline_training_arm": baseline_arm,
        "candidate_method": candidate_method,
        "candidate_training_arm": candidate_arm,
    }
    for field, expected in expected_comparison_bindings.items():
        require_equal(expected_comparison.get(field), expected, f"expected comparison {field}")

    hypotheses = family.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise StatisticsError("global family hypotheses must be a list")
    registered = [
        item
        for item in hypotheses
        if isinstance(item, dict) and item.get("comparison_id") == comparison_id
    ]
    if len(registered) != len(PDM_DEFAULT_METRICS):
        raise StatisticsError(
            "confirmatory comparison must be preregistered for exactly seven metrics"
        )
    expected_semantics = {
        "comparison_spec_repository_path": comparison_repository_path,
        "comparison_spec_sha256": sha256(args.comparison_spec),
        "comparison_receipt_repository_path": output_relative,
        "checkpoint_stage_index": checkpoint_stage_index,
        "checkpoint_stage_name": checkpoint_stage_name,
        "evaluation_stage_index": evaluation_stage_index,
        "evaluation_stage_name": evaluation_stage_name,
        "split": split,
        "evaluation_domain": evaluation_domain,
        "baseline_method": baseline_method,
        "baseline_training_arm": baseline_arm,
        "candidate_method": candidate_method,
        "candidate_training_arm": candidate_arm,
        "sorted_token_sha256": sorted_token_sha256,
        "direction": "two_sided_candidate_minus_baseline",
    }
    metrics: list[str] = []
    for item in registered:
        for field, expected in expected_semantics.items():
            require_equal(item.get(field), expected, f"preregistered {field}")
        metrics.append(str(item.get("metric")))
    if len(metrics) != len(PDM_DEFAULT_METRICS) or set(metrics) != set(PDM_DEFAULT_METRICS):
        raise StatisticsError("preregistered comparison must contain each metric exactly once")
    return {
        "family_id": family_id,
        "family_spec": str(family_path),
        "family_spec_sha256": family_hash,
        "family_repository": str(repository),
        "family_repository_path": family_repository_path,
        "family_freeze_commit": freeze_commit,
        "dataset_identity": family_contract["dataset_identity"],
        "method_registry_sha256": family_contract["method_registry_sha256"],
        "metric_registry_sha256": family_contract["metric_registry_sha256"],
        "domain_registry_sha256": family_contract["domain_registry_sha256"],
        "sealed_test_family_identity": {
            "family_id": family_id,
            "family_spec_repository_path": family_repository_path,
            "family_spec_sha256": family_hash,
            "family_freeze_commit": freeze_commit,
            "access_ledger_repository_path": access_ledger_repository_path,
            "sealed_test_access_ledger_empty_sha256": hashlib.sha256(
                frozen_access_ledger
            ).hexdigest(),
        },
        **design,
    }


def validate_method(
    method: dict[str, Any],
    *,
    artifact_root: Path,
    expected_seeds: tuple[int, ...],
    protocol_path: Path,
    protocol_sha256: str,
    protocol_content_sha256: str,
    checkpoint_stage_index: int,
    evaluation_contract: Any,
    expected_family: dict[str, Any] | None,
) -> tuple[
    str,
    str,
    dict[int, dict[str, dict[str, float]]],
    dict[str, Any],
    dict[str, set[str]],
    dict[int, str | None],
]:
    exact_keys(method, {"method_id", "training_arm", "runs"}, "method")
    method_id, training_arm = validate_method_arm(
        method.get("method_id"), method.get("training_arm")
    )
    runs = method.get("runs")
    if not isinstance(runs, list) or not runs:
        raise StatisticsError(f"method {method_id} must contain runs")
    by_seed: dict[int, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise StatisticsError(f"method {method_id} run must be an object")
        exact_keys(
            run,
            {
                "training_seed",
                "training_protocol",
                "training_result",
                "cell_summary",
                "run_registration",
                "run_registration_sha256",
                "run_registration_commit",
            },
            f"method {method_id} run",
        )
        seed = finite_integer(run.get("training_seed"), "comparison training seed")
        if seed in by_seed:
            raise StatisticsError(f"method {method_id} has duplicate seed {seed}")
        by_seed[seed] = run
    require_equal(set(by_seed), set(expected_seeds), f"method {method_id} seed set")

    expected_tokens = set(evaluation_contract.tokens)
    rows: dict[int, dict[str, dict[str, float]]] = {}
    evidence: dict[str, Any] = {
        "method_id": method_id,
        "training_arm": training_arm,
        "method_registry_sha256": None,
        "metric_registry_sha256": None,
        "domain_registry_sha256": None,
        "sealed_test_access": None,
        "runs": [],
    }
    sealed_access_initialized = False
    unique: dict[str, set[str]] = {
        label: set()
        for label in (
            "run_id",
            "run_registration_path",
            "run_registration_sha256",
            "training_protocol_path",
            "training_protocol_sha256",
            "training_result_path",
            "training_result_sha256",
            "cell_summary_path",
            "cell_summary_sha256",
            "evaluator_receipt_path",
            "evaluator_receipt_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "csv_path",
            "csv_sha256",
        )
    }
    source_by_seed: dict[int, str | None] = {}
    for seed in expected_seeds:
        run = by_seed[seed]
        training_protocol_path = resolve(
            artifact_root, run.get("training_protocol"), f"{method_id}/{seed} training protocol"
        )
        training_result_path = resolve(
            artifact_root, run.get("training_result"), f"{method_id}/{seed} training result"
        )
        cell_summary_path = resolve(
            artifact_root, run.get("cell_summary"), f"{method_id}/{seed} cell summary"
        )
        for path, label in (
            (training_protocol_path, "training protocol"),
            (training_result_path, "training result"),
            (cell_summary_path, "cell summary"),
        ):
            require_within(path, artifact_root, f"{method_id}/{seed} {label}")
        training = load_training_run_contract(training_protocol_path, training_result_path)
        registered_run_path = resolve(
            artifact_root, run.get("run_registration"), f"{method_id}/{seed} run registration"
        )
        require_equal(
            registered_run_path,
            resolve(
                training.protocol_path.parent,
                training.protocol.get("run_registration"),
                "training run registration",
            ),
            "comparison/training run-registration path",
        )
        require_equal(
            run.get("run_registration_sha256"),
            training.protocol.get("run_registration_sha256"),
            "comparison/training run-registration SHA256",
        )
        require_equal(
            run.get("run_registration_commit"),
            training.protocol.get("run_registration_commit"),
            "comparison/training run-registration commit",
        )
        require_equal(training.protocol["method_id"], method_id, "training method ID")
        require_equal(training.protocol["arm"], training_arm, "training arm")
        require_equal(training.protocol["seed"], seed, "training seed")
        require_equal(
            training.protocol["stage_index"], checkpoint_stage_index, "training checkpoint stage"
        )
        require_equal(
            training.protocol["protocol_manifest_sha256"],
            protocol_sha256,
            "training protocol manifest SHA256",
        )
        require_equal(
            training.protocol["protocol_content_sha256"],
            protocol_content_sha256,
            "training protocol content SHA256",
        )
        registered_manifest = resolve(
            training.protocol_path.parent,
            training.protocol.get("protocol_manifest"),
            "training protocol manifest",
        )
        require_equal(registered_manifest, protocol_path, "training protocol manifest path")
        source_by_seed[seed] = training.protocol.get("source_checkpoint_sha256")
        if evidence["method_registry_sha256"] is None:
            evidence["method_registry_sha256"] = training.protocol["method_registry_sha256"]
        else:
            require_equal(
                training.protocol["method_registry_sha256"],
                evidence["method_registry_sha256"],
                "method registry across seeds",
            )

        summary = load_json(cell_summary_path, "cell summary")
        require_equal(summary.get("schema"), CELL_SUMMARY_SCHEMA, "cell summary schema")
        summary_bindings = {
            "run_id": training.protocol["run_id"],
            "training_arm": training_arm,
            "training_seed": seed,
            "checkpoint_stage_index": checkpoint_stage_index,
            "training_protocol_sha256": training.protocol_sha256,
            "training_result_sha256": training.result_sha256,
            "protocol_content_sha256": protocol_content_sha256,
            "dataset_identity": evaluation_contract.protocol["dataset_identity"],
            "stage_index": evaluation_contract.receipt["stage_index"],
            "stage_name": evaluation_contract.receipt["stage_name"],
            "evaluation_domain": evaluation_contract.evaluation_domain,
            "split": evaluation_contract.receipt["split"],
            "checkpoint_sha256": training.endpoint_sha256,
            "expected_tokens": len(expected_tokens),
            "valid_tokens": len(expected_tokens),
            "invalid_tokens": 0,
            "csv_row_count": len(expected_tokens),
            "required_metrics": list(PDM_DEFAULT_METRICS),
            "metric_contract": "frozen_claim_bearing_seven_metric_family",
            "claim_eligibility": "evaluation_cell_only_no_comparative_claim",
        }
        for field, expected in summary_bindings.items():
            require_equal(summary.get(field), expected, f"cell summary {field}")
        summary_protocol = resolve(
            cell_summary_path.parent, summary.get("training_protocol"), "summary training protocol"
        )
        summary_result = resolve(
            cell_summary_path.parent, summary.get("training_result"), "summary training result"
        )
        summary_checkpoint = resolve(
            cell_summary_path.parent, summary.get("checkpoint"), "summary checkpoint"
        )
        require_equal(summary_protocol, training.protocol_path, "summary protocol path")
        require_equal(summary_result, training.result_path, "summary result path")
        require_equal(summary_checkpoint, training.endpoint_path, "summary checkpoint path")
        evaluator_receipt_path = resolve(
            cell_summary_path.parent,
            summary.get("evaluator_run_receipt"),
            "summary evaluator receipt",
        )
        require_within(
            evaluator_receipt_path, artifact_root, f"{method_id}/{seed} evaluator receipt"
        )
        evaluator = load_evaluator_run_contract(
            evaluator_receipt_path,
            training=training,
            evaluation_cell_receipt=evaluation_contract.receipt_path,
            evaluation_cell_receipt_sha256=sha256(evaluation_contract.receipt_path),
            evaluation_stage_index=int(evaluation_contract.receipt["stage_index"]),
            evaluation_stage_name=str(evaluation_contract.receipt["stage_name"]),
            evaluation_domain=evaluation_contract.evaluation_domain,
            split=str(evaluation_contract.receipt["split"]),
            token_file_path=evaluation_contract.token_file_path,
            token_file_sha256=sha256(evaluation_contract.token_file_path),
            expected_family=expected_family,
        )
        sealed_access = evaluator.receipt.get("sealed_test_access")
        if not sealed_access_initialized:
            evidence["sealed_test_access"] = sealed_access
            sealed_access_initialized = True
        else:
            require_equal(
                shared_sealed_test_access_identity(sealed_access),
                shared_sealed_test_access_identity(evidence["sealed_test_access"]),
                "sealed-test access identity across method seeds",
            )
        for field in ("metric_registry_sha256", "domain_registry_sha256"):
            if evidence[field] is None:
                evidence[field] = evaluator.receipt[field]
            else:
                require_equal(
                    evaluator.receipt[field], evidence[field], f"{field} across seeds"
                )
        for field in (
            "metric_schema_sha256",
            "metric_registry_sha256",
            "domain_registry_sha256",
            "csv_row_count",
        ):
            require_equal(
                summary.get(field), evaluator.receipt.get(field), f"summary {field}"
            )
        for field in ("metric_registry", "domain_registry"):
            summary_registry = resolve(
                cell_summary_path.parent, summary.get(field), f"summary {field}"
            )
            evaluator_registry = resolve(
                evaluator.receipt_path.parent,
                evaluator.receipt.get(field),
                f"evaluator {field}",
            )
            require_equal(summary_registry, evaluator_registry, f"summary {field} path")
        require_equal(
            summary.get("evaluator_run_receipt_sha256"),
            evaluator.receipt_sha256,
            "summary evaluator receipt SHA256",
        )
        summary_csv = resolve(cell_summary_path.parent, summary.get("csv"), "summary CSV")
        require_equal(summary_csv, evaluator.csv_path, "summary evaluator CSV path")
        require_equal(summary.get("csv_sha256"), evaluator.csv_sha256, "summary CSV SHA256")
        rows[seed] = read_pdm_rows(
            evaluator.csv_path,
            expected_tokens=expected_tokens,
            require_all_valid=True,
            required_metrics=PDM_DEFAULT_METRICS,
        )
        identities = {
            "run_id": str(training.protocol["run_id"]),
            "run_registration_path": str(registered_run_path),
            "run_registration_sha256": str(
                training.protocol["run_registration_sha256"]
            ),
            "training_protocol_path": str(training.protocol_path),
            "training_protocol_sha256": training.protocol_sha256,
            "training_result_path": str(training.result_path),
            "training_result_sha256": training.result_sha256,
            "cell_summary_path": str(cell_summary_path),
            "cell_summary_sha256": sha256(cell_summary_path),
            "evaluator_receipt_path": str(evaluator.receipt_path),
            "evaluator_receipt_sha256": evaluator.receipt_sha256,
            "checkpoint_path": str(training.endpoint_path),
            "checkpoint_sha256": training.endpoint_sha256,
            "csv_path": str(evaluator.csv_path),
            "csv_sha256": evaluator.csv_sha256,
        }
        for label, value in identities.items():
            if value in unique[label]:
                raise StatisticsError(
                    f"method {method_id} reuses {label} across distinct seeds: {value}"
                )
            unique[label].add(value)
        source_checkpoint_path = resolve(
            training.protocol_path.parent,
            training.protocol.get("source_checkpoint"),
            "training source checkpoint",
        )
        rng_record_path = resolve(
            training.protocol_path.parent,
            training.protocol.get("rng_record"),
            "training RNG record",
        )
        optimizer_receipt_path = resolve(
            training.result_path.parent,
            training.result.get("optimizer_state_receipt"),
            "training optimizer-state receipt",
        )
        evidence["runs"].append(
            {
                "training_seed": seed,
                **identities,
                "source_checkpoint_path": str(source_checkpoint_path),
                "source_checkpoint_sha256": training.protocol["source_checkpoint_sha256"],
                "rng_record_path": str(rng_record_path),
                "rng_record_sha256": training.protocol["rng_record_sha256"],
                "optimizer_state_receipt_path": str(optimizer_receipt_path),
                "optimizer_state_receipt_sha256": training.result[
                    "optimizer_state_receipt_sha256"
                ],
                "optimizer_state_sha256": training.result["optimizer_state_sha256"],
                "resources": validate_training_resources(
                    training.result.get("resources")
                ),
                "run_registration_commit": training.protocol[
                    "run_registration_commit"
                ],
            }
        )
    return method_id, training_arm, rows, evidence, unique, source_by_seed


def main() -> None:
    args = parse_args()
    spec = load_json(args.comparison_spec.resolve(), "comparison specification")
    exact_keys(
        spec,
        {
            "schema",
            "comparison_id",
            "analysis_scope",
            "metric_family",
            "expected_seed_ids",
            "bootstrap_repetitions",
            "bootstrap_seed",
            "artifact_root",
            "protocol_manifest",
            "evaluation_cell_receipt",
            "checkpoint_stage_index",
            "baseline",
            "candidate",
        },
        "comparison specification",
    )
    require_equal(spec.get("schema"), CROSSED_SPEC_SCHEMA, "crossed specification schema")
    comparison_id = spec.get("comparison_id")
    if not isinstance(comparison_id, str) or not comparison_id:
        raise StatisticsError("comparison_id must be non-empty")
    analysis_scope = spec.get("analysis_scope")
    if analysis_scope not in {"screening_only", "confirmatory"}:
        raise StatisticsError("analysis_scope must be screening_only or confirmatory")
    require_equal(
        tuple(spec.get("metric_family", [])), PDM_DEFAULT_METRICS, "comparison metric family"
    )
    expected_seeds = tuple(
        finite_integer(value, "comparison expected seed ID")
        for value in spec.get("expected_seed_ids", [])
    )
    if len(expected_seeds) < 2 or len(expected_seeds) != len(set(expected_seeds)):
        raise StatisticsError("expected_seed_ids must contain at least two unique seeds")
    repetitions = finite_integer(
        spec.get("bootstrap_repetitions"), "comparison bootstrap repetitions"
    )
    bootstrap_seed = finite_integer(
        spec.get("bootstrap_seed"), "comparison bootstrap seed"
    )
    if repetitions <= 0:
        raise StatisticsError("bootstrap repetitions must be positive")

    spec_root = args.comparison_spec.resolve().parent
    artifact_root = resolve(spec_root, spec.get("artifact_root"), "comparison artifact root")
    if not artifact_root.is_dir():
        raise StatisticsError(f"comparison artifact root is missing: {artifact_root}")
    protocol_path = resolve(spec_root, spec.get("protocol_manifest"), "protocol manifest")
    evaluation_receipt_path = resolve(
        artifact_root, spec.get("evaluation_cell_receipt"), "evaluation cell receipt"
    )
    require_within(evaluation_receipt_path, artifact_root, "evaluation cell receipt")
    evaluation = load_evaluation_cell_contract(evaluation_receipt_path)
    require_equal(evaluation.protocol_path, protocol_path, "evaluation protocol path")
    protocol = evaluation.protocol
    protocol_hash = sha256(protocol_path)
    protocol_content_hash = str(protocol["content_sha256"])
    checkpoint_stage_index = finite_integer(
        spec.get("checkpoint_stage_index"), "comparison checkpoint stage index"
    )
    checkpoint_stage = next(
        (
            value
            for value in protocol.get("stages", [])
            if isinstance(value, dict) and value.get("stage_index") == checkpoint_stage_index
        ),
        None,
    )
    if not isinstance(checkpoint_stage, dict):
        raise StatisticsError(f"protocol has no checkpoint stage {checkpoint_stage_index}")

    baseline_spec = spec.get("baseline", {})
    candidate_spec = spec.get("candidate", {})
    baseline_declared_id, baseline_declared_arm = validate_method_arm(
        baseline_spec.get("method_id"), baseline_spec.get("training_arm")
    )
    candidate_declared_id, candidate_declared_arm = validate_method_arm(
        candidate_spec.get("method_id"), candidate_spec.get("training_arm")
    )
    expected_tokens = set(evaluation.tokens)
    sorted_tokens_hash = token_sha256(expected_tokens)
    family: dict[str, Any] | None = None
    if analysis_scope == "confirmatory":
        family = verify_confirmatory_registration(
            args,
            spec=spec,
            comparison_id=comparison_id,
            expected_seeds=expected_seeds,
            checkpoint_stage_index=checkpoint_stage_index,
            checkpoint_stage_name=str(checkpoint_stage["name"]),
            evaluation_stage_index=int(evaluation.receipt["stage_index"]),
            evaluation_stage_name=str(evaluation.receipt["stage_name"]),
            evaluation_domain=evaluation.evaluation_domain,
            split=str(evaluation.receipt["split"]),
            sorted_token_sha256=sorted_tokens_hash,
            baseline_method=baseline_declared_id,
            baseline_arm=baseline_declared_arm,
            candidate_method=candidate_declared_id,
            candidate_arm=candidate_declared_arm,
            protocol_path=protocol_path,
        )
    elif any(
        value is not None
        for value in (args.family_repository, args.family_spec, args.family_freeze_commit)
    ):
        raise StatisticsError("screening comparison may not claim a confirmatory family")
    expected_family = (
        family["sealed_test_family_identity"] if family is not None else None
    )
    if family is not None:
        inference_contracts = family.get("comparison_inference_contracts")
        if not isinstance(inference_contracts, list):
            raise StatisticsError("confirmatory family lacks comparison inference contracts")
        inference_contract = next(
            (
                item
                for item in inference_contracts
                if isinstance(item, dict)
                and item.get("comparison_id") == comparison_id
            ),
            None,
        )
        if not isinstance(inference_contract, dict):
            raise StatisticsError("comparison lacks its seed-design inference contract")
        for field, expected in (
            ("bootstrap_repetitions", repetitions),
            ("bootstrap_seed", bootstrap_seed),
            ("raw_pvalue_method", RAW_PVALUE_METHOD),
            ("resampling_convention", RESAMPLING_CONVENTION),
        ):
            require_equal(
                inference_contract.get(field),
                expected,
                f"comparison seed-design {field}",
            )

    baseline = validate_method(
        baseline_spec,
        artifact_root=artifact_root,
        expected_seeds=expected_seeds,
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        protocol_content_sha256=protocol_content_hash,
        checkpoint_stage_index=checkpoint_stage_index,
        evaluation_contract=evaluation,
        expected_family=expected_family,
    )
    candidate = validate_method(
        candidate_spec,
        artifact_root=artifact_root,
        expected_seeds=expected_seeds,
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        protocol_content_sha256=protocol_content_hash,
        checkpoint_stage_index=checkpoint_stage_index,
        evaluation_contract=evaluation,
        expected_family=expected_family,
    )
    baseline_id, baseline_arm, baseline_rows, baseline_evidence, baseline_unique, baseline_sources = baseline
    candidate_id, candidate_arm, candidate_rows, candidate_evidence, candidate_unique, candidate_sources = candidate
    if baseline_id == candidate_id:
        raise StatisticsError("baseline and candidate method IDs must differ")
    for registry_field in (
        "method_registry_sha256",
        "metric_registry_sha256",
        "domain_registry_sha256",
    ):
        require_equal(
            candidate_evidence[registry_field],
            baseline_evidence[registry_field],
            f"baseline/candidate {registry_field}",
        )
    require_equal(
        shared_sealed_test_access_identity(candidate_evidence["sealed_test_access"]),
        shared_sealed_test_access_identity(baseline_evidence["sealed_test_access"]),
        "baseline/candidate sealed-test access identity",
    )
    for seed in expected_seeds:
        require_equal(
            candidate_sources[seed], baseline_sources[seed], f"stage-start checkpoint seed {seed}"
        )
    for label in baseline_unique:
        overlap = baseline_unique[label] & candidate_unique[label]
        if overlap:
            raise StatisticsError(
                f"baseline and candidate reuse {label}: {sorted(overlap)}"
            )

    comparison = crossed_seed_session_bootstrap(
        baseline_rows,
        candidate_rows,
        evaluation.token_to_log,
        metrics=PDM_DEFAULT_METRICS,
        expected_seed_ids=expected_seeds,
        repetitions=repetitions,
        seed=bootstrap_seed,
    )
    payload = {
        "schema": CROSSED_RECEIPT_SCHEMA,
        "comparison_id": comparison_id,
        "analysis_scope": analysis_scope,
        "claim_eligibility": (
            "confirmatory_pending_global_holm"
            if analysis_scope == "confirmatory"
            else "screening_only_no_superiority_claim"
        ),
        "evidence_scope": (
            "screening_only_no_paper_claim"
            if family is None
            else (
                "paper_confirmatory"
                if family["paper_claim_eligible"]
                else "synthetic_cpu_fixture_only_no_paper_claim"
            )
        ),
        "comparison_spec": str(args.comparison_spec.resolve()),
        "comparison_spec_sha256": sha256(args.comparison_spec),
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": protocol_hash,
        "protocol_content_sha256": protocol_content_hash,
        "dataset_identity": protocol["dataset_identity"],
        "method_registry_sha256": baseline_evidence["method_registry_sha256"],
        "metric_registry_sha256": baseline_evidence["metric_registry_sha256"],
        "domain_registry_sha256": baseline_evidence["domain_registry_sha256"],
        "evaluation_cell_receipt": str(evaluation_receipt_path),
        "evaluation_cell_receipt_sha256": sha256(evaluation_receipt_path),
        "checkpoint_stage_index": checkpoint_stage_index,
        "checkpoint_stage_name": checkpoint_stage["name"],
        "evaluation_stage_index": evaluation.receipt["stage_index"],
        "evaluation_stage_name": evaluation.receipt["stage_name"],
        "split": evaluation.receipt["split"],
        "evaluation_domain": evaluation.evaluation_domain,
        "expected_token_count": len(expected_tokens),
        "sorted_token_sha256": sorted_tokens_hash,
        "session_cluster_count": len(
            {session_id_from_log(value) for value in evaluation.token_to_log.values()}
        ),
        "required_metrics": list(PDM_DEFAULT_METRICS),
        "expected_seed_ids": list(expected_seeds),
        "raw_pvalue_method": RAW_PVALUE_METHOD,
        "resampling_convention": RESAMPLING_CONVENTION,
        "metric_seed_convention": (
            "bootstrap_seed_plus_index_in_frozen_PDM_DEFAULT_METRICS"
        ),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "baseline": baseline_evidence,
        "candidate": candidate_evidence,
        "candidate_minus_baseline": comparison,
        "confirmatory_registration": family,
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

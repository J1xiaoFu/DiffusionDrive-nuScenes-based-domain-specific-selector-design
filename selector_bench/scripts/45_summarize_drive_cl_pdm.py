#!/usr/bin/env python3
"""Validate and summarize one sealed/audit Drive-CL PDM evaluation cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from selector_bench.continual.statistics import PDM_DEFAULT_METRICS, read_pdm_rows
from selector_bench.continual.claim_protocol import (
    CELL_SUMMARY_SCHEMA,
    load_evaluator_run_contract,
    load_json,
    load_training_run_contract,
    resolve,
)
from selector_bench.continual.navsim_protocol import session_id_from_log
from selector_bench.continual.evaluation_contract import load_evaluation_cell_contract


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for path in (args.evaluator_receipt,):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.bootstrap_repetitions <= 0:
        parser.error("bootstrap repetitions must be positive")
    args.metrics = PDM_DEFAULT_METRICS
    return args


def main() -> None:
    args = parse_args()
    evaluator_payload = load_json(args.evaluator_receipt, "evaluator-run receipt")
    training_protocol_path = resolve(
        args.evaluator_receipt.parent,
        evaluator_payload.get("training_protocol"),
        "evaluator training protocol",
    )
    training_result_path = resolve(
        args.evaluator_receipt.parent,
        evaluator_payload.get("training_result"),
        "evaluator training result",
    )
    evaluation_receipt_path = resolve(
        args.evaluator_receipt.parent,
        evaluator_payload.get("evaluation_cell_receipt"),
        "evaluation cell receipt",
    )
    training = load_training_run_contract(training_protocol_path, training_result_path)
    contract = load_evaluation_cell_contract(evaluation_receipt_path)
    evaluator = load_evaluator_run_contract(
        args.evaluator_receipt,
        training=training,
        evaluation_cell_receipt=contract.receipt_path,
        evaluation_cell_receipt_sha256=sha256(contract.receipt_path),
        evaluation_stage_index=int(contract.receipt["stage_index"]),
        evaluation_stage_name=str(contract.receipt["stage_name"]),
        evaluation_domain=contract.evaluation_domain,
        split=str(contract.receipt["split"]),
        token_file_path=contract.token_file_path,
        token_file_sha256=sha256(contract.token_file_path),
    )
    receipt = contract.receipt
    manifest = contract.protocol
    expected = set(contract.tokens)
    rows = read_pdm_rows(
        evaluator.csv_path,
        expected_tokens=expected,
        require_all_valid=True,
        required_metrics=args.metrics,
    )
    metrics = list(args.metrics)
    token_to_session = {
        token: session_id_from_log(log_name)
        for token, log_name in contract.token_to_log.items()
    }
    rng = random.Random(args.seed)
    summaries: dict[str, dict[str, float]] = {}
    for metric in metrics:
        session_values: dict[str, list[float]] = defaultdict(list)
        token_values = []
        for token, values in rows.items():
            value = values[metric]
            token_values.append(value)
            session_values[token_to_session[token]].append(value)
        session_means = [float(np.mean(values)) for values in session_values.values()]
        if not session_means or len(token_values) != len(expected):
            raise RuntimeError(f"strict PDM summary is incomplete for metric {metric}")
        bootstrap = [
            float(np.mean([rng.choice(session_means) for _ in session_means]))
            for _ in range(args.bootstrap_repetitions)
        ]
        summaries[metric] = {
            "token_mean": float(np.mean(token_values)),
            "session_cluster_mean": float(np.mean(session_means)),
            "ci95_low": float(np.percentile(bootstrap, 2.5)),
            "ci95_high": float(np.percentile(bootstrap, 97.5)),
            "valid_tokens": len(token_values),
            "session_clusters": len(session_means),
        }
    payload = {
        "schema": CELL_SUMMARY_SCHEMA,
        "protocol_content_sha256": manifest["content_sha256"],
        "dataset_identity": manifest["dataset_identity"],
        "run_id": training.protocol["run_id"],
        "training_arm": training.protocol["arm"],
        "training_seed": training.protocol["seed"],
        "checkpoint_stage_index": training.protocol["stage_index"],
        "training_protocol": str(training.protocol_path),
        "training_protocol_sha256": training.protocol_sha256,
        "training_result": str(training.result_path),
        "training_result_sha256": training.result_sha256,
        "evaluator_run_receipt": str(evaluator.receipt_path),
        "evaluator_run_receipt_sha256": evaluator.receipt_sha256,
        "evaluation_cell_receipt": str(contract.receipt_path),
        "evaluation_cell_receipt_sha256": sha256(contract.receipt_path),
        "stage_index": receipt["stage_index"],
        "stage_name": receipt["stage_name"],
        "evaluation_domain": contract.evaluation_domain,
        "split": receipt["split"],
        "checkpoint": str(training.endpoint_path),
        "checkpoint_sha256": training.endpoint_sha256,
        "csv": str(evaluator.csv_path),
        "csv_sha256": evaluator.csv_sha256,
        "csv_row_count": evaluator.receipt["csv_row_count"],
        "metric_schema_sha256": evaluator.receipt["metric_schema_sha256"],
        "metric_registry": evaluator.receipt["metric_registry"],
        "metric_registry_sha256": evaluator.receipt["metric_registry_sha256"],
        "domain_registry": evaluator.receipt["domain_registry"],
        "domain_registry_sha256": evaluator.receipt["domain_registry_sha256"],
        "expected_tokens": len(expected),
        "valid_tokens": len(rows),
        "invalid_tokens": 0,
        "required_metrics": metrics,
        "metric_contract": "frozen_claim_bearing_seven_metric_family",
        "claim_eligibility": "evaluation_cell_only_no_comparative_claim",
        "bootstrap_unit": "complete_timestamp_vehicle_session",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "metrics": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

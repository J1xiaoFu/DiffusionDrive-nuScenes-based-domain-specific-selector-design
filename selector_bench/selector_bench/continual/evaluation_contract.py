"""Fail-closed identity contract for one Drive-CL NAVSIM evaluation cell."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from selector_bench.continual.claim_protocol import canonical_evaluation_domain
from selector_bench.continual.navsim_protocol import session_id_from_log
from selector_bench.continual.statistics import StatisticsError
from selector_bench.utils.hashing import hash_jsonable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StatisticsError(f"{label} must be a non-empty path")
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StatisticsError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StatisticsError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise StatisticsError(f"{label} must be a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class EvaluationCellContract:
    receipt_path: Path
    receipt: dict[str, Any]
    protocol_path: Path
    protocol: dict[str, Any]
    stage: dict[str, Any]
    cell: dict[str, Any]
    tokens: frozenset[str]
    token_to_log: dict[str, str]
    token_file_path: Path
    evaluation_domain: str


def load_evaluation_cell_contract(receipt_path: str | Path) -> EvaluationCellContract:
    receipt_path = Path(receipt_path).resolve()
    receipt = load_json(receipt_path, "evaluation cell receipt")
    if receipt.get("schema") != "selector_bench.drive_cl_eval_cell.v1":
        raise StatisticsError("unsupported evaluation cell receipt schema")
    protocol_path = resolve(
        receipt_path.parent, receipt.get("protocol_manifest"), "protocol manifest"
    )
    protocol = load_json(protocol_path, "protocol manifest")
    if protocol.get("schema") != "selector_bench.drive_cl_protocol.v1":
        raise StatisticsError("unsupported protocol manifest schema")
    if sha256(protocol_path) != receipt.get("protocol_manifest_sha256"):
        raise StatisticsError("evaluation receipt protocol manifest SHA256 mismatch")
    if protocol.get("content_sha256") != receipt.get("protocol_content_sha256"):
        raise StatisticsError("evaluation receipt protocol content SHA256 mismatch")
    if protocol.get("dataset_identity") != receipt.get("dataset_identity"):
        raise StatisticsError("evaluation receipt dataset identity mismatch")
    domain_registry = resolve(
        receipt_path.parent, receipt.get("domain_registry"), "domain registry"
    )
    if domain_registry != protocol_path:
        raise StatisticsError("evaluation receipt domain registry path mismatch")
    if receipt.get("domain_registry_sha256") != sha256(protocol_path):
        raise StatisticsError("evaluation receipt domain registry SHA256 mismatch")
    canonical_protocol = dict(protocol)
    registered_content_hash = canonical_protocol.pop("content_sha256", None)
    canonical_protocol.pop("created_at_utc", None)
    if registered_content_hash != hash_jsonable(canonical_protocol):
        raise StatisticsError("protocol manifest content_sha256 is not self-consistent")
    stage_index = receipt.get("stage_index")
    if not isinstance(stage_index, int):
        raise StatisticsError("evaluation receipt stage_index must be an integer")
    stage = next(
        (
            value
            for value in protocol.get("stages", [])
            if isinstance(value, dict) and value.get("stage_index") == stage_index
        ),
        None,
    )
    if not isinstance(stage, dict):
        raise StatisticsError(f"protocol does not contain evaluation stage {stage_index}")
    if stage.get("name") != receipt.get("stage_name"):
        raise StatisticsError("evaluation receipt stage name mismatch")
    evaluation_domain = canonical_evaluation_domain(protocol, stage)
    if receipt.get("evaluation_domain") != evaluation_domain:
        raise StatisticsError("evaluation receipt domain mismatch")
    split = receipt.get("split")
    if split not in {"audit", "test"}:
        raise StatisticsError("evaluation receipt split must be audit or test")
    splits = stage.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split), dict):
        raise StatisticsError("protocol does not contain the registered evaluation split")
    cell = splits[split]
    raw_tokens = cell.get("tokens")
    token_to_log = cell.get("token_to_log")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise StatisticsError("evaluation protocol cell has no token list")
    if len(raw_tokens) != len(set(raw_tokens)) or any(
        not isinstance(value, str) or not value for value in raw_tokens
    ):
        raise StatisticsError("evaluation protocol tokens must be unique non-empty strings")
    tokens = frozenset(raw_tokens)
    if not isinstance(token_to_log, dict) or set(token_to_log) != set(tokens):
        raise StatisticsError("evaluation token-to-log mapping must exactly equal the token set")
    if any(not isinstance(value, str) or not value for value in token_to_log.values()):
        raise StatisticsError("evaluation token-to-log mapping contains an invalid log")
    logs = set(token_to_log.values())
    sessions = {session_id_from_log(value) for value in logs}
    expected_counts = {
        "token_count": len(tokens),
        "log_count": len(logs),
        "session_count": len(sessions),
    }
    for key, expected in expected_counts.items():
        if receipt.get(key) != expected:
            raise StatisticsError(
                f"evaluation receipt {key} mismatch: expected={expected} actual={receipt.get(key)}"
            )
    token_file_path = resolve(
        receipt_path.parent, receipt.get("token_file"), "evaluation token file"
    )
    if not token_file_path.is_file():
        raise StatisticsError(f"missing evaluation token file: {token_file_path}")
    if sha256(token_file_path) != receipt.get("token_file_sha256"):
        raise StatisticsError("evaluation token file SHA256 mismatch")
    token_lines = token_file_path.read_text().splitlines()
    if token_lines != sorted(tokens):
        raise StatisticsError("evaluation token file must exactly equal sorted protocol tokens")
    expected_policy = (
        "configuration_must_be_frozen_before_use"
        if split == "test"
        else "method_and_trigger_selection_allowed"
    )
    if receipt.get("final_test_policy") != expected_policy:
        raise StatisticsError("evaluation final-test policy mismatch")
    if receipt.get("deterministic_seed_env") != "PDM_DETERMINISTIC_GLOBAL_SEED":
        raise StatisticsError("evaluation deterministic seed contract mismatch")
    return EvaluationCellContract(
        receipt_path=receipt_path,
        receipt=receipt,
        protocol_path=protocol_path,
        protocol=protocol,
        stage=stage,
        cell=cell,
        tokens=tokens,
        token_to_log={str(key): str(value) for key, value in token_to_log.items()},
        token_file_path=token_file_path,
        evaluation_domain=evaluation_domain,
    )

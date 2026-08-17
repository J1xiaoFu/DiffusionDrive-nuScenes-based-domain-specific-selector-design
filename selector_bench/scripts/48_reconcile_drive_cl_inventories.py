#!/usr/bin/env python3
"""Reconcile NAVSIM allowlist frames, sampled scenes, cache, and CL splits.

The NAVSIM ``tokens`` field is a frame allowlist, whereas training consumes
valid fixed-length windows sampled at ``frame_interval``.  This audit
reimplements the small, frozen selection rule and proves the exact set
equalities required by the Drive-CL paper protocol.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from selector_bench.continual.navsim_protocol import (
    session_id_from_log,
    validate_protocol,
)


class ReconciliationError(RuntimeError):
    pass


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def is_git_commit(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_set_sha256(tokens: Iterable[str]) -> str:
    values = sorted(set(tokens))
    return sha256_bytes(("\n".join(values) + "\n").encode())


def token_to_log_sha256(mapping: Mapping[str, str]) -> str:
    rows = [f"{token}\t{mapping[token]}" for token in sorted(mapping)]
    return sha256_bytes(("\n".join(rows) + "\n").encode())


def replay_selection_rule(
    receipt: Mapping[str, Any],
    *,
    configured_interval: set[str],
    unit_stride: set[str],
) -> set[str]:
    """Execute one of the frozen, reviewable cache selection rules."""

    rule = receipt.get("selection_rule")
    if not isinstance(rule, Mapping):
        raise ReconciliationError("cache provenance selection_rule must be an object")
    rule_type = rule.get("type")
    if rule_type == "official_configured_interval":
        return set(configured_interval)
    if rule_type == "full_unit_stride_eligible":
        return set(unit_stride)
    if rule_type == "sha256_rank_prefix":
        seed = int(rule["seed"])
        count = int(rule["count"])
        if count <= 0 or count > len(unit_stride):
            raise ReconciliationError("sha256_rank_prefix count is out of range")
        return set(
            sorted(
                unit_stride,
                key=lambda token: hashlib.sha256(
                    f"{seed}\0{token}".encode("utf-8")
                ).digest(),
            )[:count]
        )
    raise ReconciliationError(f"unsupported executable selection rule: {rule_type!r}")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ReconciliationError(f"cannot write empty table: {path}")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-filter-yaml", type=Path, required=True)
    parser.add_argument("--scene-loader-source", type=Path, required=True)
    parser.add_argument("--raw-log-root", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-provenance", type=Path)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.scene_filter_yaml,
        args.scene_loader_source,
        args.raw_log_root,
        args.cache_index,
        args.protocol_manifest,
        args.portable_manifest,
    ):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if args.cache_provenance is not None and not args.cache_provenance.is_file():
        parser.error(f"missing cache provenance receipt: {args.cache_provenance}")
    return args


def protocol_inventory(
    protocol: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str], set[str], dict[str, str], dict[str, str]]:
    all_tokens: set[str] = set()
    train_tokens: set[str] = set()
    audit_tokens: set[str] = set()
    test_tokens: set[str] = set()
    token_to_log: dict[str, str] = {}
    token_to_cell: dict[str, str] = {}
    for stage in protocol["stages"]:
        for split, cell in stage["splits"].items():
            values = set(cell["tokens"])
            if all_tokens & values:
                raise ReconciliationError("protocol token leakage across stage/split cells")
            all_tokens |= values
            if split == "train":
                train_tokens |= values
            elif split == "audit":
                audit_tokens |= values
            elif split == "test":
                test_tokens |= values
            else:
                raise ReconciliationError(f"unknown split: {split}")
            for token in values:
                token_to_log[token] = cell["token_to_log"][token]
                token_to_cell[token] = f"stage{stage['stage_index']}:{split}"
    return (
        all_tokens,
        train_tokens,
        audit_tokens,
        test_tokens,
        token_to_log,
        token_to_cell,
    )


def portable_inventory(
    portable: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str], set[str]]:
    groups: dict[str, set[str]] = {
        "all": set(),
        "train": set(),
        "audit": set(),
        "test": set(),
    }
    for stage in portable["stages"]:
        for split, cell in stage["splits"].items():
            values = set(cell["tokens"])
            if groups["all"] & values:
                raise ReconciliationError("portable token leakage across cells")
            groups["all"] |= values
            groups[split] |= values
    return groups["all"], groups["train"], groups["audit"], groups["test"]


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.scene_filter_yaml.read_text())
    history = int(config["num_history_frames"])
    future = int(config["num_future_frames"])
    window = history + future
    interval = int(config["frame_interval"])
    require_route = bool(config["has_route"])
    configured_logs = list(config["log_names"])
    allowed_tokens = set(config["tokens"])
    if len(configured_logs) != len(set(configured_logs)):
        raise ReconciliationError("scene-filter log_names contains duplicates")
    if len(config["tokens"]) != len(allowed_tokens):
        raise ReconciliationError("scene-filter tokens contains duplicates")

    raw_files = {path.stem: path for path in args.raw_log_root.glob("*.pkl")}
    missing_logs = sorted(set(configured_logs) - set(raw_files))
    extra_logs = sorted(set(raw_files) - set(configured_logs))
    if missing_logs:
        raise ReconciliationError(f"{len(missing_logs)} configured raw logs are missing")

    sampled_tokens: set[str] = set()
    sampled_token_to_log: dict[str, str] = {}
    unit_stride_eligible_tokens: set[str] = set()
    allowlisted_token_to_log: dict[str, str] = {}
    seen_raw_tokens: set[str] = set()
    selected_tree_rows: list[str] = []
    per_log: list[dict[str, Any]] = []
    selected_bytes = 0
    for index, log_name in enumerate(sorted(configured_logs)):
        path = raw_files[log_name]
        raw = path.read_bytes()
        selected_bytes += len(raw)
        digest = sha256_bytes(raw)
        selected_tree_rows.append(f"{log_name}\t{digest}")
        frames = pickle.loads(raw)  # noqa: S301 - trusted local NAVSIM annotation file
        if not isinstance(frames, list):
            raise ReconciliationError(f"raw log is not a frame list: {path}")
        raw_allowed = 0
        for frame in frames:
            token = str(frame["token"])
            seen_raw_tokens.add(token)
            raw_allowed += int(token in allowed_tokens)
            if token in allowed_tokens:
                if token in allowlisted_token_to_log:
                    raise ReconciliationError(f"duplicate raw allowlist token: {token}")
                allowlisted_token_to_log[token] = log_name

        for start in range(0, len(frames)):
            frame_list = frames[start : start + window]
            if len(frame_list) < window:
                continue
            current = frame_list[history - 1]
            if current.get("is_valid", True) == False:
                continue
            if require_route and len(current["roadblock_ids"]) == 0:
                continue
            token = str(current["token"])
            if token in allowed_tokens:
                unit_stride_eligible_tokens.add(token)

        counts = defaultdict(int)
        current_tokens: set[str] = set()
        for start in range(0, len(frames), interval):
            counts["candidate_windows"] += 1
            frame_list = frames[start : start + window]
            if len(frame_list) < window:
                counts["short_window"] += 1
                continue
            current = frame_list[history - 1]
            if current.get("is_valid", True) == False:  # match frozen loader semantics
                counts["invalid_current"] += 1
                continue
            if require_route and len(current["roadblock_ids"]) == 0:
                counts["no_route"] += 1
                continue
            token = str(current["token"])
            current_tokens.add(token)
            if token not in allowed_tokens:
                counts["not_in_allowlist"] += 1
                continue
            counts["eligible"] += 1
            if token in sampled_token_to_log:
                raise ReconciliationError(f"duplicate sampled token: {token}")
            sampled_tokens.add(token)
            sampled_token_to_log[token] = log_name
        per_log.append(
            {
                "log_name": log_name,
                "session_id": session_id_from_log(log_name),
                "raw_file_bytes": len(raw),
                "raw_file_sha256": digest,
                "raw_frames": len(frames),
                "raw_allowlisted_frames": raw_allowed,
                "candidate_windows": counts["candidate_windows"],
                "short_window": counts["short_window"],
                "invalid_current": counts["invalid_current"],
                "no_route": counts["no_route"],
                "not_in_allowlist": counts["not_in_allowlist"],
                "eligible_scenes": counts["eligible"],
                "unit_stride_eligible_scenes": sum(
                    token in unit_stride_eligible_tokens
                    and value == log_name
                    for token, value in allowlisted_token_to_log.items()
                ),
                "distinct_valid_window_current_tokens": len(current_tokens),
            }
        )
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "raw_scan_progress",
                        "logs_completed": index + 1,
                        "logs_total": len(configured_logs),
                        "sampled_tokens": len(sampled_tokens),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    cache_bytes = args.cache_index.read_bytes()
    cache_mapping = pickle.loads(cache_bytes)  # noqa: S301 - trusted local cache index
    if not isinstance(cache_mapping, dict):
        raise ReconciliationError("cache index is not a mapping")
    cache_tokens = {str(token) for token in cache_mapping}
    cache_token_to_log = {
        str(token): Path(path).parent.name for token, path in cache_mapping.items()
    }

    protocol = json.loads(args.protocol_manifest.read_text())
    validate_protocol(protocol)
    (
        protocol_tokens,
        train_tokens,
        audit_tokens,
        test_tokens,
        protocol_token_to_log,
        token_to_cell,
    ) = protocol_inventory(protocol)
    portable = json.loads(args.portable_manifest.read_text())
    portable_tokens, portable_train, portable_audit, portable_test = portable_inventory(
        portable
    )

    for row in per_log:
        log_name = row["log_name"]
        values = [
            token_to_cell[token]
            for token, value in protocol_token_to_log.items()
            if value == log_name and token in token_to_cell
        ]
        cells = sorted(set(values))
        row["protocol_cell"] = cells[0] if len(cells) == 1 else "|".join(cells)
        row["cache_scenes"] = sum(
            value == log_name for value in cache_token_to_log.values()
        )
        row["protocol_scenes"] = sum(
            value == log_name for value in protocol_token_to_log.values()
        )

    session_rows: list[dict[str, Any]] = []
    grouped_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_log:
        grouped_logs[str(row["session_id"])].append(row)
    for session, rows in sorted(grouped_logs.items()):
        cells = sorted({str(row["protocol_cell"]) for row in rows})
        session_rows.append(
            {
                "session_id": session,
                "log_count": len(rows),
                "raw_frames": sum(int(row["raw_frames"]) for row in rows),
                "raw_allowlisted_frames": sum(
                    int(row["raw_allowlisted_frames"]) for row in rows
                ),
                "candidate_windows": sum(int(row["candidate_windows"]) for row in rows),
                "eligible_scenes": sum(int(row["eligible_scenes"]) for row in rows),
                "unit_stride_eligible_scenes": sum(
                    int(row["unit_stride_eligible_scenes"]) for row in rows
                ),
                "cache_scenes": sum(int(row["cache_scenes"]) for row in rows),
                "protocol_scenes": sum(int(row["protocol_scenes"]) for row in rows),
                "protocol_cell": cells[0] if len(cells) == 1 else "|".join(cells),
            }
        )

    contract = portable.get("remote_training_cache_contract", {})
    source_manifest_sha = sha256_file(args.protocol_manifest)
    scene_filter_sha = sha256_file(args.scene_filter_yaml)
    scene_loader_sha = sha256_file(args.scene_loader_source)
    raw_tree_sha = sha256_bytes(("\n".join(selected_tree_rows) + "\n").encode())
    cache_provenance = (
        json.loads(args.cache_provenance.read_text())
        if args.cache_provenance is not None
        else None
    )
    cache_provenance_errors: list[str] = []
    replayed_selection: set[str] | None = None
    if cache_provenance is not None:
        environment_lock_path = Path(
            str(cache_provenance.get("environment_lock_path", ""))
        )
        environment_lock_sha = cache_provenance.get("environment_lock_sha256")
        checks = {
            "schema": cache_provenance.get("schema")
            == "selector_bench.drive_cl_cache_provenance.v2",
            "cache_index_sha256": cache_provenance.get("cache_index_sha256")
            == sha256_bytes(cache_bytes),
            "selected_token_sha256": cache_provenance.get("selected_token_sha256")
            == token_set_sha256(cache_tokens),
            "selected_token_to_log_sha256": cache_provenance.get(
                "selected_token_to_log_sha256"
            )
            == token_to_log_sha256(cache_token_to_log),
            "input_allowlist_sha256": cache_provenance.get("input_allowlist_sha256")
            == token_set_sha256(allowed_tokens),
            "scene_filter_sha256": cache_provenance.get("scene_filter_sha256")
            == scene_filter_sha,
            "scene_loader_sha256": cache_provenance.get("scene_loader_sha256")
            == scene_loader_sha,
            "raw_selected_tree_sha256": cache_provenance.get(
                "raw_selected_tree_sha256"
            )
            == raw_tree_sha,
            "builder_repository": isinstance(
                cache_provenance.get("builder_repository"), str
            )
            and bool(cache_provenance["builder_repository"]),
            "builder_commit": is_git_commit(cache_provenance.get("builder_commit")),
            "builder_worktree_clean": cache_provenance.get("builder_worktree_clean") is True,
            "builder_command": isinstance(cache_provenance.get("builder_command"), list)
            and bool(cache_provenance["builder_command"]),
            "environment_lock_path": environment_lock_path.is_file(),
            "environment_lock_sha256_format": is_sha256(environment_lock_sha),
            "environment_lock_sha256_matches": environment_lock_path.is_file()
            and is_sha256(environment_lock_sha)
            and sha256_file(environment_lock_path) == environment_lock_sha,
            "replay_status": cache_provenance.get("replay_status") == "PASS",
        }
        cache_provenance_errors.extend(
            name for name, passed in checks.items() if not passed
        )
        try:
            replayed_selection = replay_selection_rule(
                cache_provenance,
                configured_interval=sampled_tokens,
                unit_stride=unit_stride_eligible_tokens,
            )
        except (KeyError, TypeError, ValueError, ReconciliationError) as exc:
            cache_provenance_errors.append(f"selection_rule:{exc}")
        if replayed_selection is not None and replayed_selection != cache_tokens:
            cache_provenance_errors.append("replayed_selection_does_not_equal_cache")
    cache_provenance_valid = bool(
        cache_provenance is not None and not cache_provenance_errors
    )
    invariants = {
        "configured_logs_unique_and_present": not missing_logs
        and len(configured_logs) == len(set(configured_logs)),
        "allowlist_tokens_unique": len(allowed_tokens) == len(config["tokens"]),
        "allowlist_equals_unit_stride_eligible": allowed_tokens
        == unit_stride_eligible_tokens,
        "cache_is_subset_of_allowlist": cache_tokens <= allowed_tokens,
        "cache_log_mapping_matches_raw": all(
            allowlisted_token_to_log.get(token) == log_name
            for token, log_name in cache_token_to_log.items()
        ),
        "cache_equals_full_protocol": cache_tokens == protocol_tokens,
        "cache_log_mapping_equals_protocol": cache_token_to_log
        == protocol_token_to_log,
        "cache_selection_rule_is_reproducible": sampled_tokens == cache_tokens
        or cache_provenance_valid,
        "portable_all_equals_full_protocol": portable_tokens == protocol_tokens,
        "portable_train_equals_full_train": portable_train == train_tokens,
        "portable_audit_equals_full_audit": portable_audit == audit_tokens,
        "portable_test_equals_full_test": portable_test == test_tokens,
        "train_audit_test_are_disjoint": not (
            (train_tokens & audit_tokens)
            or (train_tokens & test_tokens)
            or (audit_tokens & test_tokens)
        ),
        "portable_source_manifest_hash_matches": portable.get("source_manifest_sha256")
        == source_manifest_sha,
        "portable_content_hash_matches_full": portable.get("content_sha256")
        == protocol.get("content_sha256"),
        "portable_train_count_matches_contract": len(portable_train)
        == int(contract.get("total_token_count", -1)),
        "portable_train_hash_matches_contract": token_set_sha256(portable_train)
        == contract.get("sorted_token_sha256"),
        "session_atomicity_holds": all("|" not in row["protocol_cell"] for row in session_rows),
    }
    status = "PASS" if all(invariants.values()) else "FAIL"

    log_csv = args.output_dir / "per_log_reconciliation.csv"
    session_csv = args.output_dir / "per_session_reconciliation.csv"
    write_csv(log_csv, per_log)
    write_csv(session_csv, session_rows)
    membership_path = args.output_dir / "exact_token_membership.tsv.gz"
    with membership_path.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    [
                        "token",
                        "raw_log",
                        "unit_stride_eligible",
                        "configured_interval",
                        "cache",
                        "protocol",
                        "protocol_cell",
                    ]
                )
                for token in sorted(allowed_tokens):
                    writer.writerow(
                        [
                            token,
                            allowlisted_token_to_log[token],
                            int(token in unit_stride_eligible_tokens),
                            int(token in sampled_tokens),
                            int(token in cache_tokens),
                            int(token in protocol_tokens),
                            token_to_cell.get(token, ""),
                        ]
                    )

    source_snapshot_dir = args.output_dir / "source_snapshots"
    source_snapshot_dir.mkdir(exist_ok=True)
    scene_filter_snapshot = source_snapshot_dir / "scene_filter.yaml"
    scene_loader_snapshot = source_snapshot_dir / "scene_loader.py"
    scene_filter_snapshot.write_bytes(args.scene_filter_yaml.read_bytes())
    scene_loader_snapshot.write_bytes(args.scene_loader_source.read_bytes())
    payload = {
        "schema": "selector_bench.drive_cl_inventory_reconciliation.v1",
        "status": status,
        "scene_selection_semantics": {
            "allowlist_unit": "raw_frame_token",
            "training_unit": "valid_fixed_length_sampled_window_current_frame",
            "num_history_frames": history,
            "num_future_frames": future,
            "window_frames": window,
            "frame_interval": interval,
            "current_frame_offset": history - 1,
            "requires_route": require_route,
        },
        "counts": {
            "raw_log_files_total": len(raw_files),
            "configured_logs": len(configured_logs),
            "extra_raw_logs_excluded": len(extra_logs),
            "configured_allowlist_tokens": len(allowed_tokens),
            "allowlist_tokens_seen_in_configured_raw_logs": len(
                allowed_tokens & seen_raw_tokens
            ),
            "unit_stride_eligible_scenes": len(unit_stride_eligible_tokens),
            "configured_interval_sampled_scenes": len(sampled_tokens),
            "cache_tokens": len(cache_tokens),
            "cache_fraction_of_allowlist": len(cache_tokens) / len(allowed_tokens),
            "protocol_tokens": len(protocol_tokens),
            "protocol_train_tokens": len(train_tokens),
            "protocol_audit_tokens": len(audit_tokens),
            "protocol_test_tokens": len(test_tokens),
            "sessions": len(session_rows),
        },
        "cache_selection_diagnosis": {
            "configured_interval_matches_cache": sampled_tokens == cache_tokens,
            "cache_is_unreproduced_subset_of_allowlist": cache_tokens < allowed_tokens,
            "declared_provenance_receipt_available": cache_provenance is not None,
            "declared_provenance_receipt_valid": cache_provenance_valid,
            "declared_provenance_errors": cache_provenance_errors,
            "required_remediation": (
                "Rebuild the cache from a frozen config with a machine-readable receipt, "
                "or supply the original subset-selection receipt; then regenerate the CL manifest."
                if not (sampled_tokens == cache_tokens or cache_provenance_valid)
                else None
            ),
        },
        "token_set_sha256": {
            "allowlist": token_set_sha256(allowed_tokens),
            "unit_stride_eligible": token_set_sha256(unit_stride_eligible_tokens),
            "configured_interval_sampled": token_set_sha256(sampled_tokens),
            "cache": token_set_sha256(cache_tokens),
            "protocol_all": token_set_sha256(protocol_tokens),
            "protocol_train": token_set_sha256(train_tokens),
            "protocol_audit": token_set_sha256(audit_tokens),
            "protocol_test": token_set_sha256(test_tokens),
            "portable_train": token_set_sha256(portable_train),
        },
        "sources": {
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "scene_filter_yaml": {
                "path": str(args.scene_filter_yaml.resolve()),
                "sha256": scene_filter_sha,
                "committed_snapshot": str(scene_filter_snapshot.resolve()),
                "committed_snapshot_sha256": sha256_file(scene_filter_snapshot),
            },
            "scene_loader_source": {
                "path": str(args.scene_loader_source.resolve()),
                "sha256": scene_loader_sha,
                "committed_snapshot": str(scene_loader_snapshot.resolve()),
                "committed_snapshot_sha256": sha256_file(scene_loader_snapshot),
            },
            "raw_selected_tree": {
                "path": str(args.raw_log_root.resolve()),
                "selected_bytes": selected_bytes,
                "content_sha256": raw_tree_sha,
            },
            "cache_index": {
                "path": str(args.cache_index.resolve()),
                "sha256": sha256_bytes(cache_bytes),
                "provenance_receipt": (
                    str(args.cache_provenance.resolve())
                    if args.cache_provenance is not None
                    else None
                ),
                "provenance_receipt_valid": cache_provenance_valid,
            },
            "protocol_manifest": {
                "path": str(args.protocol_manifest.resolve()),
                "sha256": source_manifest_sha,
                "content_sha256": protocol.get("content_sha256"),
            },
            "portable_manifest": {
                "path": str(args.portable_manifest.resolve()),
                "sha256": sha256_file(args.portable_manifest),
                "content_sha256": portable.get("content_sha256"),
            },
        },
        "tables": {
            "per_log_csv": str(log_csv.resolve()),
            "per_log_csv_sha256": sha256_file(log_csv),
            "per_session_csv": str(session_csv.resolve()),
            "per_session_csv_sha256": sha256_file(session_csv),
            "exact_token_membership_tsv_gz": str(membership_path.resolve()),
            "exact_token_membership_tsv_gz_sha256": sha256_file(membership_path),
        },
        "invariants": invariants,
        "failed_invariants": sorted(
            name for name, passed in invariants.items() if not passed
        ),
        "evidence_boundary": (
            "The audit proves raw allowlist membership and cache/protocol identity. "
            "A PASS additionally requires a reproducible cache-subset selection rule. "
            "It does not prove that a remote server has byte-identical raw logs or caches."
        ),
        "wall_seconds": time.monotonic() - started,
    }
    output = args.output_dir / "reconciliation.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

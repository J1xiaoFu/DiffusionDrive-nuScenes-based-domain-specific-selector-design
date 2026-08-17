"""Leakage-safe NAVSIM continual-learning protocol construction.

The atomic unit is a complete vehicle capture session, not a sampled scene or
one of the segmented NAVSIM log folders.  A session therefore belongs to one
and only one (stage, split) cell.  Exact cached tokens are written into the
manifest so training does not silently change when a source split file does.
"""

from __future__ import annotations

import csv
import hashlib
import math
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from selector_bench.utils.hashing import hash_file, hash_jsonable


SCHEMA = "selector_bench.drive_cl_protocol.v1"
SPLITS = ("train", "audit", "test")
SESSION_PATTERN = re.compile(
    r"^(?P<session>\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-[^_]+)"
    r"(?:_\d+_\d+)?$"
)


class ProtocolError(ValueError):
    """Raised when a protocol would violate an experimental invariant."""


def session_id_from_log(log_name: str) -> str:
    """Return the timestamp/vehicle session shared by segmented log folders."""

    name = Path(str(log_name)).name
    match = SESSION_PATTERN.fullmatch(name)
    if match is None:
        raise ProtocolError(f"unrecognized NAVSIM log name: {log_name!r}")
    return match.group("session")


def _stable_rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).hexdigest()


def _finite_float(row: Mapping[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ProtocolError(f"invalid {key}={raw!r}") from exc
    return value if math.isfinite(value) else None


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ProtocolError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ProtocolError(f"quantile must be in [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_cache_inventory(cache_index: str | Path) -> dict[str, dict[str, Any]]:
    """Load NAVSIM's trusted local token-to-cache-path index by full session."""

    index_path = Path(cache_index)
    with index_path.open("rb") as stream:
        token_paths = pickle.load(stream)  # noqa: S301 - trusted local NAVSIM cache
    if not isinstance(token_paths, dict) or not token_paths:
        raise ProtocolError(f"cache index is not a non-empty mapping: {index_path}")

    grouped: dict[str, dict[str, Any]] = {}
    seen_tokens: set[str] = set()
    for raw_token, raw_path in token_paths.items():
        token = str(raw_token)
        if token in seen_tokens:
            raise ProtocolError(f"duplicate cache token: {token}")
        seen_tokens.add(token)
        log_name = Path(raw_path).parent.name
        session = session_id_from_log(log_name)
        record = grouped.setdefault(session, {"logs": defaultdict(list)})
        record["logs"][log_name].append(token)

    inventory: dict[str, dict[str, Any]] = {}
    for session, record in grouped.items():
        logs = {
            log_name: sorted(tokens)
            for log_name, tokens in sorted(record["logs"].items())
        }
        inventory[session] = {
            "session_id": session,
            "logs": logs,
            "log_count": len(logs),
            "token_count": sum(len(tokens) for tokens in logs.values()),
        }
    return dict(sorted(inventory.items()))


def _partition_groups(
    sessions: Sequence[str],
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
) -> dict[str, list[str]]:
    """Deterministically balance complete sessions by cached-token count."""

    if len(ratios) != len(SPLITS) or any(value <= 0 for value in ratios):
        raise ProtocolError("split ratios must contain three positive values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ProtocolError(f"split ratios must sum to one: {ratios}")
    if len(sessions) < len(SPLITS):
        raise ProtocolError("each stage needs at least three complete sessions")

    total_weight = sum(int(inventory[item]["token_count"]) for item in sessions)
    targets = {
        split: total_weight * ratio for split, ratio in zip(SPLITS, ratios)
    }
    bins: dict[str, list[str]] = {split: [] for split in SPLITS}
    weights = {split: 0 for split in SPLITS}
    ranked = sorted(
        sessions,
        key=lambda item: (
            -int(inventory[item]["token_count"]),
            _stable_rank(item, seed),
        ),
    )
    for index, session in enumerate(ranked):
        empty = [split for split in SPLITS if not bins[split]]
        sessions_left = len(ranked) - index
        if empty and sessions_left == len(empty):
            chosen = empty[0]
        else:
            chosen = max(
                SPLITS,
                key=lambda split: (
                    (targets[split] - weights[split]) / targets[split],
                    -SPLITS.index(split),
                ),
            )
        bins[chosen].append(session)
        weights[chosen] += int(inventory[session]["token_count"])
    return {split: sorted(values) for split, values in bins.items()}


def _stage_payload(
    *,
    stage_index: int,
    name: str,
    sessions: Sequence[str],
    inventory: Mapping[str, Mapping[str, Any]],
    seed: int,
    session_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    assignment = _partition_groups(
        sessions, inventory, seed=seed + 10_007 * stage_index
    )
    split_payload: dict[str, Any] = {}
    for split, split_sessions in assignment.items():
        logs: list[str] = []
        tokens: list[str] = []
        token_to_log: dict[str, str] = {}
        for session in split_sessions:
            for log_name, log_tokens in inventory[session]["logs"].items():
                logs.append(log_name)
                tokens.extend(log_tokens)
                for token in log_tokens:
                    token_to_log[token] = log_name
        split_payload[split] = {
            "sessions": sorted(split_sessions),
            "logs": sorted(logs),
            "tokens": sorted(tokens),
            "token_to_log": dict(sorted(token_to_log.items())),
            "session_count": len(split_sessions),
            "log_count": len(logs),
            "token_count": len(tokens),
        }
    result: dict[str, Any] = {
        "stage_index": stage_index,
        "name": name,
        "splits": split_payload,
    }
    if session_metrics is not None:
        result["selection_metrics"] = {
            session: dict(session_metrics[session]) for session in sorted(sessions)
        }
    return result


def _protocol_envelope(
    *,
    protocol_id: str,
    protocol_type: str,
    seed: int,
    inventory: Mapping[str, Mapping[str, Any]],
    source_files: Mapping[str, str | Path],
    stages: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    sources = {
        label: {"path": str(Path(path).resolve()), "sha256": hash_file(path)}
        for label, path in source_files.items()
    }
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "protocol_type": protocol_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "atomic_unit": "complete_timestamp_vehicle_session",
        "session_pattern": SESSION_PATTERN.pattern,
        "split_ratios": {"train": 0.7, "audit": 0.1, "test": 0.2},
        "sources": sources,
        "inventory": {
            "session_count": len(inventory),
            "log_count": sum(int(row["log_count"]) for row in inventory.values()),
            "token_count": sum(int(row["token_count"]) for row in inventory.values()),
        },
        "selection": dict(selection),
        "stages": list(stages),
    }
    validate_protocol(protocol)
    canonical = dict(protocol)
    canonical.pop("created_at_utc")
    protocol["content_sha256"] = hash_jsonable(canonical)
    return protocol


def build_chronological_protocol(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    cache_index: str | Path,
    protocol_id: str = "chronological_cl_v1",
    seed: int = 0,
) -> dict[str, Any]:
    """Build three chronological, approximately token-balanced stages."""

    sessions = sorted(inventory)
    if len(sessions) < 9:
        raise ProtocolError("chronological protocol needs at least nine sessions")
    total = sum(int(inventory[item]["token_count"]) for item in sessions)
    prefix: list[int] = [0]
    for session in sessions:
        prefix.append(prefix[-1] + int(inventory[session]["token_count"]))
    cut1 = min(range(3, len(sessions) - 5), key=lambda i: abs(prefix[i] - total / 3))
    cut2 = min(
        range(cut1 + 3, len(sessions) - 2),
        key=lambda i: abs(prefix[i] - 2 * total / 3),
    )
    groups = (sessions[:cut1], sessions[cut1:cut2], sessions[cut2:])
    stages = [
        _stage_payload(
            stage_index=index,
            name=f"chronological_{index}",
            sessions=group,
            inventory=inventory,
            seed=seed,
        )
        for index, group in enumerate(groups, start=1)
    ]
    return _protocol_envelope(
        protocol_id=protocol_id,
        protocol_type="chronological",
        seed=seed,
        inventory=inventory,
        source_files={"cache_index": cache_index},
        stages=stages,
        selection={
            "rule": "chronological_session_order_with_nearest_one_third_token_cuts",
            "cut_session_indices": [cut1, cut2],
            "positive_transfer_is_not_assumed_to_be_conflict": True,
        },
    )


def _read_deployment_metrics(
    metrics_csv: str | Path,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    progress_quantile: float,
) -> tuple[dict[str, dict[str, float]], float]:
    fields = {
        "nc": "dd_raw_no_at_fault_collisions",
        "dac": "dd_raw_drivable_area_compliance",
        "ttc": "dd_raw_time_to_collision_within_bound",
        "comfort": "dd_raw_comfort",
        "progress": "dd_raw_ego_progress",
    }
    rows_by_session: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    with Path(metrics_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            log_name = (row.get("dd_raw_log") or "").strip()
            if not log_name:
                scene_id = (row.get("scene_id") or "").strip()
                log_name = scene_id.split("/", 1)[0]
            if not log_name:
                continue
            session = session_id_from_log(log_name)
            if session not in inventory or (row.get("dd_raw_valid") or "1") != "1":
                continue
            rows_by_session[session].append(row)

    safe_progress: list[float] = []
    for rows in rows_by_session.values():
        for row in rows:
            safety_values = [
                _finite_float(row, fields[name]) for name in ("nc", "dac", "ttc")
            ]
            progress = _finite_float(row, fields["progress"])
            if (
                all(value is not None and value >= 1.0 for value in safety_values)
                and progress is not None
            ):
                safe_progress.append(progress)
    progress_threshold = _quantile(safe_progress, progress_quantile)

    metrics: dict[str, dict[str, float]] = {}
    for session, rows in rows_by_session.items():
        values = {
            name: [
                value
                for row in rows
                if (value := _finite_float(row, source)) is not None
            ]
            for name, source in fields.items()
        }
        if any(not values[name] for name in fields):
            continue
        unsafe = [
            float(min(nc, dac, ttc) < 1.0)
            for nc, dac, ttc in zip(values["nc"], values["dac"], values["ttc"])
        ]
        mean = lambda key: sum(values[key]) / len(values[key])
        safety_risk = sum(unsafe) / len(unsafe)
        progress = mean("progress")
        comfort_risk = sum(value < 1.0 for value in values["comfort"]) / len(
            values["comfort"]
        )
        safe_rows = [
            row
            for row in rows
            if all(
                (value := _finite_float(row, fields[name])) is not None
                and value >= 1.0
                for name in ("nc", "dac", "ttc")
            )
        ]
        safe_efficiency_failures = [
            float(
                (_finite_float(row, fields["progress"]) or 0.0) < progress_threshold
                or (_finite_float(row, fields["comfort"]) or 0.0) < 1.0
            )
            for row in safe_rows
        ]
        safe_progress_values = [
            value
            for row in safe_rows
            if (value := _finite_float(row, fields["progress"])) is not None
        ]
        metrics[session] = {
            "scored_scene_count": float(len(rows)),
            "safety_union_failure_rate": safety_risk,
            "nc_failure_rate": 1.0 - mean("nc"),
            "dac_failure_rate": 1.0 - mean("dac"),
            "ttc_failure_rate": 1.0 - mean("ttc"),
            "mean_progress": progress,
            "comfort_failure_rate": comfort_risk,
            "safe_scene_count": float(len(safe_rows)),
            "safe_mean_progress": (
                sum(safe_progress_values) / len(safe_progress_values)
                if safe_progress_values
                else 0.0
            ),
            "efficiency_risk": (
                sum(safe_efficiency_failures) / len(safe_efficiency_failures)
                if safe_efficiency_failures
                else 1.0
            ),
        }
    return metrics, progress_threshold


def build_failure_patch_protocol(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    cache_index: str | Path,
    metrics_csv: str | Path,
    protocol_id: str = "failure_patch_cl_v1",
    seed: int = 0,
    safety_fraction: float = 0.3,
    efficiency_fraction: float = 0.3,
    safe_pool_fraction: float = 0.5,
    progress_quantile: float = 0.25,
) -> dict[str, Any]:
    """Build general -> safety patch -> safe/inefficient patch deployment stages."""

    for name, value in (
        ("safety_fraction", safety_fraction),
        ("efficiency_fraction", efficiency_fraction),
        ("safe_pool_fraction", safe_pool_fraction),
        ("progress_quantile", progress_quantile),
    ):
        if not 0.0 < value < 1.0:
            raise ProtocolError(f"{name} must be in (0, 1), got {value}")
    if safety_fraction + efficiency_fraction >= 1.0:
        raise ProtocolError("failure patch fractions must leave a non-empty base stage")

    metrics, progress_threshold = _read_deployment_metrics(
        metrics_csv, inventory, progress_quantile=progress_quantile
    )
    eligible = sorted(metrics)
    if len(eligible) < 9:
        raise ProtocolError("failure-patch protocol needs nine scored sessions")
    n_safety = max(3, round(len(eligible) * safety_fraction))
    n_efficiency = max(3, round(len(eligible) * efficiency_fraction))
    if n_safety + n_efficiency > len(eligible) - 3:
        raise ProtocolError("requested patch stages leave fewer than three base sessions")

    safety_ranked = sorted(
        eligible,
        key=lambda item: (
            -metrics[item]["safety_union_failure_rate"],
            -metrics[item]["nc_failure_rate"],
            -metrics[item]["ttc_failure_rate"],
            _stable_rank(item, seed),
        ),
    )
    safety_sessions = safety_ranked[:n_safety]
    remaining = [item for item in eligible if item not in set(safety_sessions)]
    safe_pool_size = max(n_efficiency, math.ceil(len(remaining) * safe_pool_fraction))
    safe_pool = sorted(
        remaining,
        key=lambda item: (
            metrics[item]["safety_union_failure_rate"],
            _stable_rank(item, seed + 1),
        ),
    )[:safe_pool_size]
    efficiency_sessions = sorted(
        safe_pool,
        key=lambda item: (
            -metrics[item]["efficiency_risk"],
            metrics[item]["safety_union_failure_rate"],
            _stable_rank(item, seed + 2),
        ),
    )[:n_efficiency]
    assigned = set(safety_sessions) | set(efficiency_sessions)
    base_sessions = [item for item in eligible if item not in assigned]

    stage_specs = (
        (1, "general_base", base_sessions),
        (2, "safety_failure_patch", safety_sessions),
        (3, "safe_efficiency_patch", efficiency_sessions),
    )
    stages = [
        _stage_payload(
            stage_index=index,
            name=name,
            sessions=sessions,
            inventory=inventory,
            seed=seed,
            session_metrics=metrics,
        )
        for index, name, sessions in stage_specs
    ]
    safety_cutoff = min(
        metrics[item]["safety_union_failure_rate"] for item in safety_sessions
    )
    safe_pool_cutoff = max(
        metrics[item]["safety_union_failure_rate"] for item in safe_pool
    )
    efficiency_cutoff = min(metrics[item]["efficiency_risk"] for item in efficiency_sessions)
    return _protocol_envelope(
        protocol_id=protocol_id,
        protocol_type="failure_patch",
        seed=seed,
        inventory=inventory,
        source_files={"cache_index": cache_index, "deployment_metrics": metrics_csv},
        stages=stages,
        selection={
            "rule": "session_level_pre_registered_ranked_deployment_failure_stream",
            "unscored_sessions_excluded": sorted(set(inventory) - set(eligible)),
            "safety_fraction": safety_fraction,
            "efficiency_fraction": efficiency_fraction,
            "safe_pool_fraction": safe_pool_fraction,
            "safe_scene_progress_quantile": progress_quantile,
            "safe_scene_progress_failure_threshold": progress_threshold,
            "safety_patch_min_union_failure_rate": safety_cutoff,
            "efficiency_safe_pool_max_union_failure_rate": safe_pool_cutoff,
            "efficiency_patch_min_efficiency_risk": efficiency_cutoff,
            "binary_any_failure_rule_rejected": True,
        },
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Fail closed on session/log/token leakage or malformed stage matrices."""

    if protocol.get("schema") != SCHEMA:
        raise ProtocolError(f"unsupported protocol schema: {protocol.get('schema')}")
    stages = protocol.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ProtocolError("protocol must contain exactly three stages")
    seen_sessions: dict[str, str] = {}
    seen_logs: dict[str, str] = {}
    seen_tokens: dict[str, str] = {}
    for stage in stages:
        if set(stage.get("splits", {})) != set(SPLITS):
            raise ProtocolError(f"stage {stage.get('name')} must contain {SPLITS}")
        for split in SPLITS:
            cell = stage["splits"][split]
            label = f"stage{stage['stage_index']}:{split}"
            for kind, seen in (
                ("sessions", seen_sessions),
                ("logs", seen_logs),
                ("tokens", seen_tokens),
            ):
                values = cell.get(kind)
                if not isinstance(values, list) or len(values) != len(set(values)):
                    raise ProtocolError(f"{label} has invalid or duplicate {kind}")
                for value in values:
                    if value in seen:
                        raise ProtocolError(
                            f"{kind[:-1]} leakage: {value} in {seen[value]} and {label}"
                        )
                    seen[value] = label
            if int(cell.get("session_count", -1)) != len(cell["sessions"]):
                raise ProtocolError(f"{label} session_count mismatch")
            if int(cell.get("log_count", -1)) != len(cell["logs"]):
                raise ProtocolError(f"{label} log_count mismatch")
            if int(cell.get("token_count", -1)) != len(cell["tokens"]):
                raise ProtocolError(f"{label} token_count mismatch")
            token_to_log = cell.get("token_to_log")
            if not isinstance(token_to_log, dict) or set(token_to_log) != set(
                cell["tokens"]
            ):
                raise ProtocolError(f"{label} token_to_log mismatch")
            if not set(token_to_log.values()).issubset(set(cell["logs"])):
                raise ProtocolError(f"{label} token_to_log references an unknown log")


def protocol_rows(protocol: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield a flat stage/split/token table for external audit tools."""

    validate_protocol(protocol)
    for stage in protocol["stages"]:
        for split in SPLITS:
            cell = stage["splits"][split]
            for token in cell["tokens"]:
                yield {
                    "protocol_id": protocol["protocol_id"],
                    "stage_index": stage["stage_index"],
                    "stage_name": stage["name"],
                    "split": split,
                    "token": token,
                    "log": cell["token_to_log"][token],
                }

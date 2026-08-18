"""Controlled, log-session-atomic nuScenes continual-learning streams.

nuScenes may place multiple scenes in one capture log.  Treating scenes as
independent units can therefore leak one physical session across stages or
train/audit/test.  This module uses the complete ``log_token`` and all of its
child scenes/samples as the indivisible unit.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCHEMA = "selector_bench.nuscenes_controlled_cl.v1"
SPLITS = ("train", "audit", "test")


class NuScenesProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class NuScenesLogSession:
    log_token: str
    location: str
    scene_tokens: tuple[str, ...]
    scene_names: tuple[str, ...]
    sample_tokens: tuple[str, ...]
    first_timestamp: int
    last_timestamp: int

    @property
    def sample_count(self) -> int:
        return len(self.sample_tokens)


def _unique_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        token = str(row.get("token", ""))
        if not token or token in indexed:
            raise NuScenesProtocolError(f"{label} rows require unique non-empty tokens")
        indexed[token] = row
    return indexed


def build_log_sessions(
    metadata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    eligible_scene_names: set[str],
) -> tuple[NuScenesLogSession, ...]:
    """Build complete eligible log sessions without crossing populations.

    A log is included only when every scene in that log belongs to the
    eligible official population.  A partially eligible log is rejected,
    because silently dropping its other scenes would break session atomicity.
    """

    for table in ("log", "scene", "sample"):
        if table not in metadata:
            raise NuScenesProtocolError(f"metadata lacks required {table} table")
    if not eligible_scene_names:
        raise NuScenesProtocolError("eligible official scene names are empty")
    logs = _unique_rows(metadata["log"], "log")
    scenes = _unique_rows(metadata["scene"], "scene")
    samples = _unique_rows(metadata["sample"], "sample")

    scenes_by_log: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for scene in scenes.values():
        log_token = str(scene.get("log_token", ""))
        if log_token not in logs:
            raise NuScenesProtocolError("scene references an unknown log token")
        scenes_by_log[log_token].append(scene)
    samples_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples.values():
        scene_token = str(sample.get("scene_token", ""))
        if scene_token not in scenes:
            raise NuScenesProtocolError("sample references an unknown scene token")
        samples_by_scene[scene_token].append(sample)

    sessions: list[NuScenesLogSession] = []
    for log_token, child_scenes in sorted(scenes_by_log.items()):
        names = {str(scene.get("name", "")) for scene in child_scenes}
        eligible = names & eligible_scene_names
        if not eligible:
            continue
        if names != eligible:
            raise NuScenesProtocolError(
                f"log {log_token} crosses the eligible official scene population"
            )
        ordered_scenes = sorted(child_scenes, key=lambda row: str(row["token"]))
        child_samples: list[Mapping[str, Any]] = []
        for scene in ordered_scenes:
            scene_samples = samples_by_scene[str(scene["token"])]
            if not scene_samples:
                raise NuScenesProtocolError("eligible scene contains no samples")
            declared = scene.get("nbr_samples")
            if declared is not None and int(declared) != len(scene_samples):
                raise NuScenesProtocolError("scene sample count disagrees with metadata")
            child_samples.extend(scene_samples)
        ordered_samples = sorted(
            child_samples,
            key=lambda row: (int(row.get("timestamp", -1)), str(row["token"])),
        )
        timestamps = [int(row.get("timestamp", -1)) for row in ordered_samples]
        if any(timestamp < 0 for timestamp in timestamps):
            raise NuScenesProtocolError("sample timestamps must be non-negative integers")
        location = str(logs[log_token].get("location", ""))
        if not location:
            raise NuScenesProtocolError("eligible log lacks a location")
        sessions.append(
            NuScenesLogSession(
                log_token=log_token,
                location=location,
                scene_tokens=tuple(str(row["token"]) for row in ordered_scenes),
                scene_names=tuple(str(row["name"]) for row in ordered_scenes),
                sample_tokens=tuple(str(row["token"]) for row in ordered_samples),
                first_timestamp=min(timestamps),
                last_timestamp=max(timestamps),
            )
        )
    if not sessions:
        raise NuScenesProtocolError("no complete eligible log sessions were found")
    return tuple(sessions)


def _rank(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def _split_sessions(
    sessions: Sequence[NuScenesLogSession], *, seed: int
) -> dict[str, tuple[NuScenesLogSession, ...]]:
    if len(sessions) < len(SPLITS):
        raise NuScenesProtocolError("every stage needs at least three complete logs")
    ratios = {"train": 0.7, "audit": 0.1, "test": 0.2}
    total = sum(session.sample_count for session in sessions)
    targets = {split: total * ratio for split, ratio in ratios.items()}
    bins: dict[str, list[NuScenesLogSession]] = {split: [] for split in SPLITS}
    weights = {split: 0 for split in SPLITS}
    ordered = sorted(
        sessions,
        key=lambda item: (-item.sample_count, _rank(item.log_token, seed)),
    )
    for index, session in enumerate(ordered):
        empty = [split for split in SPLITS if not bins[split]]
        remaining = len(ordered) - index
        if empty and remaining == len(empty):
            chosen = empty[0]
        else:
            chosen = max(
                SPLITS,
                key=lambda split: (
                    (targets[split] - weights[split]) / max(targets[split], 1.0),
                    -SPLITS.index(split),
                ),
            )
        bins[chosen].append(session)
        weights[chosen] += session.sample_count
    return {
        split: tuple(sorted(values, key=lambda item: item.log_token))
        for split, values in bins.items()
    }


def _cell_payload(sessions: Sequence[NuScenesLogSession]) -> dict[str, Any]:
    return {
        "log_tokens": [item.log_token for item in sessions],
        "scene_tokens": sorted(token for item in sessions for token in item.scene_tokens),
        "scene_names": sorted(name for item in sessions for name in item.scene_names),
        "sample_tokens": sorted(token for item in sessions for token in item.sample_tokens),
        "log_count": len(sessions),
        "scene_count": sum(len(item.scene_tokens) for item in sessions),
        "sample_count": sum(item.sample_count for item in sessions),
        "location_counts": dict(sorted(Counter(item.location for item in sessions).items())),
    }


def _manifest(
    protocol_id: str,
    protocol_type: str,
    sessions: Sequence[NuScenesLogSession],
    stage_groups: Sequence[Sequence[NuScenesLogSession]],
    *,
    seed: int,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for index, group in enumerate(stage_groups, start=1):
        split_groups = _split_sessions(group, seed=seed + 10_007 * index)
        stages.append(
            {
                "stage_index": index,
                "splits": {
                    split: _cell_payload(split_groups[split]) for split in SPLITS
                },
            }
        )
    payload = {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "protocol_type": protocol_type,
        "atomic_unit": "complete_nuscenes_log_token",
        "seed": seed,
        "selection": dict(selection),
        "session_inventory": [asdict(item) for item in sorted(sessions, key=lambda x: x.log_token)],
        "stages": stages,
    }
    validate_protocol(payload)
    return payload


def _three_way_contiguous(
    ordered: Sequence[NuScenesLogSession],
) -> tuple[tuple[NuScenesLogSession, ...], ...]:
    groups: list[list[NuScenesLogSession]] = [[], [], []]
    for index, session in enumerate(ordered):
        stage = min(2, index * 3 // len(ordered))
        groups[stage].append(session)
    if any(not group for group in groups):
        raise NuScenesProtocolError("three-stage construction produced an empty stage")
    return tuple(tuple(group) for group in groups)


def build_time_only_protocol(
    sessions: Sequence[NuScenesLogSession], *, seed: int = 0
) -> dict[str, Any]:
    """Advance time within every location, keeping location mixture present."""

    by_location: dict[str, list[NuScenesLogSession]] = defaultdict(list)
    for session in sessions:
        by_location[session.location].append(session)
    stage_groups: list[list[NuScenesLogSession]] = [[], [], []]
    for location, values in sorted(by_location.items()):
        groups = _three_way_contiguous(
            sorted(values, key=lambda item: (item.first_timestamp, item.log_token))
        )
        for stage, group in enumerate(groups):
            stage_groups[stage].extend(group)
    return _manifest(
        "nuscenes_time_only_v1",
        "time_only",
        sessions,
        stage_groups,
        seed=seed,
        selection={
            "rule": "within-location chronological terciles",
            "location_mixture_control": "each location advances through all three stages",
        },
    )


def build_geography_only_protocol(
    sessions: Sequence[NuScenesLogSession],
    *,
    location_groups: Sequence[Sequence[str]],
    seed: int = 0,
) -> dict[str, Any]:
    """Change geography while restricting all stages to their common time window."""

    if len(location_groups) != 3:
        raise NuScenesProtocolError("geography-only needs exactly three location groups")
    normalized = [set(group) for group in location_groups]
    if any(not group for group in normalized):
        raise NuScenesProtocolError("geography location groups must be non-empty")
    if any(normalized[i] & normalized[j] for i in range(3) for j in range(i + 1, 3)):
        raise NuScenesProtocolError("geography location groups must be disjoint")
    observed_locations = {session.location for session in sessions}
    if set().union(*normalized) != observed_locations:
        raise NuScenesProtocolError("geography groups must exactly cover observed locations")
    raw_groups = [
        [session for session in sessions if session.location in group]
        for group in normalized
    ]
    overlap_start = max(min(item.first_timestamp for item in group) for group in raw_groups)
    overlap_end = min(max(item.first_timestamp for item in group) for group in raw_groups)
    if overlap_start > overlap_end:
        raise NuScenesProtocolError("geography groups have no common capture-time window")
    stage_groups = [
        [item for item in group if overlap_start <= item.first_timestamp <= overlap_end]
        for group in raw_groups
    ]
    if any(len(group) < 3 for group in stage_groups):
        raise NuScenesProtocolError("time matching leaves too few complete logs in a stage")
    used = tuple(item for group in stage_groups for item in group)
    return _manifest(
        "nuscenes_geography_only_v1",
        "geography_only",
        used,
        stage_groups,
        seed=seed,
        selection={
            "rule": "disjoint location groups inside their common absolute time window",
            "location_groups": [sorted(group) for group in normalized],
            "common_time_window": [overlap_start, overlap_end],
            "excluded_outside_common_window": len(sessions) - len(used),
        },
    )


def build_natural_mixed_protocol(
    sessions: Sequence[NuScenesLogSession], *, seed: int = 0
) -> dict[str, Any]:
    """Historical-style global chronological flow where time and place may co-vary."""

    groups = _three_way_contiguous(
        sorted(sessions, key=lambda item: (item.first_timestamp, item.log_token))
    )
    return _manifest(
        "nuscenes_natural_mixed_v1",
        "natural_mixed",
        sessions,
        groups,
        seed=seed,
        selection={"rule": "global chronological terciles; geography may co-vary"},
    )


def validate_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise NuScenesProtocolError("unsupported nuScenes CL schema")
    if payload.get("atomic_unit") != "complete_nuscenes_log_token":
        raise NuScenesProtocolError("nuScenes CL must be complete-log atomic")
    inventory_rows = payload.get("session_inventory")
    if not isinstance(inventory_rows, list) or not inventory_rows:
        raise NuScenesProtocolError("session inventory is empty")
    inventory: dict[str, Mapping[str, Any]] = {}
    for row in inventory_rows:
        if not isinstance(row, dict):
            raise NuScenesProtocolError("invalid session inventory row")
        token = str(row.get("log_token", ""))
        if not token or token in inventory:
            raise NuScenesProtocolError("session inventory log tokens must be unique")
        inventory[token] = row
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise NuScenesProtocolError("nuScenes CL requires exactly three stages")

    seen_logs: set[str] = set()
    seen_scenes: set[str] = set()
    seen_samples: set[str] = set()
    for expected_stage, stage in enumerate(stages, start=1):
        if stage.get("stage_index") != expected_stage or set(stage.get("splits", {})) != set(SPLITS):
            raise NuScenesProtocolError("invalid stage or split layout")
        for split in SPLITS:
            cell = stage["splits"][split]
            logs = list(cell.get("log_tokens", []))
            if not logs:
                raise NuScenesProtocolError("every stage/split cell needs a complete log")
            expected_scenes: set[str] = set()
            expected_samples: set[str] = set()
            for log_token in logs:
                if log_token not in inventory or log_token in seen_logs:
                    raise NuScenesProtocolError("log leakage or unknown log membership")
                seen_logs.add(log_token)
                expected_scenes.update(inventory[log_token]["scene_tokens"])
                expected_samples.update(inventory[log_token]["sample_tokens"])
            scenes = set(cell.get("scene_tokens", []))
            samples = set(cell.get("sample_tokens", []))
            if scenes != expected_scenes or samples != expected_samples:
                raise NuScenesProtocolError("cell breaks complete log child closure")
            if scenes & seen_scenes or samples & seen_samples:
                raise NuScenesProtocolError("scene/sample leakage across cells")
            seen_scenes.update(scenes)
            seen_samples.update(samples)
    if seen_logs != set(inventory):
        raise NuScenesProtocolError("protocol does not cover its declared session inventory")

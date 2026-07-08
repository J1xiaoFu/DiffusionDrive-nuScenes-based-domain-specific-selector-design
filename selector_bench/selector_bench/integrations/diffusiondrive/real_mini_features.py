from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.manifest import Manifest, ManifestRecord
from selector_bench.utils.io import ensure_dir, write_json


CMD_NAMES = ["cmd_0", "cmd_1", "cmd_2"]
TEACHER_SIGNAL_SOURCE = "metadata_proxy_not_genuine_teacher"


@dataclass
class _PreparedSample:
    info: dict[str, Any]
    split: str
    domain_id: int
    domain_name: str
    location: str
    route_cmd: str
    ego_speed: float
    agent_count: int
    traj_raw: np.ndarray
    scene_base: np.ndarray
    interaction_base: np.ndarray
    proxy_raw: float


def generate_real_mini_feature_cache(
    output_dir: str | Path,
    train_info: str | Path,
    val_info: str | Path | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    scene_dim: int = 256,
    interaction_dim: int = 256,
    seed: int = 1,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    train_path = Path(train_info)
    val_path = Path(val_info) if val_info else None

    train_payload = _load_info_payload(train_path)
    val_payload = _load_info_payload(val_path) if val_path else None
    train_infos = list(train_payload["infos"])
    val_infos = list(val_payload["infos"]) if val_payload else []
    if max_train_samples is not None:
        train_infos = train_infos[:max_train_samples]
    if max_val_samples is not None:
        val_infos = val_infos[:max_val_samples]

    locations = sorted(
        {
            str(info.get("map_location", "unknown"))
            for info in [*train_infos, *val_infos]
        }
    )
    location_to_domain = {location: idx for idx, location in enumerate(locations)}

    prepared = [
        _prepare_sample(info, "train", location_to_domain)
        for info in train_infos
    ]
    prepared.extend(
        _prepare_sample(info, "val", location_to_domain)
        for info in val_infos
    )

    proxy_values = np.asarray([sample.proxy_raw for sample in prepared], dtype=np.float32)
    proxy_min = float(proxy_values.min()) if proxy_values.size else 0.0
    proxy_span = float(proxy_values.max() - proxy_min) if proxy_values.size else 1.0
    proxy_span = max(proxy_span, 1e-6)

    records: list[ManifestRecord] = []
    scene_projector = _projector(seed, "scene", max(len(sample.scene_base) for sample in prepared), scene_dim)
    interaction_projector = _projector(
        seed,
        "interaction",
        max(len(sample.interaction_base) for sample in prepared),
        interaction_dim,
    )

    for sample in prepared:
        token = str(sample.info["token"])
        feature_split_dir = ensure_dir(output / "features" / sample.split)
        feature_path = Path("features") / sample.split / f"{token}.npz"
        teacher_loss = float((sample.proxy_raw - proxy_min) / proxy_span)
        teacher_uncertainty = float(
            np.clip(0.5 * teacher_loss + 0.5 * _trajectory_uncertainty(sample.traj_raw), 0.0, 1.0)
        )
        scene_hidden = _project(sample.scene_base, scene_projector).astype(np.float32)
        interaction_hidden = _project(sample.interaction_base, interaction_projector).astype(np.float32)
        traj_hidden = _traj_summary(sample.traj_raw)

        np.savez_compressed(
            feature_split_dir / f"{token}.npz",
            traj_raw=sample.traj_raw.astype(np.float32),
            traj_hidden=traj_hidden,
            scene_hidden=scene_hidden,
            interaction_hidden=interaction_hidden,
            domain_id=np.asarray(sample.domain_id, dtype=np.int64),
            teacher_loss=np.asarray(teacher_loss, dtype=np.float32),
            teacher_uncertainty=np.asarray(teacher_uncertainty, dtype=np.float32),
        )

        records.append(
            ManifestRecord(
                sample_token=token,
                scene_token=str(sample.info.get("scene_token", "unknown")),
                timestamp=int(sample.info.get("timestamp", 0)),
                split=sample.split,
                domain_id=sample.domain_id,
                domain_name=sample.domain_name,
                location=sample.location,
                weather_tag="unknown",
                time_tag=_time_tag(sample.info.get("timestamp", 0)),
                route_cmd=sample.route_cmd,
                ego_speed=sample.ego_speed,
                agent_count=sample.agent_count,
                feature_path=str(feature_path),
                teacher_loss=teacher_loss,
                teacher_uncertainty=teacher_uncertainty,
                metadata={
                    "source": "real_nuscenes_mini_info_metadata_features",
                    "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
                    "teacher_signal_caveat": (
                        "teacher_loss and teacher_uncertainty are deterministic metadata-derived "
                        "difficulty proxies, not genuine DiffusionDrive teacher outputs."
                    ),
                    "map_location": sample.location,
                    "gt_ego_fut_cmd": _as_list(sample.info.get("gt_ego_fut_cmd")),
                    "num_gt_boxes": sample.agent_count,
                },
            )
        )

    manifest = Manifest.from_records(records)
    manifest.save(output / "manifest.jsonl")
    stats = {
        "manifest": manifest.summary(),
        "feature_shapes": {
            "traj_raw": list(prepared[0].traj_raw.shape) if prepared else [0, 0],
            "traj_hidden": [32],
            "scene_hidden": [scene_dim],
            "interaction_hidden": [interaction_dim],
        },
        "source_train_info": str(train_path),
        "source_val_info": str(val_path) if val_path else None,
        "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
        "teacher_signal_caveat": "No genuine teacher loss/uncertainty was extracted in this path.",
        "seed": seed,
    }
    write_json(output / "feature_stats.json", stats)
    write_json(
        output / "artifact_metadata.json",
        {
            **make_artifact_metadata(
                artifact_type="feature_cache",
                run_id="real_mini_metadata_features",
                config={
                    "train_info": str(train_path),
                    "val_info": str(val_path) if val_path else None,
                    "max_train_samples": max_train_samples,
                    "max_val_samples": max_val_samples,
                    "scene_dim": scene_dim,
                    "interaction_dim": interaction_dim,
                    "seed": seed,
                    "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
                },
                input_paths=[p for p in [train_path, val_path] if p is not None],
                summary=stats["manifest"],
            ),
            "teacher_signal_caveat": stats["teacher_signal_caveat"],
        },
    )
    return stats


def _load_info_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"infos": [], "metadata": {}}
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise ValueError(f"Expected DiffusionDrive info dict with an 'infos' field: {path}")
    return payload


def _prepare_sample(
    info: dict[str, Any],
    split: str,
    location_to_domain: dict[str, int],
) -> _PreparedSample:
    location = str(info.get("map_location", "unknown"))
    domain_id = int(location_to_domain[location])
    cmd_idx = _cmd_index(info.get("gt_ego_fut_cmd"))
    ego_status = np.nan_to_num(np.asarray(info.get("ego_status", []), dtype=np.float32), nan=0.0)
    gt_boxes = np.nan_to_num(np.asarray(info.get("gt_boxes", []), dtype=np.float32), nan=0.0)
    gt_velocity = np.nan_to_num(np.asarray(info.get("gt_velocity", []), dtype=np.float32), nan=0.0)
    ego_traj = np.nan_to_num(np.asarray(info.get("gt_ego_fut_trajs", []), dtype=np.float32), nan=0.0)
    traj_raw = _traj_raw(ego_traj, info.get("gt_ego_fut_masks"), cmd_idx)
    agent_count = int(gt_boxes.shape[0]) if gt_boxes.ndim == 2 else 0
    ego_speed = _ego_speed(ego_status, traj_raw)
    map_counts = _map_counts(info.get("map_annos", {}))

    scene_base = np.concatenate(
        [
            _normalize_vector(info.get("ego2global_translation"), 3, 2000.0),
            _normalize_vector(info.get("ego2global_rotation"), 4, 1.0),
            ego_status[:10] / np.asarray([20, 20, 30, 1, 1, 1, 30, 1, 1, 1], dtype=np.float32),
            _one_hot(domain_id, max(location_to_domain.values()) + 1),
            _one_hot(cmd_idx, len(CMD_NAMES)),
            np.asarray(map_counts, dtype=np.float32) / 100.0,
        ]
    ).astype(np.float32)
    interaction_base = np.concatenate(
        [
            _box_summary(gt_boxes),
            _velocity_summary(gt_velocity),
            _agent_future_summary(info.get("gt_agent_fut_trajs"), info.get("gt_agent_fut_masks")),
            np.asarray([agent_count / 128.0, ego_speed / 30.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    proxy_raw = float(
        0.45 * min(agent_count / 80.0, 1.5)
        + 0.20 * min(ego_speed / 20.0, 1.5)
        + 0.20 * _trajectory_uncertainty(traj_raw)
        + 0.15 * min(float(sum(map_counts)) / 120.0, 1.5)
    )
    return _PreparedSample(
        info=info,
        split=split,
        domain_id=domain_id,
        domain_name=location,
        location=location,
        route_cmd=CMD_NAMES[cmd_idx],
        ego_speed=ego_speed,
        agent_count=agent_count,
        traj_raw=traj_raw,
        scene_base=scene_base,
        interaction_base=interaction_base,
        proxy_raw=proxy_raw,
    )


def _traj_raw(ego_traj: np.ndarray, masks: Any, cmd_idx: int) -> np.ndarray:
    if ego_traj.ndim != 2 or ego_traj.shape[1] != 2:
        ego_traj = np.zeros((6, 2), dtype=np.float32)
    mask = np.ones((ego_traj.shape[0],), dtype=np.float32)
    if masks is not None:
        raw_mask = np.asarray(masks, dtype=np.float32).reshape(-1)
        if raw_mask.size >= mask.size:
            mask = raw_mask[: mask.size]
    delta = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), ego_traj]), axis=0)
    step_speed = np.linalg.norm(delta, axis=1)
    cumulative = np.linalg.norm(ego_traj, axis=1)
    cmd = np.full((ego_traj.shape[0],), float(cmd_idx), dtype=np.float32)
    return np.stack([ego_traj[:, 0], ego_traj[:, 1], step_speed, cumulative, cmd, mask], axis=1)


def _traj_summary(traj_raw: np.ndarray) -> np.ndarray:
    summary = np.concatenate([traj_raw.mean(axis=0), traj_raw.std(axis=0), traj_raw[-1]])
    return np.pad(summary, (0, max(0, 32 - summary.size)))[:32].astype(np.float32)


def _project(values: np.ndarray, projector: np.ndarray) -> np.ndarray:
    padded = np.zeros((projector.shape[0],), dtype=np.float32)
    padded[: values.size] = values
    hidden = np.tanh(padded @ projector)
    return hidden.astype(np.float32)


def _projector(seed: int, name: str, input_dim: int, output_dim: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(local_seed)
    return rng.normal(0.0, 1.0 / np.sqrt(max(input_dim, 1)), size=(input_dim, output_dim)).astype(np.float32)


def _cmd_index(cmd: Any) -> int:
    arr = np.asarray(cmd, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0
    return int(np.argmax(arr[: len(CMD_NAMES)]))


def _ego_speed(ego_status: np.ndarray, traj_raw: np.ndarray) -> float:
    if ego_status.size >= 3 and np.isfinite(ego_status[2]):
        return float(abs(ego_status[2]))
    return float(np.mean(traj_raw[:, 2])) if traj_raw.size else 0.0


def _normalize_vector(values: Any, target: int, scale: float) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
    out = np.zeros((target,), dtype=np.float32)
    out[: min(target, arr.size)] = arr[:target]
    return out / scale


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros((size,), dtype=np.float32)
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _box_summary(boxes: np.ndarray) -> np.ndarray:
    if boxes.ndim != 2 or boxes.size == 0:
        return np.zeros((16,), dtype=np.float32)
    centers = boxes[:, :3]
    dims = boxes[:, 3:6] if boxes.shape[1] >= 6 else np.zeros((boxes.shape[0], 3), dtype=np.float32)
    distances = np.linalg.norm(centers[:, :2], axis=1)
    values = np.asarray(
        [
            boxes.shape[0] / 128.0,
            distances.mean() / 80.0,
            distances.std() / 80.0,
            distances.min() / 80.0,
            distances.max() / 80.0,
            dims.mean(axis=0)[0] / 10.0,
            dims.mean(axis=0)[1] / 10.0,
            dims.mean(axis=0)[2] / 10.0,
            dims.std(axis=0)[0] / 10.0,
            dims.std(axis=0)[1] / 10.0,
            dims.std(axis=0)[2] / 10.0,
            float(np.mean(distances < 20.0)),
            float(np.mean(distances < 40.0)),
            float(np.mean(distances < 60.0)),
            float(np.mean(np.abs(centers[:, 0]) < 10.0)),
            float(np.mean(centers[:, 1] > 0.0)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0)


def _velocity_summary(velocity: np.ndarray) -> np.ndarray:
    if velocity.ndim != 2 or velocity.size == 0:
        return np.zeros((8,), dtype=np.float32)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    values = np.asarray(
        [
            speed.mean() / 15.0,
            speed.std() / 15.0,
            speed.max() / 15.0,
            float(np.mean(speed > 1.0)),
            float(np.mean(speed > 5.0)),
            velocity[:, 0].mean() / 15.0,
            velocity[:, 1].mean() / 15.0,
            np.linalg.norm(velocity[:, :2].mean(axis=0)) / 15.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0)


def _agent_future_summary(futures: Any, masks: Any) -> np.ndarray:
    fut = np.nan_to_num(np.asarray(futures if futures is not None else [], dtype=np.float32), nan=0.0)
    if fut.ndim != 3 or fut.size == 0:
        return np.zeros((10,), dtype=np.float32)
    mask = np.ones(fut.shape[:2], dtype=np.float32)
    if masks is not None:
        raw_mask = np.asarray(masks, dtype=np.float32)
        if raw_mask.shape[:2] == fut.shape[:2]:
            mask = raw_mask
    displacement = np.linalg.norm(fut, axis=2) * mask
    valid = np.maximum(mask.sum(axis=1), 1.0)
    per_agent = displacement.sum(axis=1) / valid
    final = np.linalg.norm(fut[:, -1, :], axis=1)
    values = np.asarray(
        [
            per_agent.mean() / 20.0,
            per_agent.std() / 20.0,
            per_agent.max() / 20.0,
            final.mean() / 20.0,
            final.std() / 20.0,
            float(np.mean(per_agent > 1.0)),
            float(np.mean(per_agent > 5.0)),
            float(np.mean(valid >= fut.shape[1])),
            float(mask.mean()),
            fut.shape[0] / 128.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0)


def _map_counts(map_annos: Any) -> list[float]:
    if not isinstance(map_annos, dict):
        return [0.0, 0.0, 0.0]
    counts = []
    for key in ["ped_crossing", "divider", "boundary"]:
        value = map_annos.get(key, [])
        try:
            counts.append(float(len(value)))
        except TypeError:
            counts.append(0.0)
    return counts


def _trajectory_uncertainty(traj_raw: np.ndarray) -> float:
    if traj_raw.ndim != 2 or traj_raw.shape[0] < 2:
        return 0.0
    heading_delta = np.diff(np.arctan2(traj_raw[:, 1], traj_raw[:, 0]))
    curvature = float(np.mean(np.abs(np.unwrap(heading_delta)))) if heading_delta.size else 0.0
    speed_var = float(np.std(traj_raw[:, 2]) / (np.mean(traj_raw[:, 2]) + 1e-3))
    return float(np.clip(0.5 * curvature + 0.5 * speed_var, 0.0, 1.0))


def _time_tag(timestamp: Any) -> str:
    try:
        hour = int(int(timestamp) / 1_000_000 / 3600) % 24
    except Exception:
        return "unknown"
    return "night" if hour < 6 or hour >= 20 else "day"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    arr = np.asarray(value)
    return arr.tolist()

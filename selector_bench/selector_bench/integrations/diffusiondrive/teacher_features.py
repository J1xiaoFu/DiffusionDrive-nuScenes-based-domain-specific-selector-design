from __future__ import annotations

import importlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.manifest import Manifest, ManifestRecord
from selector_bench.integrations.diffusiondrive.real_mini_features import (
    _as_list,
    _load_info_payload,
    _prepare_sample,
    _time_tag,
    _traj_summary,
)
from selector_bench.utils.io import ensure_dir, write_json


TEACHER_SIGNAL_SOURCE = "genuine_diffusiondrive_train_loss_and_hook_logits"


def generate_teacher_feature_cache(
    output_dir: str | Path,
    diffusiondrive_root: str | Path,
    config: str | Path,
    checkpoint: str | Path,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 123,
    overwrite: bool = False,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Score real DiffusionDrive samples and write a selector FeatureStore cache.

    This path intentionally runs the model loss code. It is therefore suitable
    as teacher-signal extraction, unlike the metadata-only real-mini cache.
    """

    output = Path(output_dir).resolve()
    if output.exists() and (output / "manifest.jsonl").exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing feature cache: {output}")
    ensure_dir(output)

    dd_root = Path(diffusiondrive_root).resolve()
    config_path = Path(config).resolve()
    checkpoint_path = Path(checkpoint).resolve()

    start_wall = time.perf_counter()
    with _working_directory(dd_root):
        _prepare_diffusiondrive_imports(dd_root)

        import torch
        from mmcv import Config
        from mmcv.parallel import scatter
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from mmdet.apis import set_random_seed
        from mmdet.datasets import build_dataset
        from mmdet.models import build_detector
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader

        _seed_everything(seed)
        set_random_seed(seed, deterministic=False)

        cfg = Config.fromfile(str(config_path))
        cfg.model.pretrained = None
        cfg.data.workers_per_gpu = 0
        data_cfg = getattr(cfg.data, split)
        data_cfg.work_dir = str(output / "_diffusiondrive_work")
        dataset = build_dataset(data_cfg)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=False,
            shuffle=False,
        )

        info_path = Path(data_cfg.ann_file)
        if not info_path.is_absolute():
            info_path = dd_root / info_path
        info_payload = _load_info_payload(info_path)
        info_metadata = dict(info_payload.get("metadata", {}))

        infos = list(dataset.data_infos)
        if max_samples is not None:
            infos = infos[:max_samples]
        locations = sorted({str(info.get("map_location", "unknown")) for info in infos})
        location_to_domain = {location: idx for idx, location in enumerate(locations)}

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        checkpoint_meta = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
        model.cuda()
        model.train()

        capture = _ActivationCapture()
        handles = capture.register(model)
        records: list[ManifestRecord] = []
        raw_losses: list[float] = []
        raw_uncertainties: list[float] = []
        feature_shapes: dict[str, list[int]] = {}
        peak_memory_mb = 0

        try:
            iterator = iter(data_loader)
            total = len(infos)
            for idx in range(total):
                info = infos[idx]
                token = str(info["token"])
                capture.clear()
                _seed_everything(seed + idx)
                data = next(iterator)
                scattered = scatter(data, [0])[0]

                with torch.no_grad():
                    losses = model.forward_train(**scattered)

                loss_components = _loss_components(losses)
                teacher_loss = float(sum(loss_components.values()))
                teacher_uncertainty = float(capture.uncertainty())
                sample = _prepare_sample(info, split, location_to_domain)

                scene_hidden = capture.scene_hidden()
                interaction_hidden = capture.interaction_hidden()
                traj_hidden = capture.traj_hidden(sample.traj_raw)
                feature_split_dir = ensure_dir(output / "features" / split)
                feature_path = Path("features") / split / f"{token}.npz"

                np.savez_compressed(
                    feature_split_dir / f"{token}.npz",
                    traj_raw=sample.traj_raw.astype(np.float32),
                    traj_hidden=traj_hidden.astype(np.float32),
                    scene_hidden=scene_hidden.astype(np.float32),
                    interaction_hidden=interaction_hidden.astype(np.float32),
                    domain_id=np.asarray(sample.domain_id, dtype=np.int64),
                    teacher_loss=np.asarray(teacher_loss, dtype=np.float32),
                    teacher_uncertainty=np.asarray(teacher_uncertainty, dtype=np.float32),
                    loss_component_names=np.asarray(list(loss_components.keys())),
                    loss_component_values=np.asarray(list(loss_components.values()), dtype=np.float32),
                )

                record = ManifestRecord(
                    sample_token=token,
                    scene_token=str(info.get("scene_token", "unknown")),
                    timestamp=int(info.get("timestamp", 0)),
                    split=split,
                    domain_id=sample.domain_id,
                    domain_name=sample.domain_name,
                    location=sample.location,
                    weather_tag="unknown",
                    time_tag=_time_tag(info.get("timestamp", 0)),
                    route_cmd=sample.route_cmd,
                    ego_speed=sample.ego_speed,
                    agent_count=sample.agent_count,
                    feature_path=str(feature_path),
                    teacher_loss=teacher_loss,
                    teacher_uncertainty=teacher_uncertainty,
                    metadata={
                        "source": "diffusiondrive_checkpoint_teacher_scoring",
                        "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
                        "teacher_signal_caveat": (
                            "teacher_loss is the summed scalar DiffusionDrive training loss "
                            "components for this sample; teacher_uncertainty is derived from "
                            "checkpoint classification-logit entropy captured by forward hooks. "
                            "The model is run in train mode under torch.no_grad() because the "
                            "planning loss path requires training-mode diffusion outputs."
                        ),
                        "checkpoint": str(checkpoint_path),
                        "config": str(config_path),
                        "split": split,
                        "loss_components": loss_components,
                        "map_location": sample.location,
                        "gt_ego_fut_cmd": _as_list(info.get("gt_ego_fut_cmd")),
                        "num_gt_boxes": sample.agent_count,
                    },
                )
                records.append(record)
                raw_losses.append(teacher_loss)
                raw_uncertainties.append(teacher_uncertainty)
                peak_memory_mb = max(peak_memory_mb, _cuda_peak_memory_mb(torch))
                if not feature_shapes:
                    feature_shapes = {
                        "traj_raw": list(sample.traj_raw.shape),
                        "traj_hidden": list(traj_hidden.shape),
                        "scene_hidden": list(scene_hidden.shape),
                        "interaction_hidden": list(interaction_hidden.shape),
                    }
                if log_interval and ((idx + 1) % log_interval == 0 or idx + 1 == total):
                    print(
                        json.dumps(
                            {
                                "processed": idx + 1,
                                "total": total,
                                "token": token,
                                "teacher_loss": round(teacher_loss, 6),
                                "teacher_uncertainty": round(teacher_uncertainty, 6),
                                "peak_gpu_mem_mb": peak_memory_mb,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            for handle in handles:
                handle.remove()

    manifest = Manifest.from_records(records)
    manifest.save(output / "manifest.jsonl")
    runtime_s = time.perf_counter() - start_wall
    stats = {
        "manifest": manifest.summary(),
        "feature_shapes": feature_shapes,
        "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
        "teacher_signal_caveat": (
            "Genuine DiffusionDrive checkpoint losses/logit entropy, but mini-scale "
            "teacher scores are not final model-quality metrics."
        ),
        "source_info": str(info_path),
        "source_info_metadata": info_metadata,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta_keys": sorted((checkpoint_meta.get("meta") or {}).keys()),
        "split": split,
        "max_samples": max_samples,
        "seed": seed,
        "runtime_s": runtime_s,
        "peak_gpu_memory_mb": peak_memory_mb,
        "teacher_loss_summary": _summary(raw_losses),
        "teacher_uncertainty_summary": _summary(raw_uncertainties),
    }
    write_json(output / "feature_stats.json", stats)
    write_json(
        output / "artifact_metadata.json",
        {
            **make_artifact_metadata(
                artifact_type="feature_cache",
                run_id="diffusiondrive_teacher_scoring",
                config={
                    "config": str(config_path),
                    "checkpoint": str(checkpoint_path),
                    "split": split,
                    "max_samples": max_samples,
                    "seed": seed,
                    "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
                },
                input_paths=[config_path, checkpoint_path, info_path],
                summary=stats["manifest"],
            ),
            "teacher_signal_caveat": stats["teacher_signal_caveat"],
            "runtime_s": runtime_s,
            "peak_gpu_memory_mb": peak_memory_mb,
        },
    )
    return stats


class _ActivationCapture:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._scene: np.ndarray | None = None
        self._det: np.ndarray | None = None
        self._map: np.ndarray | None = None
        self._traj: np.ndarray | None = None
        self._uncertainties: list[float] = []

    def register(self, model: Any) -> list[Any]:
        return [
            model.img_neck.register_forward_hook(self._img_neck_hook),
            model.head.det_head.register_forward_hook(self._det_head_hook),
            model.head.map_head.register_forward_hook(self._map_head_hook),
            model.head.motion_plan_head.register_forward_hook(self._motion_hook),
        ]

    def scene_hidden(self) -> np.ndarray:
        return _pad_or_trim(self._scene, 256)

    def interaction_hidden(self) -> np.ndarray:
        parts = [x for x in [self._det, self._map] if x is not None]
        if not parts:
            return np.zeros((256,), dtype=np.float32)
        return _pad_or_trim(np.mean(np.stack(parts), axis=0), 256)

    def traj_hidden(self, traj_raw: np.ndarray) -> np.ndarray:
        if self._traj is None:
            return _traj_summary(traj_raw)
        return _pad_or_trim(self._traj, 32)

    def uncertainty(self) -> float:
        if not self._uncertainties:
            return 0.0
        return float(np.clip(np.mean(self._uncertainties), 0.0, 1.0))

    def _img_neck_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        vectors = [_channel_vector(tensor, channel_dim=1) for tensor in _iter_tensors(output)]
        vectors = [v for v in vectors if v is not None and v.size]
        if vectors:
            self._scene = np.mean(np.stack([_pad_or_trim(v, 256) for v in vectors]), axis=0)

    def _det_head_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._det = _dict_feature(output, "instance_feature", 256)
        self._uncertainties.extend(_classification_uncertainties(output))

    def _map_head_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._map = _dict_feature(output, "instance_feature", 256)
        self._uncertainties.extend(_classification_uncertainties(output))

    def _motion_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._traj = _motion_traj_feature(output)
        self._uncertainties.extend(_classification_uncertainties(output))


def _prepare_diffusiondrive_imports(root: Path) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    importlib.import_module("projects.mmdet3d_plugin")


@contextmanager
def _working_directory(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _iter_tensors(value: Any):
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _channel_vector(tensor: Any, channel_dim: int = -1) -> np.ndarray | None:
    import torch

    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None
    value = tensor.detach().float()
    if value.dim() <= abs(channel_dim):
        return None
    if channel_dim < 0:
        channel_dim = value.dim() + channel_dim
    reduce_dims = [dim for dim in range(value.dim()) if dim != channel_dim]
    vector = value.mean(dim=reduce_dims).cpu().numpy()
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _dict_feature(output: Any, key: str, dim: int) -> np.ndarray | None:
    if not isinstance(output, dict) or key not in output:
        return None
    return _pad_or_trim(_channel_vector(output[key], channel_dim=-1), dim)


def _motion_traj_feature(output: Any) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for tensor in _iter_tensors(output):
        shape = tuple(tensor.shape)
        if len(shape) >= 2 and shape[-1] == 2 and shape[-2] == 6:
            arr = tensor.detach().float().cpu().numpy().reshape(-1, 6, 2)
            mean_traj = np.nan_to_num(arr.mean(axis=0), nan=0.0)
            delta = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), mean_traj]), axis=0)
            speed = np.linalg.norm(delta, axis=1)
            candidates.append(
                np.concatenate(
                    [
                        mean_traj.reshape(-1),
                        speed,
                        mean_traj.mean(axis=0),
                        mean_traj.std(axis=0),
                    ]
                ).astype(np.float32)
            )
    if not candidates:
        return None
    return _pad_or_trim(np.mean(np.stack([_pad_or_trim(v, 32) for v in candidates]), axis=0), 32)


def _classification_uncertainties(output: Any) -> list[float]:
    values: list[float] = []
    if isinstance(output, dict):
        for key, item in output.items():
            if "classification" in str(key):
                values.extend(_entropy_values(item))
    elif isinstance(output, (list, tuple)):
        for item in output:
            values.extend(_classification_uncertainties(item))
    return values


def _entropy_values(value: Any) -> list[float]:
    import torch

    values: list[float] = []
    if torch.is_tensor(value):
        if value.numel() == 0:
            return values
        logits = value.detach().float()
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)) / math.log(2.0)
        values.append(float(entropy.mean().cpu()))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(_entropy_values(item))
    elif isinstance(value, dict):
        for item in value.values():
            values.extend(_entropy_values(item))
    return values


def _loss_components(losses: dict[str, Any]) -> dict[str, float]:
    components: dict[str, float] = {}
    for key, value in losses.items():
        tensors = list(_iter_tensors(value))
        if not tensors:
            continue
        components[key] = float(sum(t.detach().float().mean().cpu().item() for t in tensors))
    return components


def _pad_or_trim(value: np.ndarray | None, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out[: min(dim, arr.size)] = arr[:dim]
    return out


def _cuda_peak_memory_mb(torch_module: Any) -> int:
    if not torch_module.cuda.is_available():
        return 0
    return int(torch_module.cuda.max_memory_allocated() / (1024 * 1024))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }

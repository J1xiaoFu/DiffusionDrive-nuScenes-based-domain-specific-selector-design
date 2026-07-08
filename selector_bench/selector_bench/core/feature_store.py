from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.manifest import Manifest, ManifestRecord


@dataclass
class FeatureRecord:
    token: str
    traj_raw: np.ndarray
    scene_hidden: np.ndarray
    interaction_hidden: np.ndarray
    domain_id: int
    teacher_loss: float
    teacher_uncertainty: float
    traj_hidden: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class FeatureStore:
    def __init__(
        self,
        manifest: Manifest,
        feature_root: str | Path,
        cache: bool = False,
    ):
        self.manifest = manifest
        self.feature_root = Path(feature_root)
        self.cache = cache
        self._cache: dict[str, FeatureRecord] = {}

    @classmethod
    def from_feature_dir(cls, feature_dir: str | Path, cache: bool = False) -> "FeatureStore":
        root = Path(feature_dir)
        manifest = Manifest.load(root / "manifest.jsonl")
        return cls(manifest=manifest, feature_root=root, cache=cache)

    def _path_for(self, record: ManifestRecord) -> Path:
        path = Path(record.feature_path)
        if path.is_absolute():
            return path
        return self.feature_root / path

    def get(self, token: str) -> FeatureRecord:
        if token in self._cache:
            return self._cache[token]

        manifest_record = self.manifest.get(token)
        path = self._path_for(manifest_record)
        if not path.exists():
            raise FileNotFoundError(f"Feature file does not exist: {path}")

        with np.load(path) as data:
            record = FeatureRecord(
                token=token,
                traj_raw=np.asarray(data["traj_raw"], dtype=np.float32),
                scene_hidden=np.asarray(data["scene_hidden"], dtype=np.float32),
                interaction_hidden=np.asarray(data["interaction_hidden"], dtype=np.float32),
                traj_hidden=np.asarray(data["traj_hidden"], dtype=np.float32)
                if "traj_hidden" in data
                else None,
                domain_id=int(data["domain_id"]) if "domain_id" in data else manifest_record.domain_id,
                teacher_loss=float(data["teacher_loss"])
                if "teacher_loss" in data
                else manifest_record.teacher_loss,
                teacher_uncertainty=float(data["teacher_uncertainty"])
                if "teacher_uncertainty" in data
                else manifest_record.teacher_uncertainty,
                metadata=manifest_record.metadata,
            )

        self._validate(record, path)
        if self.cache:
            self._cache[token] = record
        return record

    def get_many(self, tokens: list[str]) -> dict[str, np.ndarray]:
        records = [self.get(token) for token in tokens]
        return {
            "traj_raw": np.stack([record.traj_raw for record in records]),
            "scene_hidden": np.stack([record.scene_hidden for record in records]),
            "interaction_hidden": np.stack([record.interaction_hidden for record in records]),
            "domain_id": np.asarray([record.domain_id for record in records], dtype=np.int64),
            "teacher_loss": np.asarray([record.teacher_loss for record in records], dtype=np.float32),
            "teacher_uncertainty": np.asarray(
                [record.teacher_uncertainty for record in records], dtype=np.float32
            ),
        }

    @staticmethod
    def _validate(record: FeatureRecord, path: Path) -> None:
        arrays = [record.traj_raw, record.scene_hidden, record.interaction_hidden]
        if record.traj_hidden is not None:
            arrays.append(record.traj_hidden)
        for array in arrays:
            if not np.isfinite(array).all():
                raise ValueError(f"Feature file contains NaN/Inf values: {path}")
        if record.scene_hidden.ndim != 1:
            raise ValueError(f"scene_hidden must be 1-D: {path}")
        if record.interaction_hidden.ndim != 1:
            raise ValueError(f"interaction_hidden must be 1-D: {path}")

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.feature_store import FeatureStore
from selector_bench.core.manifest import Manifest
from selector_bench.utils.hashing import stable_unit_float
from selector_bench.utils.io import read_json


class ScoreProvider(ABC):
    name: str

    @abstractmethod
    def score(self, tokens: list[str], domain_id: int | None = None) -> np.ndarray:
        raise NotImplementedError


class RandomScoreProvider(ScoreProvider):
    name = "random"

    def __init__(self, seed: int = 1):
        self.seed = seed

    def score(self, tokens: list[str], domain_id: int | None = None) -> np.ndarray:
        return np.asarray(
            [stable_unit_float(f"{domain_id}:{token}", self.seed) for token in tokens],
            dtype=np.float32,
        )


class _ManifestScoreProvider(ScoreProvider):
    field: str

    def __init__(self, manifest: Manifest):
        self.manifest = manifest

    def score(self, tokens: list[str], domain_id: int | None = None) -> np.ndarray:
        return np.asarray(
            [float(getattr(self.manifest.get(token), self.field)) for token in tokens],
            dtype=np.float32,
        )


class TeacherLossScoreProvider(_ManifestScoreProvider):
    name = "teacher_loss"
    field = "teacher_loss"


class TeacherUncertaintyScoreProvider(_ManifestScoreProvider):
    name = "teacher_uncertainty"
    field = "teacher_uncertainty"


class MosaicLiteOriginalScoreProvider(ScoreProvider):
    name = "mosaic_lite_original"

    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        train = manifest.filter(split="train")
        self._loss_min, self._loss_span = self._range([r.teacher_loss for r in train])
        self._unc_min, self._unc_span = self._range([r.teacher_uncertainty for r in train])
        self._agent_min, self._agent_span = self._range([r.agent_count for r in train])

    def score(self, tokens: list[str], domain_id: int | None = None) -> np.ndarray:
        scores: list[float] = []
        for token in tokens:
            record = self.manifest.get(token)
            loss = (record.teacher_loss - self._loss_min) / self._loss_span
            uncertainty = (record.teacher_uncertainty - self._unc_min) / self._unc_span
            agent = (record.agent_count - self._agent_min) / self._agent_span
            scores.append(0.60 * loss + 0.30 * uncertainty + 0.10 * agent)
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def _range(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        lo = float(min(values))
        hi = float(max(values))
        return lo, max(hi - lo, 1e-8)


class OurSelectorScoreProvider(ScoreProvider):
    name = "ours"

    def __init__(
        self,
        manifest: Manifest,
        feature_store: FeatureStore,
        selector_checkpoint: str | Path,
    ):
        self.manifest = manifest
        self.feature_store = feature_store
        checkpoint = read_json(selector_checkpoint)
        self.selector = checkpoint.get("selector", checkpoint)
        self.weights = self.selector.get("weights", {})
        self.domain_bias = self.selector.get("domain_bias", {})

    def score(self, tokens: list[str], domain_id: int | None = None) -> np.ndarray:
        values: list[float] = []
        for token in tokens:
            record = self.manifest.get(token)
            features = self.feature_store.get(token)
            summary = self._summary_features(record, features)
            score = float(self.weights.get("bias", 0.0))
            for key, value in summary.items():
                score += float(self.weights.get(key, 0.0)) * value
            score += float(self.domain_bias.get(str(record.domain_id), 0.0))
            values.append(score)
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _summary_features(record: Any, features: Any) -> dict[str, float]:
        traj = features.traj_raw
        speed_col = traj[:, 2] if traj.ndim == 2 and traj.shape[1] > 2 else traj.reshape(-1)
        return {
            "scene_norm": float(np.linalg.norm(features.scene_hidden) / np.sqrt(features.scene_hidden.size)),
            "interaction_norm": float(
                np.linalg.norm(features.interaction_hidden)
                / np.sqrt(features.interaction_hidden.size)
            ),
            "traj_speed_mean": float(np.mean(np.abs(speed_col))),
            "teacher_loss": float(record.teacher_loss),
            "teacher_uncertainty": float(record.teacher_uncertainty),
            "agent_count": float(record.agent_count) / 32.0,
            "ego_speed": float(record.ego_speed) / 30.0,
        }

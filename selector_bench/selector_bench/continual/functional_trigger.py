"""Functional sentinel monitoring for factorized continual-driving repair.

The sentinel is an audit instrument, never a replay buffer.  This module is
therefore deliberately independent of model training: it consumes paired,
session-level functional measurements and emits the three switches used by
FTF-OPD.  Representation drift is intentionally absent from the interface.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class FunctionalTriggerError(ValueError):
    """Raised when sentinel measurements cannot support a paired decision."""


@dataclass(frozen=True)
class SentinelRecord:
    session_id: str
    stage_index: int
    location: str
    risk_stratum: str


@dataclass(frozen=True)
class SentinelSnapshot:
    """Normalized risks for exactly the same ordered sentinel sessions.

    All risks use the convention that larger is worse.  Planning and
    perception risks are required.  Collision is optional because some
    development evaluators do not expose it at every audit.  The two mode
    risks are paired counterfactual evaluations on the same labeled sessions:
    the current model's selected mode and the frozen teacher's selected mode.
    """

    session_ids: tuple[str, ...]
    planning_risk: tuple[float, ...]
    perception_risk: tuple[float, ...]
    collision_risk: tuple[float, ...] | None = None
    student_mode_risk: tuple[float, ...] | None = None
    teacher_mode_risk: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.session_ids or len(set(self.session_ids)) != len(self.session_ids):
            raise FunctionalTriggerError("sentinel session IDs must be non-empty and unique")
        expected = len(self.session_ids)
        for name in (
            "planning_risk",
            "perception_risk",
            "collision_risk",
            "student_mode_risk",
            "teacher_mode_risk",
        ):
            values = getattr(self, name)
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (expected,) or not np.isfinite(array).all():
                raise FunctionalTriggerError(
                    f"{name} must contain one finite value per sentinel session"
                )
        if (self.student_mode_risk is None) != (self.teacher_mode_risk is None):
            raise FunctionalTriggerError("student and teacher mode risks must be supplied together")


@dataclass(frozen=True)
class FunctionalTriggerConfig:
    planning_mde: float = 0.03
    perception_mde: float = 0.03
    collision_mde: float = 0.01
    mode_mde: float = 0.03
    confidence: float = 0.95
    bootstrap_replicates: int = 2000
    consecutive_breaches: int = 2
    consecutive_releases: int = 2
    release_fraction: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        mdes = (
            self.planning_mde,
            self.perception_mde,
            self.collision_mde,
            self.mode_mde,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in mdes):
            raise FunctionalTriggerError("minimum detectable effects must be finite and non-negative")
        if not 0.5 < self.confidence < 1.0:
            raise FunctionalTriggerError("confidence must lie strictly between 0.5 and 1")
        if self.bootstrap_replicates < 100:
            raise FunctionalTriggerError("at least 100 bootstrap replicates are required")
        if self.consecutive_breaches <= 0 or self.consecutive_releases <= 0:
            raise FunctionalTriggerError("hysteresis counts must be positive")
        if not 0.0 < self.release_fraction < 1.0:
            raise FunctionalTriggerError("release fraction must lie strictly between zero and one")


@dataclass(frozen=True)
class EffectInterval:
    mean: float
    lower: float
    upper: float
    mde: float
    breach: bool
    release: bool


@dataclass(frozen=True)
class TriggerDecision:
    audit_index: int
    perception_drift: bool
    planning_drift: bool
    mode_harm: bool
    intervals: Mapping[str, EffectInterval]
    breach_streaks: Mapping[str, int]
    release_streaks: Mapping[str, int]

    def loss_switches(self) -> dict[str, float]:
        """Return the exact factor switches consumed by the training arm."""

        return {
            "lambda_perception_switch": float(self.perception_drift),
            "lambda_planning_switch": float(self.planning_drift),
            "mode_kl_switch": float(self.planning_drift and self.mode_harm),
        }

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "selector_bench.ftf_opd_trigger_decision.v1",
            "audit_index": self.audit_index,
            "perception_drift": self.perception_drift,
            "planning_drift": self.planning_drift,
            "mode_harm": self.mode_harm,
            "intervals": {
                name: asdict(interval) for name, interval in self.intervals.items()
            },
            "breach_streaks": dict(self.breach_streaks),
            "release_streaks": dict(self.release_streaks),
            "loss_switches": self.loss_switches(),
        }


def select_stratified_sentinel(
    records: Sequence[SentinelRecord],
    *,
    fraction: float = 0.02,
    seed: int = 0,
) -> tuple[SentinelRecord, ...]:
    """Select exactly ceil(fraction*N) whole sessions by balanced round-robin.

    The strata are previous stage, location and risk.  Hash ranking makes the
    result invariant to input order without treating individual frames as
    independently sampleable units.
    """

    if not records or not 0.0 < fraction <= 1.0:
        raise FunctionalTriggerError("sentinel selection needs records and a fraction in (0,1]")
    ids = [record.session_id for record in records]
    if len(ids) != len(set(ids)):
        raise FunctionalTriggerError("sentinel candidates contain duplicate sessions")
    target = min(len(records), max(1, math.ceil(len(records) * fraction)))
    groups: dict[tuple[int, str, str], list[SentinelRecord]] = {}
    for record in records:
        if not record.session_id or not record.location or not record.risk_stratum:
            raise FunctionalTriggerError("sentinel strata must be non-empty")
        key = (record.stage_index, record.location, record.risk_stratum)
        groups.setdefault(key, []).append(record)

    def rank(record: SentinelRecord) -> bytes:
        return hashlib.sha256(f"{seed}\0{record.session_id}".encode()).digest()

    for values in groups.values():
        values.sort(key=lambda record: (rank(record), record.session_id))
    ordered_groups = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
    )
    selected: list[SentinelRecord] = []
    offset = 0
    while len(selected) < target:
        progressed = False
        for key in ordered_groups:
            values = groups[key]
            if offset < len(values):
                selected.append(values[offset])
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
        offset += 1
    return tuple(sorted(selected, key=lambda record: record.session_id))


def paired_bootstrap_interval(
    deltas: Sequence[float],
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return mean and one-sided lower/upper bootstrap confidence bounds."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise FunctionalTriggerError("paired bootstrap needs at least two finite deltas")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[samples].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(values.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, confidence)),
    )


@dataclass
class FunctionalTriggerController:
    config: FunctionalTriggerConfig = FunctionalTriggerConfig()
    audit_index: int = 0
    active: dict[str, bool] = field(
        default_factory=lambda: {"perception": False, "planning": False, "mode": False}
    )
    breach_streaks: dict[str, int] = field(
        default_factory=lambda: {"perception": 0, "planning": 0, "mode": 0}
    )
    release_streaks: dict[str, int] = field(
        default_factory=lambda: {"perception": 0, "planning": 0, "mode": 0}
    )

    def _interval(self, deltas: np.ndarray, mde: float, label: str) -> EffectInterval:
        mean, lower, upper = paired_bootstrap_interval(
            deltas,
            confidence=self.config.confidence,
            replicates=self.config.bootstrap_replicates,
            seed=self.config.seed + 1009 * self.audit_index + stable_label_seed(label),
        )
        return EffectInterval(
            mean=mean,
            lower=lower,
            upper=upper,
            mde=mde,
            breach=lower > mde,
            release=upper < mde * self.config.release_fraction,
        )

    def _advance(self, name: str, *, breach: bool, release: bool) -> None:
        if self.active[name]:
            self.breach_streaks[name] = 0
            self.release_streaks[name] = self.release_streaks[name] + 1 if release else 0
            if self.release_streaks[name] >= self.config.consecutive_releases:
                self.active[name] = False
                self.release_streaks[name] = 0
        else:
            self.release_streaks[name] = 0
            self.breach_streaks[name] = self.breach_streaks[name] + 1 if breach else 0
            if self.breach_streaks[name] >= self.config.consecutive_breaches:
                self.active[name] = True
                self.breach_streaks[name] = 0

    def observe(
        self,
        baseline: SentinelSnapshot,
        current: SentinelSnapshot,
    ) -> TriggerDecision:
        if baseline.session_ids != current.session_ids:
            raise FunctionalTriggerError("baseline/current sentinel sessions must align exactly")
        self.audit_index += 1
        intervals: dict[str, EffectInterval] = {}
        planning = np.asarray(current.planning_risk) - np.asarray(baseline.planning_risk)
        perception = np.asarray(current.perception_risk) - np.asarray(
            baseline.perception_risk
        )
        intervals["planning"] = self._interval(
            planning, self.config.planning_mde, "planning"
        )
        intervals["perception"] = self._interval(
            perception, self.config.perception_mde, "perception"
        )
        planning_breach = intervals["planning"].breach
        planning_release = intervals["planning"].release
        if baseline.collision_risk is not None or current.collision_risk is not None:
            if baseline.collision_risk is None or current.collision_risk is None:
                raise FunctionalTriggerError("collision risk must exist in both snapshots")
            collision = np.asarray(current.collision_risk) - np.asarray(
                baseline.collision_risk
            )
            intervals["collision"] = self._interval(
                collision, self.config.collision_mde, "collision"
            )
            planning_breach = planning_breach or intervals["collision"].breach
            planning_release = planning_release and intervals["collision"].release
        self._advance(
            "planning", breach=planning_breach, release=planning_release
        )
        self._advance(
            "perception",
            breach=intervals["perception"].breach,
            release=intervals["perception"].release,
        )

        if current.student_mode_risk is not None:
            mode_delta = np.asarray(current.student_mode_risk) - np.asarray(
                current.teacher_mode_risk
            )
            intervals["mode"] = self._interval(
                mode_delta, self.config.mode_mde, "mode"
            )
            self._advance(
                "mode",
                breach=intervals["mode"].breach,
                release=intervals["mode"].release,
            )
        else:
            self.active["mode"] = False
            self.breach_streaks["mode"] = 0
            self.release_streaks["mode"] = 0

        return TriggerDecision(
            audit_index=self.audit_index,
            perception_drift=self.active["perception"],
            planning_drift=self.active["planning"],
            mode_harm=self.active["mode"] and self.active["planning"],
            intervals=intervals,
            breach_streaks=dict(self.breach_streaks),
            release_streaks=dict(self.release_streaks),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "selector_bench.ftf_opd_trigger_state.v1",
            "config": asdict(self.config),
            "audit_index": self.audit_index,
            "active": dict(self.active),
            "breach_streaks": dict(self.breach_streaks),
            "release_streaks": dict(self.release_streaks),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "FunctionalTriggerController":
        if payload.get("schema") != "selector_bench.ftf_opd_trigger_state.v1":
            raise FunctionalTriggerError("unsupported functional-trigger state schema")
        controller = cls(config=FunctionalTriggerConfig(**dict(payload["config"])))
        controller.audit_index = int(payload["audit_index"])
        for field_name in ("active", "breach_streaks", "release_streaks"):
            values = dict(payload[field_name])
            if set(values) != {"perception", "planning", "mode"}:
                raise FunctionalTriggerError(f"invalid {field_name} keys")
            setattr(controller, field_name, values)
        return controller


def stable_label_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")

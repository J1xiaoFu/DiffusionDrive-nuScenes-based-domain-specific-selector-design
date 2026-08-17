"""Measured Drive-CL run budgets derived from batches actually consumed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TRANSIENT_STORAGE_SEMANTICS = (
    "maximum observed byte size of an atomic temporary artifact before final result "
    "receipt serialization"
)


class BudgetMeasurementError(ValueError):
    """Raised when a loader yields an identity outside the frozen protocol."""


@dataclass
class RunBudgetCounter:
    current_identities: frozenset[str]
    old_identities: frozenset[str]
    current_seen: set[str] = field(default_factory=set)
    old_seen: set[str] = field(default_factory=set)
    current_presentations: int = 0
    old_presentations: int = 0
    optimizer_updates: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    student_queries: int = 0
    teacher_queries: int = 0

    def observe_batch(self, identities: Iterable[str]) -> None:
        for raw in identities:
            identity = str(raw)
            in_current = identity in self.current_identities
            in_old = identity in self.old_identities
            if in_current == in_old:
                raise BudgetMeasurementError(
                    "consumed identity must belong to exactly one frozen current/old set: "
                    + identity
                )
            if in_current:
                self.current_seen.add(identity)
                self.current_presentations += 1
            else:
                self.old_seen.add(identity)
                self.old_presentations += 1

    def observe_optimizer_step(
        self,
        *,
        forward_calls: int,
        backward_calls: int,
        student_queries: int,
        teacher_queries: int,
    ) -> None:
        values = (forward_calls, backward_calls, student_queries, teacher_queries)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise BudgetMeasurementError("runtime budget increments must be non-negative integers")
        self.optimizer_updates += 1
        self.forward_calls += forward_calls
        self.backward_calls += backward_calls
        self.student_queries += student_queries
        self.teacher_queries += teacher_queries

    def snapshot(self, *, completed_epochs: int) -> dict[str, int]:
        if completed_epochs < 0:
            raise BudgetMeasurementError("completed epochs may not be negative")
        return {
            "epochs": int(completed_epochs),
            "optimizer_updates": self.optimizer_updates,
            "current_unique_identities": len(self.current_seen),
            "old_unique_identities": len(self.old_seen),
            "current_presentations": self.current_presentations,
            "old_presentations": self.old_presentations,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "student_queries": self.student_queries,
            "teacher_queries": self.teacher_queries,
        }


@dataclass
class TransientStorageCounter:
    """Peak bytes of an atomic temporary artifact observed before replacement."""

    peak_atomic_temporary_bytes: int = 0

    def observe_temporary(self, path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise BudgetMeasurementError("atomic temporary artifact is missing or symlinked")
        self.peak_atomic_temporary_bytes = max(
            self.peak_atomic_temporary_bytes, int(path.stat().st_size)
        )

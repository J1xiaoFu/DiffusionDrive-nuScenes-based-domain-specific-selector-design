"""Update-level scheduling and sidecar handshake for online functional audits.

The training process owns model/optimizer state.  A dataset-specific evaluator
runs as a sidecar: at each fractional-epoch boundary the trainer publishes a
request containing a read-only student checkpoint, then waits for one
functional-trigger decision.  This keeps sentinel examples outside the
training loader and allows NAVSIM and nuScenes to use different evaluators.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class OnlineAuditError(RuntimeError):
    """Raised when an online audit schedule or sidecar response is invalid."""


@dataclass(frozen=True)
class FractionalEpochAuditSchedule:
    updates_per_epoch: int
    fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.updates_per_epoch <= 0:
            raise OnlineAuditError("updates_per_epoch must be positive")
        reciprocal = 1.0 / self.fraction if self.fraction > 0.0 else math.inf
        if (
            not math.isfinite(self.fraction)
            or not 0.0 < self.fraction <= 1.0
            or not math.isclose(reciprocal, round(reciprocal), abs_tol=1e-9)
        ):
            raise OnlineAuditError("audit fraction must evenly divide one epoch")
        if self.updates_per_epoch < round(reciprocal):
            raise OnlineAuditError(
                "an epoch needs at least one optimizer update per audit window"
            )

    @property
    def audits_per_epoch(self) -> int:
        return round(1.0 / self.fraction)

    def boundaries(self) -> tuple[int, ...]:
        """One-indexed optimizer-update boundaries, including epoch end."""

        return tuple(
            math.ceil(self.updates_per_epoch * index / self.audits_per_epoch)
            for index in range(1, self.audits_per_epoch + 1)
        )

    def boundary_index(self, completed_updates: int) -> int | None:
        if completed_updates <= 0:
            return None
        try:
            return self.boundaries().index(completed_updates) + 1
        except ValueError:
            return None


def wait_for_sidecar_decision(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 1.0,
    loader: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Wait for one atomic JSON decision emitted by the sentinel sidecar."""

    if timeout_seconds <= 0.0 or poll_seconds <= 0.0:
        raise OnlineAuditError("sidecar timeout and poll interval must be positive")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return loader(path)
            except (json.JSONDecodeError, OSError):
                # The writer may still be completing a non-atomic move.  The
                # reference sidecar uses an atomic rename, but tolerate one
                # partially visible poll from external evaluators.
                pass
        time.sleep(poll_seconds)
    raise OnlineAuditError(f"timed out waiting for functional audit decision: {path}")

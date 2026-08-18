from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selector_bench.continual.online_audit import (
    FractionalEpochAuditSchedule,
    OnlineAuditError,
    wait_for_sidecar_decision,
)


class OnlineAuditScheduleTests(unittest.TestCase):
    def test_quarter_epoch_boundaries_cover_uneven_epochs(self) -> None:
        self.assertEqual(
            FractionalEpochAuditSchedule(10, 0.25).boundaries(),
            (3, 5, 8, 10),
        )
        schedule = FractionalEpochAuditSchedule(7, 0.25)
        self.assertEqual(schedule.boundaries(), (2, 4, 6, 7))
        self.assertEqual(schedule.boundary_index(4), 2)
        self.assertIsNone(schedule.boundary_index(3))

    def test_invalid_fraction_is_rejected(self) -> None:
        for fraction in (0.0, -0.25, 0.3, 1.1):
            with self.assertRaises(OnlineAuditError):
                FractionalEpochAuditSchedule(10, fraction)
        with self.assertRaises(OnlineAuditError):
            FractionalEpochAuditSchedule(3, 0.25)

    def test_sidecar_decision_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decision.json"
            path.write_text(json.dumps({"value": 7}))
            observed = wait_for_sidecar_decision(
                path,
                timeout_seconds=0.1,
                poll_seconds=0.01,
                loader=lambda item: json.loads(item.read_text()),
            )
            self.assertEqual(observed, {"value": 7})


if __name__ == "__main__":
    unittest.main()

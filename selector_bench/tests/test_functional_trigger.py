from __future__ import annotations

import unittest

from selector_bench.continual.functional_trigger import (
    FunctionalTriggerConfig,
    FunctionalTriggerController,
    FunctionalTriggerError,
    SentinelRecord,
    SentinelSnapshot,
    select_stratified_sentinel,
)


def snapshot(
    planning: float,
    perception: float,
    *,
    collision: float | None = None,
    student_mode: float | None = None,
    teacher_mode: float | None = None,
    count: int = 24,
) -> SentinelSnapshot:
    return SentinelSnapshot(
        session_ids=tuple(f"session-{index:03d}" for index in range(count)),
        planning_risk=(planning,) * count,
        perception_risk=(perception,) * count,
        collision_risk=(collision,) * count if collision is not None else None,
        student_mode_risk=(student_mode,) * count if student_mode is not None else None,
        teacher_mode_risk=(teacher_mode,) * count if teacher_mode is not None else None,
    )


class FunctionalTriggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FunctionalTriggerConfig(
            planning_mde=0.03,
            perception_mde=0.03,
            collision_mde=0.01,
            mode_mde=0.03,
            bootstrap_replicates=400,
            seed=17,
        )

    def test_stratified_sentinel_is_exact_and_order_invariant(self) -> None:
        records = [
            SentinelRecord(
                session_id=f"s-{index:03d}",
                stage_index=index % 2,
                location=("boston" if index % 3 else "singapore"),
                risk_stratum=("high" if index % 5 == 0 else "normal"),
            )
            for index in range(100)
        ]
        forward = select_stratified_sentinel(records, fraction=0.02, seed=5)
        reverse = select_stratified_sentinel(list(reversed(records)), fraction=0.02, seed=5)
        self.assertEqual(len(forward), 2)
        self.assertEqual(forward, reverse)

    def test_two_breaches_activate_and_two_releases_deactivate(self) -> None:
        controller = FunctionalTriggerController(self.config)
        baseline = snapshot(0.20, 0.20)
        first = controller.observe(baseline, snapshot(0.28, 0.20))
        self.assertFalse(first.planning_drift)
        second = controller.observe(baseline, snapshot(0.28, 0.20))
        self.assertTrue(second.planning_drift)

        first_release = controller.observe(baseline, snapshot(0.20, 0.20))
        self.assertTrue(first_release.planning_drift)
        second_release = controller.observe(baseline, snapshot(0.20, 0.20))
        self.assertFalse(second_release.planning_drift)

    def test_perception_and_collision_trigger_separate_factors(self) -> None:
        controller = FunctionalTriggerController(self.config)
        baseline = snapshot(0.20, 0.20, collision=0.01)
        for _ in range(2):
            decision = controller.observe(
                baseline,
                snapshot(0.20, 0.27, collision=0.04),
            )
        self.assertTrue(decision.perception_drift)
        self.assertTrue(decision.planning_drift)
        self.assertEqual(
            decision.loss_switches(),
            {
                "lambda_perception_switch": 1.0,
                "lambda_planning_switch": 1.0,
                "mode_kl_switch": 0.0,
            },
        )

    def test_mode_kl_requires_teacher_advantage_and_planning_drift(self) -> None:
        baseline = snapshot(
            0.20,
            0.20,
            student_mode=0.20,
            teacher_mode=0.20,
        )
        controller = FunctionalTriggerController(self.config)
        for _ in range(2):
            decision = controller.observe(
                baseline,
                snapshot(
                    0.27,
                    0.20,
                    student_mode=0.30,
                    teacher_mode=0.20,
                ),
            )
        self.assertTrue(decision.planning_drift)
        self.assertTrue(decision.mode_harm)
        self.assertEqual(decision.loss_switches()["mode_kl_switch"], 1.0)

        no_teacher_advantage = FunctionalTriggerController(self.config)
        for _ in range(2):
            decision = no_teacher_advantage.observe(
                baseline,
                snapshot(
                    0.27,
                    0.20,
                    student_mode=0.18,
                    teacher_mode=0.20,
                ),
            )
        self.assertTrue(decision.planning_drift)
        self.assertFalse(decision.mode_harm)

    def test_state_roundtrip_preserves_hysteresis(self) -> None:
        baseline = snapshot(0.20, 0.20)
        controller = FunctionalTriggerController(self.config)
        controller.observe(baseline, snapshot(0.27, 0.20))
        restored = FunctionalTriggerController.from_state_dict(controller.state_dict())
        decision = restored.observe(baseline, snapshot(0.27, 0.20))
        self.assertTrue(decision.planning_drift)
        self.assertEqual(decision.audit_index, 2)

    def test_misaligned_or_nonfinite_measurements_are_rejected(self) -> None:
        baseline = snapshot(0.20, 0.20)
        current = SentinelSnapshot(
            session_ids=tuple(reversed(baseline.session_ids)),
            planning_risk=baseline.planning_risk,
            perception_risk=baseline.perception_risk,
        )
        with self.assertRaises(FunctionalTriggerError):
            FunctionalTriggerController(self.config).observe(baseline, current)
        with self.assertRaises(FunctionalTriggerError):
            snapshot(float("nan"), 0.20)


if __name__ == "__main__":
    unittest.main()

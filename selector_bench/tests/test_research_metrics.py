from __future__ import annotations

import unittest

from selector_bench.continual.research_metrics import (
    continual_driving_summary,
    pareto_front,
    support_condition_curve,
    trigger_quality,
)


class ResearchMetricsTest(unittest.TestCase):
    def test_continual_summary_exposes_worst_stage_and_plasticity(self) -> None:
        result = continual_driving_summary(
            [[0.8, 0.2, 0.1], [0.7, 0.9, 0.3], [0.6, 0.85, 0.95]],
            untrained_domain_scores=[0.1, 0.1, 0.1],
        )
        self.assertAlmostEqual(result["final_worst_old_stage"], 0.6)
        self.assertAlmostEqual(result["final_old_stage_average"], 0.725)
        self.assertAlmostEqual(result["last_stage_plasticity"], 0.95)

    def test_trigger_quality_reports_episode_delay(self) -> None:
        result = trigger_quality(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [False, False, True, True, False],
            [False, True, True, True, False],
        )
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(result["mean_detection_delay_epoch"], 0.25)

    def test_support_curve_and_pareto(self) -> None:
        curve = support_condition_curve(
            [0.2, 0.3, 0.92, 0.95],
            [0.6, 0.62, 0.80, 0.82],
            [0.7, 0.70, 0.81, 0.81],
            bootstrap_repetitions=200,
            seed=4,
        )
        self.assertEqual(sum(int(item["count"]) for item in curve), 4)
        self.assertFalse(curve[0]["noninferior_at_margin"])
        self.assertTrue(curve[-1]["noninferior_at_margin"])
        self.assertEqual(
            pareto_front({"stable": (0.9, 0.6), "plastic": (0.7, 0.9), "bad": (0.6, 0.5)}),
            ("plastic", "stable"),
        )


if __name__ == "__main__":
    unittest.main()

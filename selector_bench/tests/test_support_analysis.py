from __future__ import annotations

import unittest

import numpy as np

from selector_bench.continual.support_analysis import (
    SupportAnalysisError,
    analyze_support,
    gradient_conflict,
    knn_support_coverage,
    rbf_mmd_squared,
)


class SupportAnalysisTest(unittest.TestCase):
    def test_overlap_is_high_and_shifted_support_is_low(self) -> None:
        rng = np.random.default_rng(7)
        current = rng.normal(size=(80, 5))
        old_overlap = current[:40] + rng.normal(scale=0.02, size=(40, 5))
        old_shifted = current[:40] + 12.0
        high, _ = knn_support_coverage(old_overlap, current, k=5)
        low, _ = knn_support_coverage(old_shifted, current, k=5)
        self.assertGreater(high, 0.9)
        self.assertLess(low, 0.1)

    def test_mmd_and_gradient_conflict_have_expected_direction(self) -> None:
        rng = np.random.default_rng(11)
        features = rng.normal(size=(60, 4))
        self.assertAlmostEqual(rbf_mmd_squared(features, features), 0.0, places=12)
        self.assertGreater(rbf_mmd_squared(features, features + 5.0), 0.1)
        cosine, angle = gradient_conflict(np.ones(8), np.ones(8))
        self.assertAlmostEqual(cosine, 1.0)
        self.assertLess(angle, 2e-6)
        cosine, angle = gradient_conflict(np.ones(8), -np.ones(8))
        self.assertAlmostEqual(cosine, -1.0)
        self.assertLess(abs(angle - 180.0), 2e-6)

    def test_combined_analysis_and_invalid_gradient(self) -> None:
        rng = np.random.default_rng(3)
        current = rng.normal(size=(30, 3))
        old = current + 0.01
        result = analyze_support(old, current, np.ones(6), np.ones(6), k=3)
        self.assertGreater(result.old_covered_by_current, 0.9)
        self.assertLess(result.gradient_angle_degrees, 1e-6)
        with self.assertRaises(SupportAnalysisError):
            gradient_conflict(np.zeros(3), np.ones(3))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from selector_bench.continual.statistics import (
    continual_matrix_metrics,
    paired_log_bootstrap,
)


class DriveCLStatisticsTest(unittest.TestCase):
    def test_log_cluster_bootstrap_pairs_tokens(self) -> None:
        a = {"a": {"score": 0.5}, "b": {"score": 0.6}, "c": {"score": 0.4}}
        b = {"a": {"score": 0.6}, "b": {"score": 0.7}, "c": {"score": 0.5}}
        mapping = {
            "a": "2021.05.01.00.00.00_veh-01_00000_00001",
            "b": "2021.05.01.00.00.00_veh-01_00000_00001",
            "c": "2021.05.02.00.00.00_veh-02_00000_00001",
        }
        result = paired_log_bootstrap(
            a, b, mapping, metrics=("score",), repetitions=200, seed=4
        )
        self.assertAlmostEqual(result["score"]["difference_b_minus_a"], 0.1)
        self.assertEqual(result["score"]["log_cluster_count"], 2.0)

    def test_continual_matrix_metrics(self) -> None:
        result = continual_matrix_metrics(
            [[0.8, 0.2, 0.1], [0.7, 0.9, 0.3], [0.6, 0.85, 0.95]],
            untrained_domain_scores=[0.1, 0.1, 0.1],
        )
        self.assertAlmostEqual(result["backward_transfer"], -0.125)
        self.assertAlmostEqual(result["forward_transfer"], 0.15)


if __name__ == "__main__":
    unittest.main()

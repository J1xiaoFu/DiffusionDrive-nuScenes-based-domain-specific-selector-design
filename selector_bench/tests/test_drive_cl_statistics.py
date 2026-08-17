from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from selector_bench.continual.statistics import (
    PDM_DEFAULT_METRICS,
    StatisticsError,
    complete_holm_family,
    continual_matrix_metrics,
    hierarchical_seed_session_bootstrap,
    holm_adjusted_pvalues,
    paired_log_bootstrap,
    paired_session_bootstrap,
    read_pdm_rows,
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

    def test_session_bootstrap_merges_segmented_logs(self) -> None:
        a = {key: {"score": 0.5} for key in ("a", "b", "c")}
        b = {key: {"score": 0.6} for key in ("a", "b", "c")}
        mapping = {
            "a": "2021.05.01.00.00.00_veh-01_00000_00001",
            "b": "2021.05.01.00.00.00_veh-01_00002_00003",
            "c": "2021.05.02.00.00.00_veh-02_00000_00001",
        }
        result = paired_session_bootstrap(
            a, b, mapping, metrics=("score",), repetitions=200, seed=4
        )
        self.assertEqual(result["score"]["session_cluster_count"], 2.0)

    def test_pdm_reader_fails_closed_on_invalid_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pdm.csv"
            path.write_text("token,valid,score\na,true,0.5\nb,false,0.2\n")
            with self.assertRaisesRegex(ValueError, "invalid PDM"):
                read_pdm_rows(
                    path,
                    expected_tokens={"a", "b"},
                    required_metrics=("score",),
                )

    def test_pdm_reader_fails_closed_on_missing_or_nonfinite_metric(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("missing", ""), ("nan", "nan"), ("text", "bad")):
                path = root / f"{name}.csv"
                path.write_text(f"token,valid,score\na,true,{value}\n")
                with self.assertRaises(StatisticsError):
                    read_pdm_rows(
                        path,
                        expected_tokens={"a"},
                        required_metrics=("score",),
                    )

    def test_pdm_reader_rejects_unregistered_or_ragged_columns(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / "unknown.csv"
            unknown.write_text("token,valid,score,mystery\na,true,0.5,1\n")
            with self.assertRaisesRegex(StatisticsError, "unregistered columns"):
                read_pdm_rows(
                    unknown,
                    expected_tokens={"a"},
                    required_metrics=("score",),
                )
            ragged = root / "ragged.csv"
            ragged.write_text("token,valid,score\na,true,0.5,extra\n")
            with self.assertRaisesRegex(StatisticsError, "excess cells"):
                read_pdm_rows(
                    ragged,
                    expected_tokens={"a"},
                    required_metrics=("score",),
                )
        self.assertIn("driving_direction_compliance", PDM_DEFAULT_METRICS)

    def test_pairing_rejects_any_token_or_metric_intersection(self) -> None:
        mapping = {
            "a": "2021.05.01.00.00.00_veh-01_00000_00001",
            "b": "2021.05.02.00.00.00_veh-02_00000_00001",
        }
        with self.assertRaisesRegex(StatisticsError, "exactly equal"):
            paired_session_bootstrap(
                {"a": {"score": 0.1}},
                {"a": {"score": 0.2}, "b": {"score": 0.3}},
                mapping,
                metrics=("score",),
                repetitions=20,
            )
        with self.assertRaisesRegex(StatisticsError, "metric set differs"):
            paired_session_bootstrap(
                {"a": {"score": 0.1}, "b": {}},
                {"a": {"score": 0.2}, "b": {"score": 0.3}},
                mapping,
                metrics=("score",),
                repetitions=20,
            )

    def test_holm_and_hierarchical_seed_session_bootstrap(self) -> None:
        adjusted = holm_adjusted_pvalues({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        family = complete_holm_family(
            {"h1": 0.01, "h2": 0.04},
            expected_hypothesis_ids=("h1", "h2"),
        )
        self.assertTrue(family["h1"]["reject_global_null"])
        with self.assertRaisesRegex(StatisticsError, "exactly equal"):
            complete_holm_family(
                {"h1": 0.01}, expected_hypothesis_ids=("h1", "h2")
            )
        logs = {
            "a": "2021.05.01.00.00.00_veh-01_00000_00001",
            "b": "2021.05.02.00.00.00_veh-02_00000_00001",
        }
        seed_a = {
            seed: {token: {"score": 0.4 + 0.01 * seed} for token in logs}
            for seed in (0, 1, 2)
        }
        seed_b = {
            seed: {token: {"score": 0.5 + 0.01 * seed} for token in logs}
            for seed in (0, 1, 2)
        }
        result = hierarchical_seed_session_bootstrap(
            seed_a,
            seed_b,
            logs,
            metrics=("score",),
            expected_seed_ids=(0, 1, 2),
            repetitions=200,
            seed=3,
        )
        self.assertAlmostEqual(result["score"]["difference_b_minus_a"], 0.1)
        self.assertEqual(result["score"]["training_seed_count"], 3.0)
        self.assertEqual(result["score"]["crossed_seed_session_design"], 1.0)
        self.assertEqual(result["score"]["minimum_cluster_dimension"], 2.0)
        self.assertEqual(result["score"]["small_seed_inference_caveat"], 1.0)
        self.assertIn("null_centered_studentized_pvalue", result["score"])

        with self.assertRaisesRegex(StatisticsError, "exactly equal expected seed"):
            hierarchical_seed_session_bootstrap(
                seed_a,
                {0: seed_b[0], 1: seed_b[1]},
                logs,
                metrics=("score",),
                expected_seed_ids=(0, 1, 2),
                repetitions=20,
            )

    def test_continual_matrix_metrics(self) -> None:
        result = continual_matrix_metrics(
            [[0.8, 0.2, 0.1], [0.7, 0.9, 0.3], [0.6, 0.85, 0.95]],
            untrained_domain_scores=[0.1, 0.1, 0.1],
        )
        self.assertAlmostEqual(result["backward_transfer"], -0.125)
        self.assertAlmostEqual(result["forward_transfer"], 0.15)


if __name__ == "__main__":
    unittest.main()

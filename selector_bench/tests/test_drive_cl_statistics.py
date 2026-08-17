from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from selector_bench.continual.seed_design import (
    _IN_PROCESS_REPLAY_MEMO,
    _memo_lookup,
    _memo_store,
    seed_design_replay_identity,
    simulate_family_design,
)
from selector_bench.continual.statistics import (
    PDM_DEFAULT_METRICS,
    StatisticsError,
    complete_holm_family,
    continual_matrix_metrics,
    crossed_matrix_bootstrap_inference,
    crossed_matrix_bootstrap_pvalues_batch,
    crossed_matrix_bootstrap_shift_pvalues_batch,
    hierarchical_seed_session_bootstrap,
    holm_adjusted_pvalues,
    holm_rejection_matrix,
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

    def test_seed_design_and_production_share_bootstrap_and_holm_kernels(self) -> None:
        matrix = np.asarray(
            [[-0.04, 0.01, 0.02], [0.03, -0.02, 0.05], [0.01, 0.04, -0.01]],
            dtype=np.float64,
        )
        tokens = ("a", "b", "c")
        logs = {
            token: f"2021.05.0{index + 1}.00.00.00_veh-0{index + 1}_00000_00001"
            for index, token in enumerate(tokens)
        }
        baseline = {
            seed: {token: {"score": 0.0} for token in tokens}
            for seed in range(3)
        }
        candidate = {
            seed: {
                token: {"score": float(matrix[seed, session])}
                for session, token in enumerate(tokens)
            }
            for seed in range(3)
        }
        production = hierarchical_seed_session_bootstrap(
            baseline,
            candidate,
            logs,
            metrics=("score",),
            expected_seed_ids=(0, 1, 2),
            repetitions=200,
            seed=17,
        )
        shared = crossed_matrix_bootstrap_inference(
            matrix, repetitions=200, seed=17
        )
        self.assertEqual(
            production["score"]["null_centered_studentized_pvalue"],
            shared["null_centered_studentized_pvalue"],
        )

        pvalues = np.asarray([[0.01, 0.02, 0.5], [0.04, 0.001, 0.03]])
        rejected = holm_rejection_matrix(pvalues, alpha=0.05)
        for row_index, row in enumerate(pvalues):
            exact = complete_holm_family(
                {f"h{index}": float(value) for index, value in enumerate(row)},
                expected_hypothesis_ids=tuple(
                    f"h{index}" for index in range(len(row))
                ),
                alpha=0.05,
            )
            self.assertEqual(
                rejected[row_index].tolist(),
                [exact[f"h{index}"]["reject_global_null"] for index in range(len(row))],
            )

    def test_conditional_batch_matches_production_and_not_unconditional_surrogate(
        self,
    ) -> None:
        pilot = np.asarray(
            [
                [-0.3, 0.2, 0.1, 0.0, 0.4],
                [0.15, -0.1, 0.05, 0.35, -0.25],
                [0.5, 0.1, -0.4, 0.2, 0.0],
                [-0.2, 0.3, 0.25, -0.05, 0.1],
            ],
            dtype=np.float64,
        )
        observed = np.asarray(
            [
                [-0.32, -0.32, -0.32, -0.02, 0.28],
                [0.03, 0.03, 0.03, 0.18, -0.12],
                [-0.32, -0.32, -0.32, -0.02, 0.28],
                [-0.32, -0.32, -0.32, -0.02, 0.28],
                [-0.32, -0.32, -0.32, -0.02, 0.28],
            ],
            dtype=np.float64,
        )
        matrices = np.stack((observed, observed + 0.04), axis=0)
        batch = crossed_matrix_bootstrap_pvalues_batch(
            matrices, repetitions=997, seed=13
        )
        for index, matrix in enumerate(matrices):
            scalar = crossed_matrix_bootstrap_inference(
                matrix, repetitions=997, seed=13
            )
            self.assertEqual(
                float(batch["pvalues"][index]),
                scalar["null_centered_studentized_pvalue"],
            )

        pilot_null = crossed_matrix_bootstrap_inference(
            pilot, repetitions=997, seed=13
        )["null_statistics"]
        observed_statistic = crossed_matrix_bootstrap_inference(
            observed, repetitions=997, seed=13
        )["observed_statistic"]
        unconditional_surrogate = float(
            (1.0 + np.sum(pilot_null >= observed_statistic)) / 998.0
        )
        conditional_production = float(batch["pvalues"][0])
        self.assertGreater(
            abs(unconditional_surrogate - conditional_production), 0.1
        )

        rng = np.random.default_rng(20260818)
        for shape, repetitions, seed in (
            ((2, 3), 37, 0),
            ((3, 5), 113, 19),
            ((6, 4), 257, 2**16 + 7),
        ):
            observed_family = rng.normal(size=(3, *shape))
            exact_batch = crossed_matrix_bootstrap_pvalues_batch(
                observed_family,
                repetitions=repetitions,
                seed=seed,
                matrix_batch_size=2,
            )
            shifts = (0.0, 0.07, -0.11)
            shifted_batch = crossed_matrix_bootstrap_shift_pvalues_batch(
                observed_family,
                constant_shifts=shifts,
                repetitions=repetitions,
                seed=seed,
                matrix_batch_size=2,
            )
            for matrix_index, base in enumerate(observed_family):
                materialized = crossed_matrix_bootstrap_inference(
                    base, repetitions=repetitions, seed=seed
                )
                self.assertEqual(
                    float(exact_batch["pvalues"][matrix_index]),
                    materialized["null_centered_studentized_pvalue"],
                )
                materialized_pvalue = float(
                    (
                        1.0
                        + np.sum(
                            materialized["null_statistics"]
                            >= materialized["observed_statistic"]
                        )
                    )
                    / (repetitions + 1.0)
                )
                self.assertEqual(
                    float(exact_batch["pvalues"][matrix_index]),
                    materialized_pvalue,
                )
                for shift_index, shift in enumerate(shifts):
                    scalar = crossed_matrix_bootstrap_inference(
                        base + shift,
                        repetitions=repetitions,
                        seed=seed,
                    )["null_centered_studentized_pvalue"]
                    self.assertEqual(
                        float(shifted_batch["pvalues"][matrix_index, shift_index]),
                        scalar,
                    )

    def test_production_metric_seed_offset_is_frozen_order_offset(self) -> None:
        matrix = np.asarray(
            [[-0.04, 0.01, 0.02], [0.03, -0.02, 0.05], [0.01, 0.04, -0.01]],
            dtype=np.float64,
        )
        tokens = ("a", "b", "c")
        logs = {
            token: f"2021.05.0{index + 1}.00.00.00_veh-0{index + 1}_00000_00001"
            for index, token in enumerate(tokens)
        }
        baseline = {
            seed: {
                token: {"score": 0.0, "comfort": 0.0}
                for token in tokens
            }
            for seed in range(3)
        }
        candidate = {
            seed: {
                token: {
                    "score": float(matrix[seed, session]),
                    "comfort": float(2.0 * matrix[seed, session]),
                }
                for session, token in enumerate(tokens)
            }
            for seed in range(3)
        }
        production = hierarchical_seed_session_bootstrap(
            baseline,
            candidate,
            logs,
            metrics=("score", "comfort"),
            expected_seed_ids=(0, 1, 2),
            repetitions=211,
            seed=43,
        )
        self.assertEqual(
            production["score"]["null_centered_studentized_pvalue"],
            crossed_matrix_bootstrap_inference(
                matrix, repetitions=211, seed=43
            )["null_centered_studentized_pvalue"],
        )
        self.assertEqual(
            production["comfort"]["null_centered_studentized_pvalue"],
            crossed_matrix_bootstrap_inference(
                2.0 * matrix, repetitions=211, seed=44
            )["null_centered_studentized_pvalue"],
        )

    def test_family_power_replay_uses_each_observed_conditional_production_pvalue(
        self,
    ) -> None:
        matrices = {
            "h0": np.asarray(
                [[-0.3, 0.2, 0.1], [0.15, -0.1, 0.05], [0.5, 0.1, -0.4]]
            ),
            "h1": np.asarray(
                [[0.2, -0.1, 0.4], [-0.2, 0.3, 0.1], [0.05, -0.3, 0.25]]
            ),
        }
        effects = {"h0": 0.2, "h1": 0.15}
        repetitions = [127, 131]
        seeds = [11, 29]
        outer = 5
        audit = simulate_family_design(
            matrices,
            effects,
            ordered_hypothesis_ids=["h0", "h1"],
            ordered_hypothesis_bootstrap_repetitions=repetitions,
            ordered_hypothesis_kernel_seeds=seeds,
            seed_count=4,
            alpha=0.05,
            outer_repetitions=outer,
            rng=np.random.default_rng(71),
        )

        replay_rng = np.random.default_rng(71)
        seed_indices = replay_rng.integers(0, 3, size=(outer, 4))
        session_indices = replay_rng.integers(0, 3, size=(outer, 3))
        null_pvalues = np.empty((outer, 2))
        positive_pvalues = np.empty_like(null_pvalues)
        negative_pvalues = np.empty_like(null_pvalues)
        for column, hypothesis_id in enumerate(("h0", "h1")):
            centered = matrices[hypothesis_id] - matrices[hypothesis_id].mean()
            observed = centered[
                seed_indices[:, :, None], session_indices[:, None, :]
            ]
            for row in range(outer):
                for destination, shifted in (
                    (null_pvalues, observed[row]),
                    (positive_pvalues, observed[row] + effects[hypothesis_id]),
                    (negative_pvalues, observed[row] - effects[hypothesis_id]),
                ):
                    destination[row, column] = crossed_matrix_bootstrap_inference(
                        shifted,
                        repetitions=repetitions[column],
                        seed=seeds[column],
                    )["null_centered_studentized_pvalue"]
        null_rejected = holm_rejection_matrix(null_pvalues, alpha=0.05)
        self.assertEqual(
            audit["heldout_null_fwer"],
            float(np.mean(np.any(null_rejected, axis=1))),
        )
        positive_rejected = holm_rejection_matrix(positive_pvalues, alpha=0.05)
        negative_rejected = holm_rejection_matrix(negative_pvalues, alpha=0.05)
        for column, hypothesis_id in enumerate(("h0", "h1")):
            self.assertEqual(
                audit["hypothesis_power"][hypothesis_id]["positive_effect"],
                float(np.mean(positive_rejected[:, column])),
            )
            self.assertEqual(
                audit["hypothesis_power"][hypothesis_id]["negative_effect"],
                float(np.mean(negative_rejected[:, column])),
            )

    def test_in_process_replay_identity_invalidates_every_bound_input(self) -> None:
        invocation = {
            "family_spec_sha256": "1" * 64,
            "pilot_matrix_sha256": "2" * 64,
            "ordered_comparison_contracts": [
                {
                    "comparison_spec_sha256": "3" * 64,
                    "bootstrap_repetitions": 10000,
                    "bootstrap_seed": 7,
                }
            ],
            "simulation_repetitions": 1000,
            "simulation_seed": 5,
        }
        arguments = {
            "normalized_invocation_payload": invocation,
            "pilot_reference": "pilot.json",
            "generator_module_repository_path": "seed_design.py",
            "generator_module_sha256": "4" * 64,
            "generator_entrypoint_repository_path": "design_cli.py",
            "generator_entrypoint_sha256": "5" * 64,
            "inference_module_repository_path": "statistics.py",
            "inference_module_sha256": "6" * 64,
        }
        original, _ = seed_design_replay_identity(**arguments)
        mutations = []
        for field in (
            "pilot_reference",
            "generator_module_repository_path",
            "generator_module_sha256",
            "generator_entrypoint_repository_path",
            "generator_entrypoint_sha256",
            "inference_module_repository_path",
            "inference_module_sha256",
        ):
            mutated = dict(arguments)
            value = mutated[field]
            mutated[field] = (
                "7" * 64 if field.endswith("sha256") else str(value) + ".changed"
            )
            mutations.append(mutated)
        for invocation_field, value in (
            ("family_spec_sha256", "8" * 64),
            ("pilot_matrix_sha256", "9" * 64),
            ("simulation_repetitions", 1001),
            ("simulation_seed", 6),
        ):
            mutated = dict(arguments)
            changed_invocation = json.loads(json.dumps(invocation))
            changed_invocation[invocation_field] = value
            mutated["normalized_invocation_payload"] = changed_invocation
            mutations.append(mutated)
        changed_spec = dict(arguments)
        changed_invocation = json.loads(json.dumps(invocation))
        changed_invocation["ordered_comparison_contracts"][0][
            "bootstrap_repetitions"
        ] = 1000
        changed_spec["normalized_invocation_payload"] = changed_invocation
        mutations.append(changed_spec)
        for mutated in mutations:
            observed, _ = seed_design_replay_identity(**mutated)
            self.assertNotEqual(original, observed)

    def test_in_process_replay_memo_is_one_entry_and_clears_on_miss(self) -> None:
        _IN_PROCESS_REPLAY_MEMO.clear()
        _memo_store("a" * 64, b'{"first":true}')
        self.assertEqual(_memo_lookup("a" * 64), b'{"first":true}')
        _memo_store("b" * 64, b'{"second":true}')
        self.assertEqual(len(_IN_PROCESS_REPLAY_MEMO), 1)
        self.assertNotIn("a" * 64, _IN_PROCESS_REPLAY_MEMO)
        self.assertIsNone(_memo_lookup("c" * 64))
        self.assertEqual(_IN_PROCESS_REPLAY_MEMO, {})

    def test_continual_matrix_metrics(self) -> None:
        result = continual_matrix_metrics(
            [[0.8, 0.2, 0.1], [0.7, 0.9, 0.3], [0.6, 0.85, 0.95]],
            untrained_domain_scores=[0.1, 0.1, 0.1],
        )
        self.assertAlmostEqual(result["backward_transfer"], -0.125)
        self.assertAlmostEqual(result["forward_transfer"], 0.15)


if __name__ == "__main__":
    unittest.main()

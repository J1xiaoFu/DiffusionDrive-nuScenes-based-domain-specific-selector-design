from __future__ import annotations

import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

from selector_bench.integrations.diffusiondrive.mock_features import generate_mock_feature_cache
from selector_bench.training.stage1_heuristic import build_stage1_selector_checkpoint
from selector_bench.training.stage2_high_frequency import (
    parse_eval_metrics,
    run_stage2_high_frequency_update,
    write_subset_train_config,
)
from selector_bench.training.stage2_neural import (
    append_episode_record,
    build_episode_record,
    reweight_episode_record,
    sample_gumbel_topk_selection,
    score_neural_selector,
    train_neural_selector_stage1_imitation,
    train_neural_selector_stage2_feedback,
    write_episode_schema,
)
from selector_bench.utils.io import read_json, read_jsonl


class Stage2NeuralTest(unittest.TestCase):
    def _load_generalization_script(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "30_diagnose_stage2_feedback_generalization.py"
        spec = importlib.util.spec_from_file_location("stage2_generalization_diagnostic", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_candidate_gate_script(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "31_score_stage2_candidates.py"
        spec = importlib.util.spec_from_file_location("stage2_candidate_gate", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_imitation_training_and_gumbel_sampling_are_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            selection_path = root / "selection.json"
            generate_mock_feature_cache(
                feature_dir,
                max_train_samples=30,
                max_val_samples=6,
                num_domains=3,
                seed=17,
            )
            build_stage1_selector_checkpoint(feature_dir, teacher_path)

            artifact = train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=20,
                hidden_dim=12,
                seed=3,
            )
            self.assertEqual(artifact["selector"]["format"], "selector_bench.neural_selector.v0")

            selection = sample_gumbel_topk_selection(
                feature_dir=feature_dir,
                selector_checkpoint=neural_path,
                output_path=selection_path,
                budget=0.2,
                temperature=0.1,
                seed=5,
            )
            payload = selection["selection"]
            self.assertEqual(payload["selected_count"], 6)
            self.assertEqual(sum(int(value) for value in payload["domain_counts"].values()), 6)

            loaded = read_json(selection_path)
            self.assertEqual(loaded["selection"]["selection_hash"], payload["selection_hash"])


    def test_stage2_feedback_training_moves_subset_gap_toward_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            feedback_path = root / "neural_feedback.json"
            selector_selection_path = root / "selector_selection.json"
            random_selection_path = root / "random_selection.json"
            episode_path = root / "episodes.jsonl"
            generate_mock_feature_cache(feature_dir, max_train_samples=28, max_val_samples=4, seed=29)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=15,
                hidden_dim=10,
                seed=7,
            )
            sample_gumbel_topk_selection(feature_dir, neural_path, selector_selection_path, budget=0.25, seed=3)
            sample_gumbel_topk_selection(feature_dir, neural_path, random_selection_path, budget=0.25, seed=4)
            record = build_episode_record(
                episode_id="feedback_unit",
                selector_selection_path=selector_selection_path,
                random_selection_path=random_selection_path,
                selector_metrics={"l2": 4.0, "obj_box_col": 0.0, "obj_col": 0.0},
                random_metrics={"l2": 5.0, "obj_box_col": 0.0, "obj_col": 0.0},
            )
            append_episode_record(episode_path, record)

            artifact = train_neural_selector_stage2_feedback(
                feature_dir=feature_dir,
                init_selector_path=neural_path,
                episode_jsonl=episode_path,
                output_path=feedback_path,
                epochs=60,
                lr=0.02,
                seed=13,
            )
            metrics = artifact["metadata"]["metrics"]
            self.assertLess(metrics["after_feedback_mse"], metrics["before_feedback_mse"])
            self.assertEqual(artifact["selector"]["stage2_feedback"]["source"], "stage2_episode_jsonl")
            self.assertTrue(feedback_path.exists())


    def test_domain_residual_head_scores_and_feedback_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural_residual.json"
            feedback_path = root / "neural_residual_feedback.json"
            selector_selection_path = root / "selector_selection.json"
            random_selection_path = root / "random_selection.json"
            episode_path = root / "episodes.jsonl"
            generate_mock_feature_cache(feature_dir, max_train_samples=30, max_val_samples=4, num_domains=3, seed=31)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            artifact = train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=12,
                hidden_dim=9,
                use_domain_residual_head=True,
                seed=11,
            )
            self.assertTrue(artifact["selector"]["architecture"]["use_domain_residual_head"])
            self.assertIn("domain_residual_w", artifact["selector"]["params"])

            sample_gumbel_topk_selection(feature_dir, neural_path, selector_selection_path, budget=0.2, seed=7)
            sample_gumbel_topk_selection(feature_dir, neural_path, random_selection_path, budget=0.2, seed=8)
            record = build_episode_record(
                episode_id="residual_feedback_unit",
                selector_selection_path=selector_selection_path,
                random_selection_path=random_selection_path,
                selector_metrics={"l2": 6.0, "obj_box_col": 0.0, "obj_col": 0.0},
                random_metrics={"l2": 4.0, "obj_box_col": 0.0, "obj_col": 0.0},
            )
            append_episode_record(episode_path, record)
            feedback = train_neural_selector_stage2_feedback(
                feature_dir=feature_dir,
                init_selector_path=neural_path,
                episode_jsonl=episode_path,
                output_path=feedback_path,
                epochs=40,
                lr=0.015,
                seed=17,
            )
            self.assertTrue(feedback["selector"]["architecture"]["use_domain_residual_head"])
            self.assertLess(
                feedback["metadata"]["metrics"]["after_feedback_mse"],
                feedback["metadata"]["metrics"]["before_feedback_mse"],
            )

    def test_episode_schema_and_reward_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            selector_selection_path = root / "selector_selection.json"
            random_selection_path = root / "random_selection.json"
            schema_path = root / "episode_schema.json"
            generate_mock_feature_cache(feature_dir, max_train_samples=20, max_val_samples=4, seed=19)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=10,
                hidden_dim=8,
            )
            sample_gumbel_topk_selection(feature_dir, neural_path, selector_selection_path, budget=0.2, seed=1)
            sample_gumbel_topk_selection(feature_dir, neural_path, random_selection_path, budget=0.2, seed=2)

            schema = write_episode_schema(schema_path)
            self.assertEqual(schema["schema"], "selector_bench.stage2_episode.v0")

            record = build_episode_record(
                episode_id="unit_episode",
                selector_selection_path=selector_selection_path,
                random_selection_path=random_selection_path,
                selector_metrics={"l2": 4.0, "obj_box_col": 0.2, "obj_col": 0.0},
                random_metrics={"l2": 5.0, "obj_box_col": 0.2, "obj_col": 0.0},
                rho_box=0.05,
                rho_obj=0.05,
            )
            self.assertAlmostEqual(record["reward"], 1.0)
            self.assertEqual(record["schema"], "selector_bench.stage2_episode.v0")

            reweighted = reweight_episode_record(
                record,
                rho_box=100.0,
                episode_id_suffix="_rho_box100",
                notes_suffix="rho_box=100 unit reweight",
            )
            self.assertEqual(reweighted["episode_id"], "unit_episode_rho_box100")
            self.assertAlmostEqual(reweighted["reward"], 1.0)
            self.assertEqual(reweighted["reward_definition"]["rho_box"], 100.0)
            self.assertIn("rho_box=100 unit reweight", reweighted["notes"])


    def test_replay_penalty_avoids_negative_episode_selector_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            first_selection_path = root / "first_selection.json"
            random_selection_path = root / "random_selection.json"
            replay_selection_path = root / "replay_selection.json"
            episode_path = root / "episodes.jsonl"
            generate_mock_feature_cache(feature_dir, max_train_samples=45, max_val_samples=4, num_domains=3, seed=37)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=10,
                hidden_dim=8,
                seed=19,
            )
            first = sample_gumbel_topk_selection(feature_dir, neural_path, first_selection_path, budget=0.2, seed=5)
            sample_gumbel_topk_selection(feature_dir, neural_path, random_selection_path, budget=0.2, seed=6)
            record = build_episode_record(
                episode_id="negative_replay_unit",
                selector_selection_path=first_selection_path,
                random_selection_path=random_selection_path,
                selector_metrics={"l2": 6.0, "obj_box_col": 0.0, "obj_col": 0.0},
                random_metrics={"l2": 4.0, "obj_box_col": 0.0, "obj_col": 0.0},
            )
            append_episode_record(episode_path, record)

            replay = sample_gumbel_topk_selection(
                feature_dir,
                neural_path,
                replay_selection_path,
                budget=0.2,
                seed=5,
                replay_episode_jsonl=episode_path,
                replay_penalty=1e6,
            )
            overlap = set(first["selection"]["selected_tokens"]) & set(replay["selection"]["selected_tokens"])
            self.assertEqual(overlap, set())
            self.assertEqual(replay["selection"]["replay_avoidance"]["selected_replay_overlap_count"], 0)

    def test_negative_token_calibration_offsets_negative_selector_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            feedback_path = root / "neural_feedback_calibrated.json"
            selector_selection_path = root / "selector_selection.json"
            random_selection_path = root / "random_selection.json"
            episode_path = root / "episodes.jsonl"
            generate_mock_feature_cache(feature_dir, max_train_samples=36, max_val_samples=4, num_domains=3, seed=43)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=10,
                hidden_dim=8,
                seed=21,
            )
            selector = sample_gumbel_topk_selection(feature_dir, neural_path, selector_selection_path, budget=0.2, seed=5)
            sample_gumbel_topk_selection(feature_dir, neural_path, random_selection_path, budget=0.2, seed=6)
            append_episode_record(
                episode_path,
                build_episode_record(
                    episode_id="negative_calibration_unit",
                    selector_selection_path=selector_selection_path,
                    random_selection_path=random_selection_path,
                    selector_metrics={"l2": 6.0, "obj_box_col": 0.0, "obj_col": 0.0},
                    random_metrics={"l2": 4.0, "obj_box_col": 0.0, "obj_col": 0.0},
                ),
            )

            artifact = train_neural_selector_stage2_feedback(
                feature_dir=feature_dir,
                init_selector_path=neural_path,
                episode_jsonl=episode_path,
                output_path=feedback_path,
                epochs=0,
                negative_token_penalty=0.5,
                seed=13,
            )
            offsets = artifact["selector"]["score_adjustments"]["token_offsets"]
            selected_tokens = selector["selection"]["selected_tokens"]
            self.assertTrue(all(token in offsets for token in selected_tokens))
            self.assertEqual(
                artifact["metadata"]["metrics"]["score_adjustment_token_count"],
                len(set(selected_tokens)),
            )

            from selector_bench.core.feature_store import FeatureStore

            store = FeatureStore.from_feature_dir(feature_dir, cache=True)
            before = score_neural_selector(store, neural_path, selected_tokens)
            after = score_neural_selector(store, feedback_path, selected_tokens)
            self.assertLess(float(after.mean()), float(before.mean()))


    def test_neural_scores_are_per_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            generate_mock_feature_cache(feature_dir, max_train_samples=12, max_val_samples=2, seed=23)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=5,
                hidden_dim=6,
            )
            from selector_bench.core.feature_store import FeatureStore

            store = FeatureStore.from_feature_dir(feature_dir, cache=True)
            tokens = store.manifest.tokens(split="train")[:4]
            scores = score_neural_selector(store, neural_path, tokens)
            self.assertEqual(scores.shape, (4,))


    def test_leave_one_out_generalization_diagnostic_writes_report(self) -> None:
        diagnostic = self._load_generalization_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            episode_path = root / "episodes.jsonl"
            work_dir = root / "loo"
            report_md = root / "report.md"
            generate_mock_feature_cache(feature_dir, max_train_samples=42, max_val_samples=4, num_domains=3, seed=41)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=8,
                hidden_dim=8,
                seed=23,
            )
            selections = []
            for idx, seed in enumerate([3, 4, 5, 6, 7, 8]):
                path = root / f"selection_{idx}.json"
                sample_gumbel_topk_selection(feature_dir, neural_path, path, budget=0.2, seed=seed)
                selections.append(path)
            for idx, (selector_path, random_path, selector_l2, random_l2) in enumerate(
                [
                    (selections[0], selections[1], 4.0, 5.0),
                    (selections[2], selections[3], 6.0, 4.0),
                    (selections[4], selections[5], 4.5, 5.5),
                ]
            ):
                append_episode_record(
                    episode_path,
                    build_episode_record(
                        episode_id=f"fold_unit_{idx}",
                        selector_selection_path=selector_path,
                        random_selection_path=random_path,
                        selector_metrics={"l2": selector_l2, "obj_box_col": 0.0, "obj_col": 0.0},
                        random_metrics={"l2": random_l2, "obj_box_col": 0.0, "obj_col": 0.0},
                    ),
                )

            report = diagnostic.build_leave_one_out_report(
                feature_dir=feature_dir,
                init_selector_path=neural_path,
                episode_jsonl=episode_path,
                work_dir=work_dir,
                run_id="unit_loo",
                epochs=6,
                lr=0.01,
                l2=1e-4,
                reward_scale=None,
                target_clip=2.0,
                min_abs_reward=0.0,
                seed=31,
            )
            diagnostic.write_markdown(report, report_md)
            self.assertEqual(report["schema"], "selector_bench.stage2_feedback_generalization.v0")
            self.assertEqual(report["summary"]["fold_count"], 3)
            self.assertEqual(len(report["folds"]), 3)
            self.assertTrue(report_md.exists())


    def test_candidate_gate_scores_candidates_with_fold_checkpoint(self) -> None:
        gate = self._load_candidate_gate_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            selector_selection_path = root / "selector_selection.json"
            random_selection_path = root / "random_selection.json"
            episode_path = root / "episodes.jsonl"
            fold_report_path = root / "fold_report.json"
            candidate_report_path = root / "candidate_report.json"
            report_md = root / "candidate_gate.md"
            generate_mock_feature_cache(feature_dir, max_train_samples=36, max_val_samples=4, num_domains=3, seed=47)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=8,
                hidden_dim=8,
                seed=29,
            )
            candidate = sample_gumbel_topk_selection(feature_dir, neural_path, selector_selection_path, budget=0.2, seed=5)
            sample_gumbel_topk_selection(feature_dir, neural_path, random_selection_path, budget=0.2, seed=6)
            append_episode_record(
                episode_path,
                build_episode_record(
                    episode_id="candidate_gate_unit",
                    selector_selection_path=selector_selection_path,
                    random_selection_path=random_selection_path,
                    selector_metrics={"l2": 6.0, "obj_box_col": 0.0, "obj_col": 0.0},
                    random_metrics={"l2": 4.0, "obj_box_col": 0.0, "obj_col": 0.0},
                ),
            )
            from selector_bench.utils.io import write_json

            write_json(
                fold_report_path,
                {
                    "folds": [
                        {
                            "fold": 0,
                            "checkpoint": str(neural_path),
                        }
                    ]
                },
            )
            write_json(
                candidate_report_path,
                {
                    "items": [
                        {
                            "rank": 1,
                            "selection_hash": candidate["selection"]["selection_hash"],
                            "hash12": candidate["selection"]["selection_hash"][:12],
                            "path": str(selector_selection_path),
                            "seed": 5,
                            "replay_penalty": 0.0,
                        }
                    ],
                    "best": {
                        "hash12": candidate["selection"]["selection_hash"][:12],
                    },
                },
            )
            report = gate.build_candidate_gate_report(
                feature_dir=feature_dir,
                episode_jsonl=episode_path,
                fold_report_json=fold_report_path,
                candidate_report_json=candidate_report_path,
                max_negative_overlap=999,
                bootstrap_samples=16,
                bootstrap_seed=123,
                min_bootstrap_pass_rate=0.0,
            )
            gate.write_markdown(report, report_md)
            self.assertEqual(report["schema"], "selector_bench.stage2_candidate_gate.v0")
            self.assertEqual(report["summary"]["candidate_count"], 1)
            self.assertIn(report["candidates"][0]["verdict"], {"pass", "reject"})
            self.assertIn("candidate_minus_random_mean", report["candidates"][0])
            self.assertIn("bootstrap_random_margin", report["candidates"][0])
            self.assertEqual(report["candidates"][0]["bootstrap_random_margin"]["samples"], 16)
            self.assertTrue(report_md.exists())

    def test_high_frequency_feedback_runs_micro_round_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            teacher_path = root / "stage1.json"
            neural_path = root / "neural.json"
            episode_path = root / "episodes.jsonl"
            output_dir = root / "hf"
            source_info = root / "DiffusionDrive" / "data" / "infos" / "partial" / "train.pkl"
            base_config = root / "base.py"
            generate_mock_feature_cache(feature_dir, max_train_samples=48, max_val_samples=4, num_domains=3, seed=53)
            build_stage1_selector_checkpoint(feature_dir, teacher_path)
            train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=teacher_path,
                output_path=neural_path,
                epochs=8,
                hidden_dim=8,
                seed=31,
            )
            tokens = [row["sample_token"] for row in read_jsonl(feature_dir / "manifest.jsonl") if row["split"] == "train"]
            source_info.parent.mkdir(parents=True, exist_ok=True)
            with source_info.open("wb") as f:
                pickle.dump({"infos": [{"token": token} for token in tokens], "metadata": {}}, f)
            base_config.write_text("data = {}\n", encoding="utf-8")

            def metrics(round_idx: int, context: dict[str, object]):
                self.assertTrue(Path(str(context["selector_selection"])).exists())
                selector_l2 = 5.0 if round_idx % 2 else 3.8
                random_l2 = 4.2
                return (
                    {"l2": selector_l2, "obj_box_col": 0.01 * round_idx, "obj_col": 0.0},
                    {"l2": random_l2, "obj_box_col": 0.0, "obj_col": 0.0},
                )

            report = run_stage2_high_frequency_update(
                feature_dir=feature_dir,
                init_selector_path=neural_path,
                episode_jsonl=episode_path,
                output_dir=output_dir,
                source_info=source_info,
                diffusiondrive_root=root / "DiffusionDrive",
                base_config=base_config,
                run_id="hf_unit",
                rounds=3,
                selection_count=5,
                selector_epochs=6,
                selector_lr=0.01,
                metric_provider=metrics,
                disjoint_pair=True,
            )
            self.assertEqual(report["schema"], "selector_bench.stage2_high_frequency_update.v0")
            self.assertEqual(report["summary"]["status"], "complete")
            self.assertEqual(report["summary"]["completed_rounds"], 3)
            self.assertEqual(len(read_jsonl(episode_path)), 3)
            self.assertTrue((output_dir / "selectors" / "hf_unit_r03_updated_selector.json").exists())
            for item in report["rounds"]:
                self.assertEqual(item["selector_count"], 5)
                self.assertEqual(item["random_count"], 5)
                self.assertTrue(Path(item["selector_config"]).exists())
                selector_tokens = set(read_json(item["selector_selection"])["selection"]["selected_tokens"])
                random_tokens = set(read_json(item["random_selection"])["selection"]["selected_tokens"])
                self.assertEqual(selector_tokens & random_tokens, set())

    def test_high_frequency_eval_metric_parser_and_config_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            log_path = eval_dir / "outer_eval.log"
            status_path = eval_dir / "eval_status.json"
            log_path.write_text(
                "noise\n{'obj_col': 0.003, 'obj_box_col': 0.026, 'L2': 4.33}\n",
                encoding="utf-8",
            )
            status_path.write_text('{"runtime_seconds": 12.5, "peak_gpu_memory_nvidia_smi_mb": 2048}\n', encoding="utf-8")
            metrics = parse_eval_metrics(log_path, status_path)
            self.assertAlmostEqual(metrics["l2"], 4.33)
            self.assertAlmostEqual(metrics["obj_box_col"], 0.026)
            self.assertEqual(metrics["peak_gpu_mb"], 2048.0)

            dd_root = root / "DiffusionDrive"
            ann_file = dd_root / "data" / "infos" / "subsets" / "train.pkl"
            ann_file.parent.mkdir(parents=True)
            ann_file.write_bytes(b"")
            init_checkpoint = dd_root / "ckpts" / "fixed_init.pth"
            init_checkpoint.parent.mkdir(parents=True)
            init_checkpoint.write_bytes(b"")
            base_config = root / "configs" / "base.py"
            base_config.parent.mkdir()
            base_config.write_text("data = {}\n", encoding="utf-8")
            output_config = root / "configs" / "child.py"
            write_subset_train_config(
                base_config,
                output_config,
                ann_file,
                dd_root,
                planner_init_checkpoint=init_checkpoint,
                planner_lr=1e-6,
                planner_max_iters=100,
            )
            text = output_config.read_text(encoding="utf-8")
            self.assertIn('"base.py"', text)
            self.assertIn('"data/infos/subsets/train.pkl"', text)
            self.assertIn('optimizer = dict(lr=1e-06)', text)
            self.assertIn('runner = dict(type="IterBasedRunner", max_iters=100)', text)
            self.assertIn('evaluation = dict(interval=100)', text)
            self.assertIn('checkpoint_config = dict(interval=100)', text)
            self.assertIn('load_from = "ckpts/fixed_init.pth"', text)
            self.assertIn('resume_from = None', text)



if __name__ == "__main__":
    unittest.main()

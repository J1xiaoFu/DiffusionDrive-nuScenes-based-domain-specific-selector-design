from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from selector_bench.core.feature_store import FeatureStore
from selector_bench.integrations.diffusiondrive.mock_features import generate_mock_feature_cache
from selector_bench.training.domain_clustering import fit_stage1_domain_clusters
from selector_bench.training.stage1_domain_targets import build_stage1_domain_targets
from selector_bench.training.stage2_neural import score_neural_selector, train_neural_selector_stage1_imitation
from selector_bench.utils.io import read_json, read_jsonl, write_json, write_jsonl


class Stage1DomainTargetsTest(unittest.TestCase):
    def test_explicit_sidecar_targets_feed_neural_imitation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            labels_path = root / "labels.jsonl"
            target_path = root / "targets.jsonl"
            selector_path = root / "selector.json"
            generate_mock_feature_cache(feature_dir, max_train_samples=18, max_val_samples=3, num_domains=2, seed=101)
            store = FeatureStore.from_feature_dir(feature_dir)
            write_jsonl(
                labels_path,
                (
                    {
                        "sample_token": record.sample_token,
                        "domain_label": "fast" if idx % 2 == 0 else "dense",
                    }
                    for idx, record in enumerate(store.manifest.filter(split="train"))
                ),
            )
            config_path = root / "target_config.json"
            write_json(config_path, _target_config(feature_dir, target_path, labels_path))

            result = build_stage1_domain_targets(config_path)
            rows = result["rows"]
            self.assertEqual(len(rows), 18)
            self.assertEqual({row["heuristic_name"] for row in rows}, {"dense", "fast"})
            self.assertTrue((Path(str(target_path) + ".metadata.json")).exists())

            artifact = train_neural_selector_stage1_imitation(
                feature_dir=feature_dir,
                teacher_selector_path=None,
                target_score_path=target_path,
                output_path=selector_path,
                epochs=12,
                hidden_dim=8,
                seed=5,
            )
            self.assertEqual(artifact["selector"]["training_target"]["source"], "stage1_target_scores_jsonl")
            tokens = store.manifest.tokens(split="train")[:5]
            scores = score_neural_selector(store, selector_path, tokens)
            self.assertEqual(scores.shape, (5,))

    def test_cluster_labels_are_deterministic_and_build_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            labels_path = root / "cluster_labels.jsonl"
            model_path = root / "cluster_model.json"
            profile_path = root / "cluster_profile.md"
            target_path = root / "cluster_targets.jsonl"
            generate_mock_feature_cache(feature_dir, max_train_samples=24, max_val_samples=3, num_domains=3, seed=103)
            cluster_config_path = root / "cluster_config.json"
            cluster_config = _cluster_config(feature_dir, labels_path, model_path, profile_path)
            write_json(cluster_config_path, cluster_config)

            first = fit_stage1_domain_clusters(cluster_config_path)
            first_labels = [row["domain_label"] for row in first["labels"]]
            second = fit_stage1_domain_clusters(cluster_config_path)
            second_labels = [row["domain_label"] for row in second["labels"]]
            self.assertEqual(first_labels, second_labels)
            self.assertTrue(model_path.exists())
            self.assertTrue(profile_path.exists())

            target_config_path = root / "cluster_target_config.json"
            write_json(target_config_path, _cluster_target_config(feature_dir, labels_path, target_path))
            result = build_stage1_domain_targets(target_config_path)
            rows = read_jsonl(target_path)
            self.assertEqual(len(result["rows"]), 24)
            self.assertEqual(len(rows), 24)
            model = read_json(model_path)
            self.assertEqual(model["cluster_model"]["num_clusters"], 3)

    def test_raw_scale_does_not_rescale_minmax_target_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            labels_path = root / "labels.jsonl"
            target_path = root / "targets.jsonl"
            generate_mock_feature_cache(feature_dir, max_train_samples=12, max_val_samples=2, num_domains=1, seed=107)
            store = FeatureStore.from_feature_dir(feature_dir)
            write_jsonl(
                labels_path,
                (
                    {"sample_token": record.sample_token, "domain_label": "dense"}
                    for record in store.manifest.filter(split="train")
                ),
            )
            config = _target_config(feature_dir, target_path, labels_path)
            config["features"] = {
                "agent_density": {"kind": "agent_count_norm", "normalization": "minmax", "scale": 96.0}
            }
            config["heuristics"] = {
                "dense": {"weights": {"agent_density": 1.0}},
                "default": {"weights": {"agent_density": 1.0}},
            }
            config_path = root / "target_config.json"
            write_json(config_path, config)

            build_stage1_domain_targets(config_path)
            rows = read_jsonl(target_path)
            values = [row["feature_values"]["agent_density"] for row in rows]
            self.assertGreaterEqual(min(values), 0.0)
            self.assertLessEqual(max(values), 1.0)


def _target_config(feature_dir: Path, target_path: Path, labels_path: Path) -> dict:
    return {
        "schema": "selector_bench.stage1_target_config.v0",
        "run_id": "unit_stage1_targets",
        "feature_dir": str(feature_dir),
        "split": "train",
        "domain_source": {
            "type": "sidecar_labels",
            "path": str(labels_path),
            "label_field": "domain_label",
        },
        "default_domain": "default",
        "features": {
            "teacher_loss": {"kind": "teacher_loss", "normalization": "standard"},
            "teacher_uncertainty": {"kind": "teacher_uncertainty", "normalization": "standard"},
            "agent_density": {"kind": "agent_count_norm", "normalization": "minmax", "scale": 32.0},
            "ego_speed": {"kind": "ego_speed_norm", "normalization": "minmax", "scale": 30.0},
        },
        "heuristics": {
            "fast": {"weights": {"ego_speed": 1.0, "teacher_loss": 0.2}},
            "dense": {"weights": {"agent_density": 1.0, "teacher_uncertainty": 0.2}},
            "default": {"weights": {"teacher_loss": 1.0}},
        },
        "outputs": {"target_scores_path": str(target_path)},
    }


def _cluster_config(feature_dir: Path, labels_path: Path, model_path: Path, profile_path: Path) -> dict:
    return {
        "schema": "selector_bench.stage1_target_config.v0",
        "run_id": "unit_stage1_clusters",
        "feature_dir": str(feature_dir),
        "split": "train",
        "clustering": {"num_clusters": 3, "seed": 11, "max_iter": 20},
        "cluster_features": {
            "ego_speed": {"kind": "ego_speed_norm", "normalization": "standard", "cluster_weight": 1.0},
            "agent_density": {"kind": "agent_count_norm", "normalization": "standard", "cluster_weight": 1.0},
            "teacher_loss": {"kind": "teacher_loss", "normalization": "standard", "cluster_weight": 0.5},
        },
        "outputs": {
            "labels_path": str(labels_path),
            "cluster_model_path": str(model_path),
            "cluster_profile_path": str(profile_path),
        },
    }


def _cluster_target_config(feature_dir: Path, labels_path: Path, target_path: Path) -> dict:
    return {
        "schema": "selector_bench.stage1_target_config.v0",
        "run_id": "unit_cluster_targets",
        "feature_dir": str(feature_dir),
        "split": "train",
        "domain_source": {"type": "cluster", "labels_path": str(labels_path)},
        "default_domain": "default",
        "features": {
            "teacher_loss": {"kind": "teacher_loss", "normalization": "standard"},
            "agent_density": {"kind": "agent_count_norm", "normalization": "minmax"},
        },
        "heuristics": {
            "cluster_0": {"weights": {"teacher_loss": 0.5}},
            "cluster_1": {"weights": {"agent_density": 0.5}},
            "cluster_2": {"weights": {"teacher_loss": 0.2, "agent_density": 0.2}},
            "default": {"weights": {"teacher_loss": 1.0}},
        },
        "outputs": {"target_scores_path": str(target_path)},
    }


if __name__ == "__main__":
    unittest.main()

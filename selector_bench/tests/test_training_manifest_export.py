from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_exporter():
    path = ROOT / "selector_bench/scripts/47_export_drive_cl_training_manifest.py"
    spec = importlib.util.spec_from_file_location("training_manifest_export", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def cell(prefix: str, count: int = 2) -> dict[str, object]:
    sessions = [f"{prefix}-session-{index}" for index in range(count)]
    logs = [f"{prefix}-log-{index}" for index in range(count)]
    tokens = [f"{prefix}-token-{index}" for index in range(count)]
    return {
        "sessions": sessions,
        "logs": logs,
        "tokens": tokens,
        "token_to_log": dict(zip(tokens, logs)),
        "session_count": count,
        "log_count": count,
        "token_count": count,
    }


def manifest_fixture() -> dict[str, object]:
    return {
        "schema": "selector_bench.drive_cl_protocol.v1",
        "protocol_id": "synthetic",
        "stages": [
            {
                "stage_index": stage,
                "name": f"stage-{stage}",
                "splits": {
                    split: cell(f"s{stage}-{split}")
                    for split in ("train", "audit", "test")
                },
                "selection_metrics": {f"hidden-session-{stage}": {"risk": 1.0}},
            }
            for stage in (1, 2, 3)
        ],
    }


class TrainingManifestExportTest(unittest.TestCase):
    def test_nontraining_membership_is_empty_but_count_summary_is_retained(self) -> None:
        exporter = load_exporter()
        portable = exporter.build_training_derivative(
            manifest_fixture(), source_manifest_sha256="a" * 64
        )
        observed_train_tokens: list[str] = []
        for stage in portable["stages"]:
            self.assertNotIn("selection_metrics", stage)
            observed_train_tokens.extend(stage["splits"]["train"]["tokens"])
            self.assertTrue(stage["splits"]["train"]["token_to_log"])
            for split in ("audit", "test"):
                current = stage["splits"][split]
                self.assertEqual(current["sessions"], [])
                self.assertEqual(current["logs"], [])
                self.assertEqual(current["tokens"], [])
                self.assertEqual(current["token_to_log"], {})
                self.assertEqual(current["session_count"], 0)
                self.assertEqual(current["log_count"], 0)
                self.assertEqual(current["token_count"], 0)
                summary = portable["remote_training_cache_contract"][
                    "excluded_membership_count_summary"
                ][str(stage["stage_index"])][split]
                self.assertEqual(
                    summary,
                    {"session_count": 2, "log_count": 2, "token_count": 2},
                )
        self.assertEqual(len(observed_train_tokens), len(set(observed_train_tokens)))
        self.assertTrue(
            portable["remote_training_cache_contract"]["audit_and_test_excluded"]
        )

    def test_training_token_overlap_across_stages_is_rejected(self) -> None:
        exporter = load_exporter()
        source = manifest_fixture()
        duplicate = source["stages"][0]["splits"]["train"]["tokens"][0]
        second = source["stages"][1]["splits"]["train"]
        old = second["tokens"][0]
        second["tokens"][0] = duplicate
        second["token_to_log"][duplicate] = second["token_to_log"].pop(old)
        with self.assertRaisesRegex(RuntimeError, "overlap across stages"):
            exporter.build_training_derivative(source, source_manifest_sha256="b" * 64)

    def test_inconsistent_declared_counts_fail_closed(self) -> None:
        exporter = load_exporter()
        source = manifest_fixture()
        source["stages"][0]["splits"]["audit"]["token_count"] = 99
        with self.assertRaisesRegex(RuntimeError, "counts are inconsistent"):
            exporter.build_training_derivative(source, source_manifest_sha256="c" * 64)


if __name__ == "__main__":
    unittest.main()

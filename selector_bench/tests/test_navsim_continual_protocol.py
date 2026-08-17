from __future__ import annotations

import csv
import pickle
import tempfile
import unittest
from pathlib import Path

from selector_bench.continual.navsim_protocol import (
    ProtocolError,
    build_chronological_protocol,
    build_failure_patch_protocol,
    load_cache_inventory,
    session_id_from_log,
    validate_protocol,
)


class NavsimContinualProtocolTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        index: dict[str, Path] = {}
        rows: list[dict[str, str]] = []
        fields = [
            "scene_id",
            "dd_raw_log",
            "dd_raw_valid",
            "dd_raw_no_at_fault_collisions",
            "dd_raw_drivable_area_compliance",
            "dd_raw_ego_progress",
            "dd_raw_time_to_collision_within_bound",
            "dd_raw_comfort",
        ]
        for session_index in range(12):
            session = f"2021.05.{session_index + 1:02d}.10.00.00_veh-{session_index:02d}"
            for segment in range(2):
                log_name = f"{session}_{segment * 10:05d}_{segment * 10 + 9:05d}"
                for sample in range(2):
                    token = f"token-{session_index:02d}-{segment}-{sample}"
                    index[token] = root / "cache" / log_name / token
                    safety = 0.0 if session_index >= 8 else 1.0
                    rows.append(
                        {
                            "scene_id": f"{log_name}/{token}",
                            "dd_raw_log": log_name,
                            "dd_raw_valid": "1",
                            "dd_raw_no_at_fault_collisions": str(safety),
                            "dd_raw_drivable_area_compliance": "1",
                            "dd_raw_ego_progress": str(0.2 + session_index / 20),
                            "dd_raw_time_to_collision_within_bound": "1",
                            "dd_raw_comfort": "0" if session_index < 3 else "1",
                        }
                    )
        index_path = root / "valid_cache_train.pkl"
        with index_path.open("wb") as stream:
            pickle.dump(index, stream)
        csv_path = root / "metrics.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return index_path, csv_path

    def test_session_parser_groups_segmented_logs(self) -> None:
        expected = "2021.05.12.19.36.12_veh-35"
        self.assertEqual(
            session_id_from_log(f"{expected}_00005_00204"), expected
        )
        self.assertEqual(session_id_from_log(expected), expected)
        with self.assertRaises(ProtocolError):
            session_id_from_log("not-a-navsim-log")

    def test_both_protocols_are_deterministic_and_leakage_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path, csv_path = self._fixture(Path(tmp))
            inventory = load_cache_inventory(index_path)
            chronological = build_chronological_protocol(
                inventory, cache_index=index_path, seed=17
            )
            failure = build_failure_patch_protocol(
                inventory,
                cache_index=index_path,
                metrics_csv=csv_path,
                seed=17,
                safety_fraction=0.25,
                efficiency_fraction=0.25,
            )
            validate_protocol(chronological)
            validate_protocol(failure)
            repeat = build_failure_patch_protocol(
                inventory,
                cache_index=index_path,
                metrics_csv=csv_path,
                seed=17,
                safety_fraction=0.25,
                efficiency_fraction=0.25,
            )
            self.assertEqual(failure["content_sha256"], repeat["content_sha256"])
            self.assertEqual(failure["stages"], repeat["stages"])
            safety_sessions = set(failure["stages"][1]["selection_metrics"])
            self.assertEqual(len(safety_sessions), 3)
            self.assertTrue(
                safety_sessions.issubset(
                    {
                        f"2021.05.{index + 1:02d}.10.00.00_veh-{index:02d}"
                        for index in range(8, 12)
                    }
                )
            )

    def test_validator_rejects_cross_split_session_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path, _ = self._fixture(Path(tmp))
            protocol = build_chronological_protocol(
                load_cache_inventory(index_path), cache_index=index_path
            )
            leaked = protocol["stages"][0]["splits"]["train"]["sessions"][0]
            protocol["stages"][0]["splits"]["audit"]["sessions"].append(leaked)
            protocol["stages"][0]["splits"]["audit"]["session_count"] += 1
            with self.assertRaisesRegex(ProtocolError, "session leakage"):
                validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()

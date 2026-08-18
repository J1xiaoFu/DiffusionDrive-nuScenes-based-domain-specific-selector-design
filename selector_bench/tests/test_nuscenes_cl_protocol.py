from __future__ import annotations

import copy
import unittest

from selector_bench.continual.nuscenes_cl_protocol import (
    NuScenesProtocolError,
    build_geography_only_protocol,
    build_log_sessions,
    build_natural_mixed_protocol,
    build_time_only_protocol,
    validate_protocol,
)


def metadata_fixture() -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    metadata: dict[str, list[dict[str, object]]] = {"log": [], "scene": [], "sample": []}
    eligible: set[str] = set()
    locations = ("boston", "singapore-one", "singapore-queen")
    for location_index, location in enumerate(locations):
        for log_index in range(9):
            log_token = f"log-{location_index}-{log_index}"
            metadata["log"].append({"token": log_token, "location": location})
            for scene_offset in range(2):
                scene_token = f"scene-{location_index}-{log_index}-{scene_offset}"
                scene_name = f"scene-name-{location_index}-{log_index}-{scene_offset}"
                eligible.add(scene_name)
                metadata["scene"].append(
                    {
                        "token": scene_token,
                        "name": scene_name,
                        "log_token": log_token,
                        "nbr_samples": 2,
                    }
                )
                for sample_offset in range(2):
                    # All locations span a shared time window while retaining
                    # deterministic within-location chronological progression.
                    timestamp = 1_000 + log_index * 100 + scene_offset * 10 + sample_offset
                    metadata["sample"].append(
                        {
                            "token": f"sample-{location_index}-{log_index}-{scene_offset}-{sample_offset}",
                            "scene_token": scene_token,
                            "timestamp": timestamp,
                        }
                    )
    return metadata, eligible


class NuScenesControlledProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        metadata, eligible = metadata_fixture()
        self.metadata = metadata
        self.eligible = eligible
        self.sessions = build_log_sessions(metadata, eligible_scene_names=eligible)

    def test_log_is_the_atomic_unit_even_with_multiple_scenes(self) -> None:
        self.assertEqual(len(self.sessions), 27)
        self.assertTrue(all(len(session.scene_tokens) == 2 for session in self.sessions))
        self.assertTrue(all(len(session.sample_tokens) == 4 for session in self.sessions))

    def test_time_only_keeps_each_location_in_every_stage(self) -> None:
        protocol = build_time_only_protocol(self.sessions, seed=3)
        validate_protocol(protocol)
        for stage in protocol["stages"]:
            locations: set[str] = set()
            for split in ("train", "audit", "test"):
                locations.update(stage["splits"][split]["location_counts"])
            self.assertEqual(locations, {"boston", "singapore-one", "singapore-queen"})

    def test_geography_only_matches_time_and_changes_location(self) -> None:
        protocol = build_geography_only_protocol(
            self.sessions,
            location_groups=(("boston",), ("singapore-one",), ("singapore-queen",)),
            seed=5,
        )
        validate_protocol(protocol)
        observed = []
        for stage in protocol["stages"]:
            locations: set[str] = set()
            for split in ("train", "audit", "test"):
                locations.update(stage["splits"][split]["location_counts"])
            observed.append(locations)
        self.assertEqual(
            observed,
            [{"boston"}, {"singapore-one"}, {"singapore-queen"}],
        )

    def test_natural_mixed_and_child_closure(self) -> None:
        protocol = build_natural_mixed_protocol(self.sessions, seed=7)
        validate_protocol(protocol)
        broken = copy.deepcopy(protocol)
        broken["stages"][0]["splits"]["train"]["sample_tokens"].pop()
        with self.assertRaises(NuScenesProtocolError):
            validate_protocol(broken)

    def test_partial_official_log_population_is_rejected(self) -> None:
        eligible = set(self.eligible)
        eligible.remove("scene-name-0-0-1")
        with self.assertRaises(NuScenesProtocolError):
            build_log_sessions(self.metadata, eligible_scene_names=eligible)

    def test_duplicate_log_assignment_is_rejected(self) -> None:
        protocol = build_time_only_protocol(self.sessions, seed=3)
        broken = copy.deepcopy(protocol)
        duplicate = broken["stages"][0]["splits"]["train"]["log_tokens"][0]
        broken["stages"][0]["splits"]["audit"]["log_tokens"][0] = duplicate
        with self.assertRaises(NuScenesProtocolError):
            validate_protocol(broken)


if __name__ == "__main__":
    unittest.main()

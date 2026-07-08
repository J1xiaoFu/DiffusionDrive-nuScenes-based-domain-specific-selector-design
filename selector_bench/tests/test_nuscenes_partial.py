from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from selector_bench.integrations.diffusiondrive.nuscenes_partial import (
    SceneSummary,
    build_scene_summaries,
    collect_required_sample_data_filenames,
    build_blob_manifest,
    extract_blob_files_resumable,
    extract_zip_archive,
    safe_extract_member,
    select_balanced_scenes,
)


class NuScenesPartialTest(unittest.TestCase):
    def test_balanced_scene_sampler_covers_locations(self) -> None:
        scenes = [
            SceneSummary("t0", "scene-0000", "boston-seaport", "train", 40),
            SceneSummary("t1", "scene-0001", "boston-seaport", "train", 40),
            SceneSummary("t2", "scene-0002", "singapore-onenorth", "train", 40),
            SceneSummary("t3", "scene-0003", "singapore-onenorth", "train", 40),
        ]
        selected = select_balanced_scenes(scenes, "train", target_samples=80, seed=7)
        self.assertEqual(sum(scene.sample_count for scene in selected), 80)
        self.assertEqual({scene.location for scene in selected}, {"boston-seaport", "singapore-onenorth"})

    def test_scene_summaries_accept_explicit_splits(self) -> None:
        metadata = {
            "log": [{"token": "log0", "location": "boston-seaport"}],
            "scene": [
                {"token": "scene0", "name": "scene-0000", "log_token": "log0", "nbr_samples": 2},
                {"token": "scene1", "name": "scene-0001", "log_token": "log0", "nbr_samples": 3},
            ],
            "sample": [
                {"token": "s0", "scene_token": "scene0"},
                {"token": "s1", "scene_token": "scene0"},
                {"token": "s2", "scene_token": "scene1"},
            ],
            "sample_data": [],
        }
        scenes, info = build_scene_summaries(
            metadata,
            split_names={
                "train": {"scene-0000"},
                "val": {"scene-0001"},
                "_source": {"name": "unit"},
            },
        )
        self.assertFalse(info["fallback_split_used"])
        self.assertEqual([scene.split for scene in sorted(scenes, key=lambda scene: scene.name)], ["train", "val"])

    def test_collect_keyframes_and_lidar_sweeps(self) -> None:
        metadata = {
            "sample": [
                {
                    "token": "sample0",
                }
            ],
            "sample_data": [
                {
                    "token": "lidar0",
                    "sample_token": "sample0",
                    "filename": "samples/LIDAR_TOP/0.pcd.bin",
                    "is_key_frame": True,
                    "prev": "lidar_prev",
                },
                {
                    "token": "cam0",
                    "sample_token": "sample0",
                    "filename": "samples/CAM_FRONT/0.jpg",
                    "is_key_frame": True,
                    "prev": "",
                },
                {
                    "token": "lidar_prev",
                    "sample_token": "sample_prev",
                    "filename": "sweeps/LIDAR_TOP/prev.pcd.bin",
                    "is_key_frame": False,
                    "prev": "",
                },
            ],
        }
        files = collect_required_sample_data_filenames(
            metadata,
            {"sample0"},
            include_sweeps=True,
            max_sweeps=1,
        )
        self.assertEqual(
            files,
            [
                "samples/CAM_FRONT/0.jpg",
                "samples/LIDAR_TOP/0.pcd.bin",
                "sweeps/LIDAR_TOP/prev.pcd.bin",
            ],
        )

    def test_blob_manifest_resumes_scanned_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tgz(root / "v1.0-trainval01_blobs.tgz", {"samples/a.bin": b"a"})
            _write_tgz(root / "v1.0-trainval02_blobs.tgz", {"samples/b.bin": b"b"})
            output = root / "manifest.json"
            output.write_text(
                json.dumps(
                    {
                        "mapping": {"samples/a.bin": "v1.0-trainval01_blobs.tgz"},
                        "scanned_archives": ["v1.0-trainval01_blobs.tgz"],
                    }
                )
            )

            manifest = build_blob_manifest(root, ["samples/a.bin", "samples/b.bin"], output)

            self.assertEqual(manifest["found_count"], 2)
            self.assertEqual(manifest["missing_count"], 0)
            self.assertEqual(manifest["archive_count_this_run"], 1)
            self.assertEqual(
                manifest["mapping"],
                {
                    "samples/a.bin": "v1.0-trainval01_blobs.tgz",
                    "samples/b.bin": "v1.0-trainval02_blobs.tgz",
                },
            )


    def test_resumable_blob_extract_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            _write_tgz(root / "v1.0-trainval01_blobs.tgz", {"samples/a.bin": b"a"})
            _write_tgz(root / "v1.0-trainval02_blobs.tgz", {"samples/b.bin": b"b"})
            manifest_path = root / "extract_manifest.json"

            manifest = extract_blob_files_resumable(
                root,
                target,
                ["samples/a.bin", "samples/b.bin"],
                manifest_path,
            )

            self.assertEqual(manifest["extracted_count"], 2)
            self.assertTrue((target / "samples" / "a.bin").exists())
            self.assertTrue((target / "samples" / "b.bin").exists())
            self.assertTrue(manifest_path.exists())


    def test_safe_zip_extract_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract_zip_archive(archive_path, Path(tmp) / "target")


    def test_safe_extract_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "bad.tgz"
            with tarfile.open(archive_path, "w:gz") as tar:
                payload = b"bad"
                member = tarfile.TarInfo("../escape.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
            with tarfile.open(archive_path, "r:gz") as tar:
                member = tar.next()
                assert member is not None
                with self.assertRaises(ValueError):
                    safe_extract_member(tar, member, Path(tmp) / "target")


def _write_tgz(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))


if __name__ == "__main__":
    unittest.main()

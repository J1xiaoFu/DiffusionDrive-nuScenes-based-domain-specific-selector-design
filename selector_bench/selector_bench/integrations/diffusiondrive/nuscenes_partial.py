from __future__ import annotations

import json
import os
import random
import shutil
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from selector_bench.utils.io import ensure_dir, write_json


PARTIAL_PLAN_SCHEMA = "selector_bench.nuscenes_partial_plan.v0"
METADATA_ARCHIVE_NAME = "v1.0-trainval_meta.tgz"


@dataclass(frozen=True)
class SceneSummary:
    token: str
    name: str
    location: str
    split: str
    sample_count: int


def load_trainval_metadata(source_trainval_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load nuScenes trainval metadata JSON tables directly from the meta tgz."""
    source = Path(source_trainval_dir)
    archive = source / METADATA_ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(f"Missing nuScenes metadata archive: {archive}")

    tables: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name)
            if len(name.parts) != 2 or name.parts[0] != "v1.0-trainval" or name.suffix != ".json":
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            tables[name.stem] = json.load(extracted)
    required = {"scene", "log", "sample", "sample_data"}
    missing = sorted(required.difference(tables))
    if missing:
        raise ValueError(f"Metadata archive is missing required tables: {missing}")
    return tables


def build_scene_summaries(
    metadata: dict[str, list[dict[str, Any]]],
    split_names: dict[str, set[str]] | None = None,
) -> tuple[list[SceneSummary], dict[str, Any]]:
    split_names = split_names or load_nuscenes_trainval_split_names()
    log_by_token = {row["token"]: row for row in metadata["log"]}
    sample_counts: dict[str, int] = defaultdict(int)
    for sample in metadata["sample"]:
        sample_counts[str(sample["scene_token"])] += 1

    scenes: list[SceneSummary] = []
    unknown = 0
    for scene in metadata["scene"]:
        name = str(scene["name"])
        if name in split_names.get("train", set()):
            split = "train"
        elif name in split_names.get("val", set()):
            split = "val"
        else:
            split = "unknown"
            unknown += 1
        log = log_by_token.get(str(scene.get("log_token")), {})
        scenes.append(
            SceneSummary(
                token=str(scene["token"]),
                name=name,
                location=str(log.get("location", "unknown")),
                split=split,
                sample_count=int(scene.get("nbr_samples") or sample_counts[str(scene["token"])]),
            )
        )

    fallback_used = False
    if scenes and unknown == len(scenes):
        scenes = _apply_sorted_scene_fallback_split(scenes)
        fallback_used = True

    return scenes, {
        "split_source": split_names.get("_source", {"name": "provided"}),
        "fallback_split_used": fallback_used,
        "unknown_scene_count": unknown,
    }


def load_nuscenes_trainval_split_names() -> dict[str, set[str]]:
    try:
        from nuscenes.utils.splits import create_splits_scenes

        splits = create_splits_scenes()
        return {
            "train": set(splits["train"]),
            "val": set(splits["val"]),
            "_source": {"name": "nuscenes.utils.splits.create_splits_scenes"},
        }
    except Exception:
        return {
            "train": set(),
            "val": set(),
            "_source": {
                "name": "fallback_sorted_scene_80_20",
                "caveat": "nuscenes-devkit split lists were not importable.",
            },
        }


def select_balanced_scenes(
    scenes: list[SceneSummary],
    split: str,
    target_samples: int,
    seed: int,
) -> list[SceneSummary]:
    if target_samples <= 0:
        return []
    groups: dict[str, list[SceneSummary]] = defaultdict(list)
    for scene in scenes:
        if scene.split == split:
            groups[scene.location].append(scene)
    if not groups:
        return []

    rng = random.Random(seed)
    for location, values in groups.items():
        values.sort(key=lambda scene: (scene.sample_count, scene.name))
        rng.shuffle(values)

    offsets = {location: 0 for location in groups}
    selected: list[SceneSummary] = []
    selected_count = 0
    selected_by_location = {location: 0 for location in groups}
    while selected_count < target_samples:
        made_progress = False
        ordered_locations = sorted(
            groups,
            key=lambda location: (selected_by_location[location], location),
        )
        for location in ordered_locations:
            offset = offsets[location]
            if offset >= len(groups[location]):
                continue
            scene = groups[location][offset]
            offsets[location] += 1
            selected.append(scene)
            selected_count += scene.sample_count
            selected_by_location[location] += scene.sample_count
            made_progress = True
            if selected_count >= target_samples:
                break
        if not made_progress:
            break
    selected.sort(key=lambda scene: scene.name)
    return selected


def build_partial_plan(
    source_trainval_dir: str | Path,
    target_root: str | Path,
    target_train_samples: int,
    target_val_samples: int,
    seed: int,
    version: str = "v1.0-trainval",
    include_sweeps: bool = False,
    max_sweeps: int = 0,
) -> dict[str, Any]:
    metadata = load_trainval_metadata(source_trainval_dir)
    scenes, split_info = build_scene_summaries(metadata)
    train_scenes = select_balanced_scenes(scenes, "train", target_train_samples, seed)
    val_scenes = select_balanced_scenes(scenes, "val", target_val_samples, seed + 1009)

    train_scene_tokens = {scene.token for scene in train_scenes}
    val_scene_tokens = {scene.token for scene in val_scenes}
    train_sample_tokens = _sample_tokens_for_scenes(metadata, train_scene_tokens)
    val_sample_tokens = _sample_tokens_for_scenes(metadata, val_scene_tokens)
    selected_sample_tokens = set(train_sample_tokens).union(val_sample_tokens)
    required = collect_required_sample_data_filenames(
        metadata,
        selected_sample_tokens,
        include_sweeps=include_sweeps,
        max_sweeps=max_sweeps,
    )

    warnings: list[str] = []
    if split_info["fallback_split_used"]:
        warnings.append("nuScenes split lists were unavailable; used sorted-scene 80/20 fallback.")
    if not include_sweeps:
        warnings.append("required_data_filenames contains key-frame sample data only; lidar sweeps are disabled.")

    return {
        "schema": PARTIAL_PLAN_SCHEMA,
        "version": version,
        "source_trainval_dir": str(Path(source_trainval_dir)),
        "target_root": str(Path(target_root)),
        "seed": seed,
        "requested": {
            "target_train_samples": target_train_samples,
            "target_val_samples": target_val_samples,
            "whole_scene_selection": True,
        },
        "split_info": split_info,
        "table_counts": {name: len(rows) for name, rows in sorted(metadata.items())},
        "selection": {
            "train": _selection_summary(train_scenes, train_sample_tokens),
            "val": _selection_summary(val_scenes, val_sample_tokens),
        },
        "sample_tokens": {
            "train": train_sample_tokens,
            "val": val_sample_tokens,
        },
        "required_data": {
            "include_sweeps": include_sweeps,
            "max_sweeps": max_sweeps,
            "filename_count": len(required),
            "filenames": required,
        },
        "warnings": warnings,
    }


def collect_required_sample_data_filenames(
    metadata: dict[str, list[dict[str, Any]]],
    sample_tokens: set[str],
    include_sweeps: bool = False,
    max_sweeps: int = 0,
) -> list[str]:
    sample_by_token = {str(row["token"]): row for row in metadata["sample"]}
    sample_data_by_token = {str(row["token"]): row for row in metadata["sample_data"]}
    sample_data_by_sample_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata["sample_data"]:
        sample_data_by_sample_token[str(row.get("sample_token", ""))].append(row)
    required: set[str] = set()

    for sample_token in sorted(sample_tokens):
        sample = sample_by_token.get(sample_token)
        lidar_tokens: list[str] = []
        for row in sample_data_by_sample_token.get(sample_token, []):
            if _is_key_frame_sample_data(row):
                required.add(str(row["filename"]))
                if _is_lidar_top_sample_data(row):
                    lidar_tokens.append(str(row["token"]))

        if sample:
            # nuScenes devkit adds sample["data"] at runtime. Raw JSON does not
            # contain it, but tests or converted metadata may.
            for sample_data_token in sample.get("data", {}).values():
                row = sample_data_by_token.get(str(sample_data_token))
                if row:
                    required.add(str(row["filename"]))
                    if _is_lidar_top_sample_data(row):
                        lidar_tokens.append(str(row["token"]))
        if include_sweeps and max_sweeps > 0:
            for lidar_token in lidar_tokens:
                _add_prev_sweeps(required, sample_data_by_token, lidar_token, max_sweeps)
    return sorted(required)


def _is_key_frame_sample_data(row: dict[str, Any]) -> bool:
    return bool(row.get("is_key_frame", False)) or str(row.get("filename", "")).startswith("samples/")


def _is_lidar_top_sample_data(row: dict[str, Any]) -> bool:
    filename = str(row.get("filename", ""))
    return filename.startswith("samples/LIDAR_TOP/") or filename.startswith("sweeps/LIDAR_TOP/")


def extract_metadata_archive(source_trainval_dir: str | Path, target_root: str | Path) -> None:
    source = Path(source_trainval_dir) / METADATA_ARCHIVE_NAME
    target = ensure_dir(target_root)
    with tarfile.open(source, "r:gz") as tar:
        for member in tar:
            safe_extract_member(tar, member, target)


def extract_zip_archive(zip_path: str | Path, target_root: str | Path) -> dict[str, Any]:
    zip_file = Path(zip_path)
    target = ensure_dir(target_root)
    extracted = 0
    with zipfile.ZipFile(zip_file) as archive:
        for info in archive.infolist():
            name = info.filename
            resolved = (target / name).resolve()
            if not _is_relative_to(resolved, target.resolve()):
                raise ValueError(f"Refusing to extract zip member outside target root: {name}")
            if info.is_dir():
                ensure_dir(resolved)
                continue
            ensure_dir(resolved.parent)
            with archive.open(info) as src, resolved.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return {"zip_path": str(zip_file), "extracted_files": extracted}


def build_blob_manifest(
    source_trainval_dir: str | Path,
    required_filenames: list[str],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    required = set(required_filenames)
    archives = sorted(Path(source_trainval_dir).glob("v1.0-trainval*_blobs.tgz"))
    output = Path(output_path) if output_path else None
    mapping: dict[str, str] = {}
    scanned_archives: set[str] = set()

    if output and output.exists():
        with output.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_mapping = existing.get("mapping", {})
        mapping.update(
            {
                str(filename): str(archive_name)
                for filename, archive_name in existing_mapping.items()
                if str(filename) in required
            }
        )
        scanned_archives.update(str(name) for name in existing.get("scanned_archives", []))

    remaining = set(required).difference(mapping)
    archive_count_this_run = 0

    for archive in archives:
        if not remaining:
            break
        if archive.name in scanned_archives:
            continue
        archive_count_this_run += 1
        before = len(remaining)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.name in remaining:
                    mapping[member.name] = archive.name
                    remaining.remove(member.name)
                    if not remaining:
                        break
        scanned_archives.add(archive.name)
        found_delta = before - len(remaining)
        if output:
            write_json(
                output,
                _blob_manifest_payload(
                    source_trainval_dir,
                    archives,
                    required,
                    mapping,
                    remaining,
                    scanned_archives,
                    archive_count_this_run,
                ),
            )
        print(
            f"blob_manifest_progress archive={archive.name} found_delta={found_delta} "
            f"found={len(mapping)} remaining={len(remaining)}",
            flush=True,
        )

    manifest = _blob_manifest_payload(
        source_trainval_dir,
        archives,
        required,
        mapping,
        remaining,
        scanned_archives,
        archive_count_this_run,
    )
    if output:
        write_json(output, manifest)
    return manifest


def _blob_manifest_payload(
    source_trainval_dir: str | Path,
    archives: list[Path],
    required: set[str],
    mapping: dict[str, str],
    remaining: set[str],
    scanned_archives: set[str],
    archive_count_this_run: int,
) -> dict[str, Any]:
    all_archive_names = {archive.name for archive in archives}
    complete = not remaining or all_archive_names.issubset(scanned_archives)
    return {
        "source_trainval_dir": str(Path(source_trainval_dir)),
        "archive_count_total": len(archives),
        "archive_count_scanned": len(scanned_archives),
        "archive_count_this_run": archive_count_this_run,
        "scanned_archives": sorted(scanned_archives),
        "complete": complete,
        "required_count": len(required),
        "found_count": len(mapping),
        "missing_count": len(remaining),
        "missing": sorted(remaining),
        "mapping": dict(sorted(mapping.items())),
    }


def extract_blob_files(
    source_trainval_dir: str | Path,
    target_root: str | Path,
    required_filenames: list[str],
    blob_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = blob_manifest or build_blob_manifest(source_trainval_dir, required_filenames)
    mapping = manifest["mapping"]
    target = ensure_dir(target_root)
    by_archive: dict[str, list[str]] = defaultdict(list)
    for filename, archive_name in mapping.items():
        by_archive[archive_name].append(filename)

    extracted = 0
    for archive_name, filenames in sorted(by_archive.items()):
        wanted = set(filenames)
        archive = Path(source_trainval_dir) / archive_name
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.name not in wanted:
                    continue
                safe_extract_member(tar, member, target)
                extracted += 1
    return {
        "requested_count": len(required_filenames),
        "found_count": len(mapping),
        "extracted_count": extracted,
        "missing_count": int(manifest["missing_count"]),
    }


def extract_blob_files_resumable(
    source_trainval_dir: str | Path,
    target_root: str | Path,
    required_filenames: list[str],
    manifest_output_path: str | Path,
) -> dict[str, Any]:
    """Extract required blob files while checkpointing after each archive.

    This avoids a full manifest pass followed by a second extraction pass.
    If interrupted mid-archive, rerunning will rescan that archive and overwrite
    any already-extracted members from the incomplete archive.
    """
    required = set(required_filenames)
    archives = sorted(Path(source_trainval_dir).glob("v1.0-trainval*_blobs.tgz"))
    target = ensure_dir(target_root)
    output = Path(manifest_output_path)
    mapping: dict[str, str] = {}
    scanned_archives: set[str] = set()

    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        mapping.update(
            {
                str(filename): str(archive_name)
                for filename, archive_name in existing.get("mapping", {}).items()
                if str(filename) in required
            }
        )
        scanned_archives.update(str(name) for name in existing.get("scanned_archives", []))

    remaining = set(required).difference(mapping)
    archive_count_this_run = 0
    extracted_this_run = 0
    for archive in archives:
        if not remaining:
            break
        if archive.name in scanned_archives:
            continue
        archive_count_this_run += 1
        before = len(remaining)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.name not in remaining:
                    continue
                safe_extract_member(tar, member, target)
                mapping[member.name] = archive.name
                remaining.remove(member.name)
                extracted_this_run += 1
                if not remaining:
                    break
        scanned_archives.add(archive.name)
        found_delta = before - len(remaining)
        manifest = _blob_manifest_payload(
            source_trainval_dir,
            archives,
            required,
            mapping,
            remaining,
            scanned_archives,
            archive_count_this_run,
        )
        manifest["extracted_count"] = len(mapping)
        manifest["extracted_this_run"] = extracted_this_run
        write_json(output, manifest)
        print(
            f"blob_extract_progress archive={archive.name} extracted_delta={found_delta} "
            f"extracted={len(mapping)} remaining={len(remaining)}",
            flush=True,
        )

    manifest = _blob_manifest_payload(
        source_trainval_dir,
        archives,
        required,
        mapping,
        remaining,
        scanned_archives,
        archive_count_this_run,
    )
    manifest["extracted_count"] = len(mapping)
    manifest["extracted_this_run"] = extracted_this_run
    write_json(output, manifest)
    return manifest


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target_root: str | Path) -> Path:
    target_root_path = Path(target_root).resolve()
    member_path = target_root_path / member.name
    resolved = member_path.resolve()
    if not _is_relative_to(resolved, target_root_path):
        raise ValueError(f"Refusing to extract outside target root: {member.name}")
    if member.islnk() or member.issym():
        raise ValueError(f"Refusing to extract link member: {member.name}")
    tar.extract(member, path=target_root_path)
    return resolved


def _apply_sorted_scene_fallback_split(scenes: list[SceneSummary]) -> list[SceneSummary]:
    sorted_scenes = sorted(scenes, key=lambda scene: scene.name)
    train_cut = int(len(sorted_scenes) * 0.8)
    train_names = {scene.name for scene in sorted_scenes[:train_cut]}
    return [
        SceneSummary(
            token=scene.token,
            name=scene.name,
            location=scene.location,
            split="train" if scene.name in train_names else "val",
            sample_count=scene.sample_count,
        )
        for scene in scenes
    ]


def _sample_tokens_for_scenes(
    metadata: dict[str, list[dict[str, Any]]],
    scene_tokens: set[str],
) -> list[str]:
    rows = [
        row
        for row in metadata["sample"]
        if str(row["scene_token"]) in scene_tokens
    ]
    rows.sort(key=lambda row: (str(row["scene_token"]), int(row.get("timestamp", 0)), str(row["token"])))
    return [str(row["token"]) for row in rows]


def _selection_summary(scenes: list[SceneSummary], sample_tokens: list[str]) -> dict[str, Any]:
    location_counts: dict[str, int] = defaultdict(int)
    scene_counts: dict[str, int] = defaultdict(int)
    for scene in scenes:
        location_counts[scene.location] += scene.sample_count
        scene_counts[scene.location] += 1
    return {
        "scene_count": len(scenes),
        "sample_count": len(sample_tokens),
        "sample_count_by_location": dict(sorted(location_counts.items())),
        "scene_count_by_location": dict(sorted(scene_counts.items())),
        "scene_names": [scene.name for scene in scenes],
        "scenes": [
            {
                "token": scene.token,
                "name": scene.name,
                "location": scene.location,
                "sample_count": scene.sample_count,
            }
            for scene in scenes
        ],
    }


def _add_prev_sweeps(
    required: set[str],
    sample_data_by_token: dict[str, dict[str, Any]],
    lidar_token: Any,
    max_sweeps: int,
) -> None:
    prev_token = str(lidar_token or "")
    for _ in range(max_sweeps):
        current = sample_data_by_token.get(prev_token)
        if not current:
            break
        prev_token = str(current.get("prev") or "")
        if not prev_token:
            break
        prev = sample_data_by_token.get(prev_token)
        if not prev:
            break
        required.add(str(prev["filename"]))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

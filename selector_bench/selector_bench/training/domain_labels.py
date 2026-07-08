from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from selector_bench.core.manifest import Manifest, ManifestRecord
from selector_bench.utils.io import read_json, read_jsonl, write_jsonl


def resolve_labels(
    manifest: Manifest,
    source: dict[str, Any],
    split: str = "train",
    output_path: str | Path | None = None,
) -> dict[str, str]:
    """Resolve per-sample domain/scenario labels from manifest fields or sidecars."""
    kind = source.get("type", source.get("domain_source", "manifest_field"))
    default_label = str(source.get("default_label", source.get("default_domain", "default")))
    records = manifest.filter(split=split)

    if kind in {"manifest_field", "metadata_field"}:
        field = str(source.get("field", source.get("domain_key", "domain_name")))
        labels = {
            record.sample_token: _string_or_default(_lookup_record_field(record, field, metadata_only=kind == "metadata_field"), default_label)
            for record in records
        }
    elif kind == "sidecar_labels":
        path = source.get("path")
        if not path:
            raise ValueError("sidecar_labels source requires a path.")
        key_field = str(source.get("key_field", "sample_token"))
        label_field = str(source.get("label_field", source.get("domain_key", "domain_label")))
        sidecar = _load_sidecar_labels(path, key_field=key_field, label_field=label_field)
        labels = {
            record.sample_token: _string_or_default(sidecar.get(record.sample_token), default_label)
            for record in records
        }
    else:
        raise ValueError(f"Unsupported label source type: {kind}")

    if output_path is not None:
        write_resolved_labels(output_path, labels, manifest, split=split, source=source)
    return labels


def write_resolved_labels(
    output_path: str | Path,
    labels: dict[str, str],
    manifest: Manifest,
    split: str = "train",
    source: dict[str, Any] | None = None,
) -> None:
    rows = []
    source = source or {}
    for record in manifest.filter(split=split):
        label = labels.get(record.sample_token)
        rows.append(
            {
                "sample_token": record.sample_token,
                "split": record.split,
                "domain_label": label,
                "domain_source": source.get("type", source.get("domain_source", "unknown")),
                "domain_key": source.get("field", source.get("domain_key", source.get("label_field", ""))),
            }
        )
    write_jsonl(output_path, rows)


def load_label_jsonl(path: str | Path, label_field: str = "domain_label") -> dict[str, str]:
    rows = read_jsonl(path)
    labels: dict[str, str] = {}
    for row in rows:
        token = row.get("sample_token")
        if token is None:
            continue
        label = row.get(label_field, row.get("cluster_label", row.get("cluster_id")))
        if label is not None:
            labels[str(token)] = str(label)
    return labels


def _lookup_record_field(record: ManifestRecord, field: str, metadata_only: bool = False) -> Any:
    if metadata_only:
        return _lookup_nested(record.metadata, field)
    if hasattr(record, field):
        return getattr(record, field)
    if field.startswith("metadata."):
        return _lookup_nested(record.metadata, field[len("metadata.") :])
    return _lookup_nested(record.metadata, field)


def _lookup_nested(data: dict[str, Any] | None, path: str) -> Any:
    current: Any = data or {}
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _load_sidecar_labels(path: str | Path, key_field: str, label_field: str) -> dict[str, str]:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(target)
    elif suffix == ".json":
        raw = read_json(target)
        if isinstance(raw, dict):
            rows = [
                {key_field: key, label_field: value}
                if not isinstance(value, dict)
                else {key_field: key, **value}
                for key, value in raw.items()
            ]
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError(f"Unsupported JSON sidecar shape: {target}")
    elif suffix == ".csv":
        with target.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported sidecar label file type: {target}")

    labels: dict[str, str] = {}
    for row in rows:
        if key_field not in row or label_field not in row:
            continue
        labels[str(row[key_field])] = str(row[label_field])
    return labels


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.training.domain_labels import load_label_jsonl, resolve_labels
from selector_bench.training.stage1_feature_transforms import compute_raw_feature, normalize_feature_values
from selector_bench.utils.hashing import hash_jsonable
from selector_bench.utils.io import read_json, write_json, write_jsonl


def load_stage1_target_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            return json.loads(text)
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Stage 1 target config must be a mapping: {target}")
    if data.get("schema") != "selector_bench.stage1_target_config.v0":
        raise ValueError(f"Unsupported Stage 1 target config schema: {data.get('schema')}")
    return data


def build_stage1_domain_targets(
    config_path: str | Path,
    output_path: str | Path | None = None,
    labels_output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_stage1_target_config(config_path)
    feature_dir = config["feature_dir"]
    split = config.get("split", "train")
    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    records = store.manifest.filter(split=split)
    if not records:
        raise ValueError(f"No records found for split={split!r}.")

    labels = _resolve_config_labels(config, store, split=split, labels_output_path=labels_output_path)
    feature_specs = dict(config.get("features", {}))
    if not feature_specs:
        raise ValueError("Stage 1 target config requires features.")

    raw_by_feature = {name: [] for name in feature_specs}
    for record in records:
        features = store.get(record.sample_token)
        for name, spec in feature_specs.items():
            raw_by_feature[name].append(compute_raw_feature(record, features, spec))
    values_by_feature, normalizer = normalize_feature_values(raw_by_feature, feature_specs)

    heuristics = dict(config.get("heuristics", {}))
    default_heuristic = str(config.get("default_domain", "default"))
    if default_heuristic not in heuristics:
        raise ValueError(f"default_domain heuristic is missing: {default_heuristic}")
    config_hash = hash_jsonable(config)
    rows: list[dict[str, Any]] = []
    target_scores: list[float] = []
    for idx, record in enumerate(records):
        label = labels.get(record.sample_token, default_heuristic)
        heuristic_name = label if label in heuristics else default_heuristic
        fallback_used = heuristic_name != label
        feature_values = {name: float(values_by_feature[name][idx]) for name in feature_specs}
        score, terms = _evaluate_heuristic(feature_values, heuristics[heuristic_name])
        target_scores.append(score)
        rows.append(
            {
                "sample_token": record.sample_token,
                "split": record.split,
                "domain_label": label,
                "domain_source": _domain_source_name(config),
                "heuristic_name": heuristic_name,
                "target_score": score,
                "feature_values": feature_values,
                "score_terms": terms,
                "fallback_used": fallback_used,
                "config_hash": config_hash,
            }
        )

    if output_path is None:
        output_path = config.get("outputs", {}).get("target_scores_path")
    if output_path is None:
        raise ValueError("No output_path provided and outputs.target_scores_path is missing.")
    write_jsonl(output_path, rows)
    summary = _target_summary(rows, target_scores)
    metadata = {
        "metadata": make_artifact_metadata(
            artifact_type="stage1_target_scores",
            run_id=str(config.get("run_id", "stage1_domain_targets")),
            config={"config_path": str(config_path), "config_hash": config_hash},
            input_paths=[Path(feature_dir) / "manifest.jsonl", config_path],
            summary=summary,
        ),
        "feature_normalization": normalizer,
        "target_scores_path": str(output_path),
        "validation_boundary": {
            "stage1_targets_are": "cold_start_design_artifacts_only",
            "not_diffusiondrive_validation_evidence": True,
        },
    }
    write_json(str(output_path) + ".metadata.json", metadata)
    return {"rows": rows, "metadata": metadata}


def load_target_score_table(path: str | Path) -> tuple[list[str], np.ndarray, list[str], dict[str, dict[str, Any]]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    tokens = [str(row["sample_token"]) for row in rows]
    scores = np.asarray([float(row["target_score"]) for row in rows], dtype=np.float64)
    labels = [str(row.get("domain_label", row.get("heuristic_name", "default"))) for row in rows]
    by_token = {token: row for token, row in zip(tokens, rows)}
    return tokens, scores, labels, by_token


def _resolve_config_labels(
    config: dict[str, Any],
    store: FeatureStore,
    split: str,
    labels_output_path: str | Path | None,
) -> dict[str, str]:
    domain_source = config.get("domain_source", {"type": "manifest_field", "field": "domain_name"})
    if isinstance(domain_source, str):
        domain_source = {"type": domain_source, "field": config.get("domain_key", "domain_name")}
    domain_source = dict(domain_source)
    source_type = domain_source.get("type", domain_source.get("domain_source", "manifest_field"))
    if source_type == "cluster":
        labels_path = domain_source.get("labels_path", config.get("labels_path"))
        if not labels_path:
            raise ValueError("cluster domain_source requires labels_path.")
        return load_label_jsonl(labels_path)
    if source_type == "precomputed_labels":
        labels_path = domain_source.get("labels_path", config.get("labels_path"))
        if not labels_path:
            raise ValueError("precomputed_labels domain_source requires labels_path.")
        return load_label_jsonl(labels_path, label_field=str(domain_source.get("label_field", "domain_label")))
    if labels_output_path is None:
        labels_output_path = config.get("outputs", {}).get("resolved_labels_path")
    return resolve_labels(store.manifest, domain_source, split=split, output_path=labels_output_path)


def _domain_source_name(config: dict[str, Any]) -> str:
    source = config.get("domain_source", {})
    if isinstance(source, str):
        return source
    return str(source.get("type", source.get("domain_source", "manifest_field")))


def _evaluate_heuristic(feature_values: dict[str, float], heuristic: dict[str, Any]) -> tuple[float, dict[str, float]]:
    terms: dict[str, float] = {}
    score = float(heuristic.get("bias", 0.0))
    if score:
        terms["bias"] = score
    for name, weight in dict(heuristic.get("weights", {})).items():
        value = float(feature_values.get(name, 0.0))
        contribution = float(weight) * value
        terms[name] = contribution
        score += contribution
    clip = heuristic.get("clip")
    if clip is not None:
        score = float(np.clip(score, float(clip[0]), float(clip[1])))
    calibration = heuristic.get("calibration", {})
    score = score * float(calibration.get("scale", 1.0)) + float(calibration.get("offset", 0.0))
    return float(score), terms


def _target_summary(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    label_counts: dict[str, int] = {}
    heuristic_counts: dict[str, int] = {}
    fallback_count = 0
    for row in rows:
        label_counts[row["domain_label"]] = label_counts.get(row["domain_label"], 0) + 1
        heuristic_counts[row["heuristic_name"]] = heuristic_counts.get(row["heuristic_name"], 0) + 1
        fallback_count += int(bool(row.get("fallback_used")))
    return {
        "num_records": len(rows),
        "target_score_min": float(arr.min()) if arr.size else 0.0,
        "target_score_mean": float(arr.mean()) if arr.size else 0.0,
        "target_score_max": float(arr.max()) if arr.size else 0.0,
        "domain_label_counts": label_counts,
        "heuristic_counts": heuristic_counts,
        "fallback_count": fallback_count,
    }

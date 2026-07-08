from __future__ import annotations

import ast
import fcntl
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from selector_bench.core.baseline_adapter import DomainStratifiedAdapter
from selector_bench.core.feature_store import FeatureStore
from selector_bench.core.score_provider import RandomScoreProvider
from selector_bench.core.selection_runner import run_selection
from selector_bench.training.stage2_neural import (
    append_episode_record,
    build_episode_record,
    sample_gumbel_topk_selection,
    train_neural_selector_stage2_feedback,
)
from selector_bench.utils.io import ensure_dir, read_json, read_jsonl, write_json


MetricProvider = Callable[[int, dict[str, Any]], tuple[dict[str, float], dict[str, float]]]


def normalize_eval_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """Normalize DiffusionDrive planning metrics into Stage 2 episode keys."""

    l2_key = "l2" if "l2" in raw else "L2"
    if l2_key not in raw:
        raise ValueError(f"Missing L2/l2 metric in {raw}")
    metrics = {
        str(key): float(value)
        for key, value in raw.items()
        if key != "L2" and isinstance(value, (int, float))
    }
    metrics["l2"] = float(raw[l2_key])
    metrics.setdefault("obj_box_col", 0.0)
    metrics.setdefault("obj_col", 0.0)
    return metrics


def parse_eval_metrics(eval_log: str | Path, status_json: str | Path | None = None) -> dict[str, float]:
    """Parse the final planning metric dict emitted by DiffusionDrive eval logs."""

    log_path = Path(eval_log)
    if not log_path.exists():
        raise FileNotFoundError(f"Eval log does not exist: {log_path}")

    parsed: dict[str, Any] | None = None
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text.startswith("{") or "obj_box_col" not in text:
            continue
        try:
            candidate = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            continue
        if isinstance(candidate, dict) and ("L2" in candidate or "l2" in candidate):
            parsed = candidate
            break
    if parsed is None:
        raise ValueError(f"Could not find DiffusionDrive metrics dict in {log_path}")

    metrics = normalize_eval_metrics(parsed)
    if status_json is not None and Path(status_json).exists():
        status = read_json(status_json)
        if "runtime_seconds" in status:
            metrics["runtime_s"] = float(status["runtime_seconds"])
        peak = status.get("peak_gpu_memory_nvidia_smi_mb")
        if peak is not None:
            metrics["peak_gpu_mb"] = float(peak)
    return metrics


def export_diffusiondrive_info_subset(
    source_info: str | Path,
    selection_path: str | Path,
    output_path: str | Path,
    metadata_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write a DiffusionDrive info PKL filtered to a selection artifact."""

    source = Path(source_info)
    selection = Path(selection_path)
    output = Path(output_path)
    artifact = read_json(selection)
    selected_tokens = set(artifact["selection"]["selected_tokens"])

    with source.open("rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"]
    subset_infos = [info for info in infos if info.get("token") in selected_tokens]
    found_tokens = {info.get("token") for info in subset_infos}
    missing = sorted(selected_tokens - found_tokens)
    if missing:
        raise ValueError(f"{len(missing)} selected tokens were not found in {source}: {missing[:5]}")

    exported = dict(payload)
    exported["infos"] = subset_infos
    metadata = dict(exported.get("metadata", {}))
    metadata.update(
        {
            "subset_source_info": str(source),
            "subset_source_selection": str(selection),
            "subset_selected_count": len(subset_infos),
            "subset_order": "source_info_order",
        }
    )
    exported["metadata"] = metadata

    ensure_dir(output.parent)
    with output.open("wb") as f:
        pickle.dump(exported, f)

    sidecar = {
        "source_info": str(source),
        "selection": str(selection),
        "output": str(output),
        "source_count": len(infos),
        "selected_count": len(subset_infos),
        "subset_order": "source_info_order",
    }
    if metadata_json is not None:
        write_json(metadata_json, sidecar)
    return sidecar


def write_subset_train_config(
    base_config: str | Path,
    output_config: str | Path,
    train_ann_file: str | Path,
    diffusiondrive_root: str | Path,
    sequences_split_num: int = 1,
    planner_init_checkpoint: str | Path | None = None,
    planner_lr: float | None = None,
    planner_max_iters: int | None = None,
) -> Path:
    """Write the short DiffusionDrive config used for one exact-token subset."""

    if sequences_split_num < 1:
        raise ValueError("sequences_split_num must be positive.")
    if planner_lr is not None and planner_lr <= 0:
        raise ValueError("planner_lr must be positive.")
    if planner_max_iters is not None and planner_max_iters < 1:
        raise ValueError("planner_max_iters must be positive.")
    base = Path(base_config).resolve()
    output = Path(output_config)
    ann_ref = _diffusiondrive_ref(train_ann_file, diffusiondrive_root)
    planner_ref = (
        _diffusiondrive_ref(planner_init_checkpoint, diffusiondrive_root)
        if planner_init_checkpoint is not None
        else None
    )
    base_ref = _relative_ref(base, output.parent)
    lines = [
        "_base_ = [",
        f"    {json.dumps(base_ref)},",
        "]",
        "",
        "data = dict(",
        "    train=dict(",
        f"        ann_file={json.dumps(ann_ref)},",
        f"        sequences_split_num={int(sequences_split_num)},",
        "    ),",
        ")",
        "",
    ]
    if planner_lr is not None:
        lines.extend(
            [
                "# Conservative planner fine-tuning overrides.",
                f"optimizer = dict(lr={float(planner_lr)!r})",
                'lr_config = dict(_delete_=True, policy="CosineAnnealing", warmup=None, min_lr_ratio=1.0)',
                "",
            ]
        )
    if planner_max_iters is not None:
        max_iters = int(planner_max_iters)
        log_interval = max(1, max_iters // 10)
        lines.extend(
            [
                f'runner = dict(type="IterBasedRunner", max_iters={max_iters})',
                f"evaluation = dict(interval={max_iters})",
                f"checkpoint_config = dict(interval={max_iters})",
                f"log_config = dict(interval={log_interval})",
                "",
            ]
        )
    if planner_ref is not None:
        lines.extend(
            [
                "# Fixed planner initialization for selector-quality feedback.",
                f"load_from = {json.dumps(planner_ref)}",
                "resume_from = None",
                "",
            ]
        )
    text = "\n".join(lines)
    ensure_dir(output.parent)
    output.write_text(text, encoding="utf-8")
    return output


def run_stage2_high_frequency_update(
    feature_dir: str | Path,
    init_selector_path: str | Path,
    episode_jsonl: str | Path,
    output_dir: str | Path,
    source_info: str | Path,
    diffusiondrive_root: str | Path,
    base_config: str | Path,
    run_id: str = "stage2_high_frequency",
    rounds: int = 1,
    selection_count: int = 323,
    split: str = "train",
    temperature: float = 0.15,
    sample_seed: int = 4100,
    random_seed: int = 5100,
    selector_seed: int = 6100,
    train_seed: int = 0,
    eval_seed: int = 0,
    selector_epochs: int = 40,
    selector_lr: float = 0.01,
    selector_l2: float = 1e-4,
    reward_scale: float | None = None,
    target_clip: float = 2.0,
    min_abs_reward: float = 0.0,
    negative_token_penalty: float = 0.0,
    negative_random_bonus: float = 0.0,
    negative_reward_max: float = -1e-12,
    replay_episode_jsonl: str | Path | None = None,
    replay_penalty: float = 0.0,
    replay_reward_max: float = -1e-12,
    rho_box: float = 0.05,
    rho_obj: float = 0.05,
    subset_info_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    work_dir_root: str | Path | None = None,
    state_json: str | Path | None = None,
    execute_diffusiondrive: bool = False,
    metric_provider: MetricProvider | None = None,
    diffusiondrive_python: str | Path | None = None,
    wrapper_python: str | Path | None = None,
    planner_init_checkpoint: str | Path | None = None,
    planner_lr: float | None = 1e-6,
    planner_max_iters: int | None = 100,
    disjoint_pair: bool = True,
    remove_random_tokens: bool = True,
    restart: bool = False,
) -> dict[str, Any]:
    """Run or prepare a Stage 2 feedback loop.

    One round selects selection_count samples with the current selector,
    compares them against a same-budget random subset, records one feedback
    episode, and immediately trains the selector used by the next round. The
    default is now a single 323-sample comparison because evaluation, not
    fine-tuning, is the runtime bottleneck.
    """

    if rounds < 1:
        raise ValueError("rounds must be positive.")
    if selection_count < 1:
        raise ValueError("selection_count must be positive.")
    if planner_lr is not None and planner_lr <= 0:
        raise ValueError("planner_lr must be positive.")
    if planner_max_iters is not None and planner_max_iters < 1:
        raise ValueError("planner_max_iters must be positive.")
    if execute_diffusiondrive and metric_provider is not None:
        raise ValueError("Use either execute_diffusiondrive or metric_provider, not both.")

    output_root = ensure_dir(output_dir)
    _lock_handle = _acquire_run_lock(output_root / f"{run_id}.lock")
    selection_root = ensure_dir(output_root / "selections")
    selector_root = ensure_dir(output_root / "selectors")
    sidecar_root = ensure_dir(output_root / "subset_metadata")
    dd_root = Path(diffusiondrive_root)
    subset_root = ensure_dir(subset_info_dir or dd_root / "data" / "infos" / "trainval_partial_selector_subsets")
    config_root = ensure_dir(config_dir or output_root / "configs")
    work_root = ensure_dir(work_dir_root or output_root / "work_dirs")
    state_path = Path(state_json) if state_json is not None else output_root / f"{run_id}_state.json"

    state = _initial_state(
        state_path=state_path,
        episode_jsonl=Path(episode_jsonl),
        run_id=run_id,
        init_selector_path=Path(init_selector_path),
        selector_root=selector_root,
        restart=restart,
    )
    used_tokens = set(state["used_tokens"])
    current_selector = Path(state["current_selector"])
    completed_rounds = int(state["completed_rounds"])
    round_records: list[dict[str, Any]] = list(state.get("rounds", []))
    existing_episode_ids = _episode_ids(episode_jsonl)

    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    report_status = "complete" if completed_rounds >= rounds else "running"

    for round_idx in range(completed_rounds + 1, rounds + 1):
        paths = _round_paths(
            run_id=run_id,
            round_idx=round_idx,
            selection_count=selection_count,
            selection_root=selection_root,
            selector_root=selector_root,
            subset_root=subset_root,
            sidecar_root=sidecar_root,
            config_root=config_root,
            work_root=work_root,
        )
        selector_selection = _ensure_selector_selection(
            feature_dir=feature_dir,
            selector_checkpoint=current_selector,
            output_path=paths["selector_selection"],
            run_id=f"{run_id}_r{round_idx:02d}_selector{selection_count}",
            selection_count=selection_count,
            split=split,
            temperature=temperature,
            seed=sample_seed + round_idx - 1,
            replay_episode_jsonl=replay_episode_jsonl,
            replay_penalty=replay_penalty,
            replay_reward_max=replay_reward_max,
            exclude_tokens=used_tokens,
        )
        selector_tokens = selector_selection["selection"]["selected_tokens"]
        random_exclusions = set(used_tokens)
        if disjoint_pair:
            random_exclusions.update(selector_tokens)
        random_selection = _ensure_random_selection(
            store=store,
            output_path=paths["random_selection"],
            run_id=f"{run_id}_r{round_idx:02d}_random{selection_count}",
            selection_count=selection_count,
            split=split,
            seed=random_seed + round_idx - 1,
            exclude_tokens=random_exclusions,
            feature_dir=feature_dir,
        )
        random_tokens = random_selection["selection"]["selected_tokens"]

        _ensure_subset_artifacts(
            source_info=source_info,
            selector_selection=paths["selector_selection"],
            random_selection=paths["random_selection"],
            selector_info=paths["selector_info"],
            random_info=paths["random_info"],
            selector_sidecar=paths["selector_sidecar"],
            random_sidecar=paths["random_sidecar"],
            base_config=base_config,
            selector_config=paths["selector_config"],
            random_config=paths["random_config"],
            diffusiondrive_root=dd_root,
            planner_init_checkpoint=planner_init_checkpoint,
            planner_lr=planner_lr,
            planner_max_iters=planner_max_iters,
        )

        context = {
            "round": round_idx,
            "selector_selection": str(paths["selector_selection"]),
            "random_selection": str(paths["random_selection"]),
            "selector_config": str(paths["selector_config"]),
            "random_config": str(paths["random_config"]),
            "selector_info": str(paths["selector_info"]),
            "random_info": str(paths["random_info"]),
            "selector_work_dir": str(paths["selector_work_dir"]),
            "random_work_dir": str(paths["random_work_dir"]),
            "selector_eval_work_dir": str(paths["selector_eval_work_dir"]),
            "random_eval_work_dir": str(paths["random_eval_work_dir"]),
        }
        round_records = [
            item for item in round_records if int(item.get("round", -1)) != round_idx
        ]

        episode_id = f"{run_id}_r{round_idx:02d}"
        if episode_id in existing_episode_ids:
            record = _episode_by_id(episode_jsonl, episode_id)
        else:
            if metric_provider is not None:
                selector_metrics, random_metrics = metric_provider(round_idx, context)
                train_work_dirs = {
                    "selector": str(paths["selector_work_dir"]),
                    "random_same_budget": str(paths["random_work_dir"]),
                }
                eval_work_dirs: dict[str, str] = {}
            elif execute_diffusiondrive:
                selector_metrics, random_metrics, train_work_dirs, eval_work_dirs = _run_diffusiondrive_pair(
                    paths=paths,
                    diffusiondrive_root=dd_root,
                    train_seed=train_seed,
                    eval_seed=eval_seed,
                    diffusiondrive_python=diffusiondrive_python,
                    wrapper_python=wrapper_python,
                )
            else:
                report_status = "prepared_round_needs_feedback"
                round_records.append(
                    {
                        **context,
                        "status": report_status,
                        "selector_hash": selector_selection["selection"]["selection_hash"],
                        "random_hash": random_selection["selection"]["selection_hash"],
                        "selector_count": selector_selection["selection"]["selected_count"],
                        "random_count": random_selection["selection"]["selected_count"],
                    }
                )
                _write_state(
                    state_path,
                    run_id,
                    current_selector,
                    used_tokens,
                    completed_rounds,
                    rounds,
                    selection_count,
                    round_records,
                    report_status,
                )
                break

            record = build_episode_record(
                episode_id=episode_id,
                selector_selection_path=paths["selector_selection"],
                random_selection_path=paths["random_selection"],
                selector_metrics=normalize_eval_metrics(selector_metrics),
                random_metrics=normalize_eval_metrics(random_metrics),
                rho_box=rho_box,
                rho_obj=rho_obj,
                train_work_dirs=train_work_dirs,
                eval_work_dirs=eval_work_dirs,
                notes=(
                    f"High-frequency Stage 2 micro-round {round_idx}/{rounds}; "
                    f"{selection_count} selector samples vs {selection_count} random samples."
                ),
            )
            if "loss_delta" in record.get("metrics", {}).get("selector", {}):
                record["reward_definition"] = {
                    "cost": "loss_proxy_cost; l2 field stores the lower-is-better scalar derived from train loss",
                    "reward": "R=loss_proxy_cost(random_same_budget)-loss_proxy_cost(selector)",
                    "rho_box": float(rho_box),
                    "rho_obj": float(rho_obj),
                    "source": "DiffusionDrive train-only loss proxy",
                }
            append_episode_record(episode_jsonl, record)
            existing_episode_ids.add(episode_id)

        if not paths["updated_selector"].exists():
            train_neural_selector_stage2_feedback(
                feature_dir=feature_dir,
                init_selector_path=current_selector,
                episode_jsonl=episode_jsonl,
                output_path=paths["updated_selector"],
                run_id=f"{run_id}_r{round_idx:02d}_selector_update",
                epochs=selector_epochs,
                lr=selector_lr,
                l2=selector_l2,
                reward_scale=reward_scale,
                target_clip=target_clip,
                min_abs_reward=min_abs_reward,
                negative_token_penalty=negative_token_penalty,
                negative_random_bonus=negative_random_bonus,
                negative_reward_max=negative_reward_max,
                seed=selector_seed + round_idx - 1,
            )
        current_selector = paths["updated_selector"]
        used_tokens.update(selector_tokens)
        if remove_random_tokens:
            used_tokens.update(random_tokens)
        completed_rounds = round_idx
        report_status = "complete" if completed_rounds >= rounds else "running"
        round_records.append(
            {
                **context,
                "status": "updated",
                "episode_id": episode_id,
                "reward": float(record["reward"]),
                "selector_hash": selector_selection["selection"]["selection_hash"],
                "random_hash": random_selection["selection"]["selection_hash"],
                "selector_count": selector_selection["selection"]["selected_count"],
                "random_count": random_selection["selection"]["selected_count"],
                "updated_selector": str(current_selector),
            }
        )
        _write_state(
            state_path,
            run_id,
            current_selector,
            used_tokens,
            completed_rounds,
            rounds,
            selection_count,
            round_records,
            report_status,
        )

    report = {
        "schema": "selector_bench.stage2_high_frequency_update.v0",
        "run_id": run_id,
        "state_json": str(state_path),
        "episode_jsonl": str(episode_jsonl),
        "current_selector": str(current_selector),
        "config": {
            "rounds": rounds,
            "selection_count": selection_count,
            "split": split,
            "temperature": temperature,
            "sample_seed": sample_seed,
            "random_seed": random_seed,
            "selector_epochs": selector_epochs,
            "selector_lr": selector_lr,
            "selector_l2": selector_l2,
            "rho_box": rho_box,
            "rho_obj": rho_obj,
            "execute_diffusiondrive": execute_diffusiondrive,
            "planner_init_checkpoint": str(planner_init_checkpoint) if planner_init_checkpoint is not None else None,
            "planner_lr": planner_lr,
            "planner_max_iters": planner_max_iters,
            "disjoint_pair": disjoint_pair,
            "remove_random_tokens": remove_random_tokens,
        },
        "summary": {
            "status": report_status,
            "completed_rounds": completed_rounds,
            "target_rounds": rounds,
            "selection_count": selection_count,
            "used_token_count": len(used_tokens),
        },
        "rounds": round_records,
    }
    write_json(output_root / f"{run_id}_report.json", report)
    return report


def _ensure_selector_selection(
    feature_dir: str | Path,
    selector_checkpoint: str | Path,
    output_path: Path,
    run_id: str,
    selection_count: int,
    split: str,
    temperature: float,
    seed: int,
    replay_episode_jsonl: str | Path | None,
    replay_penalty: float,
    replay_reward_max: float,
    exclude_tokens: set[str],
) -> dict[str, Any]:
    if output_path.exists():
        return read_json(output_path)
    return sample_gumbel_topk_selection(
        feature_dir=feature_dir,
        selector_checkpoint=selector_checkpoint,
        output_path=output_path,
        run_id=run_id,
        budget=float(selection_count),
        split=split,
        temperature=temperature,
        seed=seed,
        replay_episode_jsonl=replay_episode_jsonl,
        replay_penalty=replay_penalty,
        replay_reward_max=replay_reward_max,
        exclude_tokens=exclude_tokens,
    )


def _ensure_random_selection(
    store: FeatureStore,
    output_path: Path,
    run_id: str,
    selection_count: int,
    split: str,
    seed: int,
    exclude_tokens: set[str],
    feature_dir: str | Path,
) -> dict[str, Any]:
    if output_path.exists():
        return read_json(output_path)
    return run_selection(
        manifest=store.manifest,
        adapter=DomainStratifiedAdapter(),
        score_provider=RandomScoreProvider(seed=seed),
        budget=float(selection_count),
        output_path=output_path,
        run_id=run_id,
        split=split,
        config={
            "features": str(feature_dir),
            "baseline": "random",
            "score": "random",
            "budget": float(selection_count),
            "budget_count": selection_count,
            "seed": seed,
            "excluded_token_count": len(exclude_tokens),
        },
        input_paths=[Path(feature_dir) / "manifest.jsonl"],
        exclude_tokens=exclude_tokens,
    )


def _ensure_subset_artifacts(
    source_info: str | Path,
    selector_selection: Path,
    random_selection: Path,
    selector_info: Path,
    random_info: Path,
    selector_sidecar: Path,
    random_sidecar: Path,
    base_config: str | Path,
    selector_config: Path,
    random_config: Path,
    diffusiondrive_root: Path,
    planner_init_checkpoint: str | Path | None,
    planner_lr: float | None,
    planner_max_iters: int | None,
) -> None:
    if not selector_info.exists():
        export_diffusiondrive_info_subset(source_info, selector_selection, selector_info, selector_sidecar)
    if not random_info.exists():
        export_diffusiondrive_info_subset(source_info, random_selection, random_info, random_sidecar)
    if not selector_config.exists():
        write_subset_train_config(
            base_config,
            selector_config,
            selector_info,
            diffusiondrive_root,
            planner_init_checkpoint=planner_init_checkpoint,
            planner_lr=planner_lr,
            planner_max_iters=planner_max_iters,
        )
    if not random_config.exists():
        write_subset_train_config(
            base_config,
            random_config,
            random_info,
            diffusiondrive_root,
            planner_init_checkpoint=planner_init_checkpoint,
            planner_lr=planner_lr,
            planner_max_iters=planner_max_iters,
        )


def _run_diffusiondrive_pair(
    paths: dict[str, Path],
    diffusiondrive_root: Path,
    train_seed: int,
    eval_seed: int,
    diffusiondrive_python: str | Path | None,
    wrapper_python: str | Path | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, str], dict[str, str]]:
    selector_metrics = _run_diffusiondrive_one(
        config=paths["selector_config"],
        train_work_dir=paths["selector_work_dir"],
        eval_work_dir=paths["selector_eval_work_dir"],
        diffusiondrive_root=diffusiondrive_root,
        train_seed=train_seed,
        eval_seed=eval_seed,
        diffusiondrive_python=diffusiondrive_python,
        wrapper_python=wrapper_python,
    )
    random_metrics = _run_diffusiondrive_one(
        config=paths["random_config"],
        train_work_dir=paths["random_work_dir"],
        eval_work_dir=paths["random_eval_work_dir"],
        diffusiondrive_root=diffusiondrive_root,
        train_seed=train_seed,
        eval_seed=eval_seed,
        diffusiondrive_python=diffusiondrive_python,
        wrapper_python=wrapper_python,
    )
    return (
        selector_metrics,
        random_metrics,
        {
            "selector": str(paths["selector_work_dir"]),
            "random_same_budget": str(paths["random_work_dir"]),
        },
        {
            "selector": str(paths["selector_eval_work_dir"]),
            "random_same_budget": str(paths["random_eval_work_dir"]),
        },
    )


def _run_diffusiondrive_one(
    config: Path,
    train_work_dir: Path,
    eval_work_dir: Path,
    diffusiondrive_root: Path,
    train_seed: int,
    eval_seed: int,
    diffusiondrive_python: str | Path | None,
    wrapper_python: str | Path | None,
) -> dict[str, float]:
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    wrapper = str(wrapper_python or sys.executable)
    train_status = train_work_dir / "run_status.json"
    eval_status = eval_work_dir / "eval_status.json"

    if not _successful_train_status(train_status):
        cmd = [
            wrapper,
            str(scripts_root / "20_train_diffusiondrive_checkpoint.py"),
            "--diffusiondrive-root",
            str(diffusiondrive_root),
            "--config",
            str(config),
            "--work-dir",
            str(train_work_dir),
            "--seed",
            str(train_seed),
            "--no-validate",
            "--status-json",
            str(train_status),
        ]
        if diffusiondrive_python is not None:
            cmd.extend(["--python", str(diffusiondrive_python)])
        _run_checked(cmd)

    checkpoint = _checkpoint_from_status(train_status)
    if not _successful_eval_status(eval_status):
        cmd = [
            wrapper,
            str(scripts_root / "18_eval_diffusiondrive_checkpoint.py"),
            "--diffusiondrive-root",
            str(diffusiondrive_root),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--work-dir",
            str(eval_work_dir),
            "--seed",
            str(eval_seed),
            "--status-json",
            str(eval_status),
        ]
        if diffusiondrive_python is not None:
            cmd.extend(["--python", str(diffusiondrive_python)])
        _run_checked(cmd)

    metrics = parse_eval_metrics(eval_work_dir / "outer_eval.log", eval_status)
    train = read_json(train_status)
    eval_status_payload = read_json(eval_status)
    metrics["train_runtime_s"] = float(train.get("runtime_seconds", 0.0))
    metrics["eval_runtime_s"] = float(eval_status_payload.get("runtime_seconds", metrics.get("runtime_s", 0.0)))
    train_peak = train.get("peak_gpu_memory_nvidia_smi_mb")
    eval_peak = eval_status_payload.get("peak_gpu_memory_nvidia_smi_mb")
    peaks = [float(item) for item in [train_peak, eval_peak] if item is not None]
    if peaks:
        metrics["peak_gpu_mb"] = max(peaks)
    return metrics


def _run_checked(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd}")


def _successful_train_status(path: Path) -> bool:
    if not path.exists():
        return False
    status = read_json(path)
    return int(status.get("exit_code", 1)) == 0 and bool(status.get("checkpoint_files"))


def _successful_eval_status(path: Path) -> bool:
    if not path.exists():
        return False
    status = read_json(path)
    return int(status.get("exit_code", 1)) == 0


def _checkpoint_from_status(path: Path) -> Path:
    status = read_json(path)
    checkpoints = [Path(item) for item in status.get("checkpoint_files", [])]
    if not checkpoints:
        raise ValueError(f"No checkpoint files recorded in {path}")
    return sorted(checkpoints)[-1]


def _round_paths(
    run_id: str,
    round_idx: int,
    selection_count: int,
    selection_root: Path,
    selector_root: Path,
    subset_root: Path,
    sidecar_root: Path,
    config_root: Path,
    work_root: Path,
) -> dict[str, Path]:
    stem = f"{run_id}_r{round_idx:02d}"
    selector_name = f"{stem}_selector{selection_count}"
    random_name = f"{stem}_random{selection_count}"
    return {
        "selector_selection": selection_root / f"{selector_name}.json",
        "random_selection": selection_root / f"{random_name}.json",
        "selector_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{selector_name}.pkl",
        "random_info": subset_root / f"nuscenes_partial_1600_400_infos_train_{random_name}.pkl",
        "selector_sidecar": sidecar_root / f"{selector_name}_subset.json",
        "random_sidecar": sidecar_root / f"{random_name}_subset.json",
        "selector_config": config_root / f"dd_{selector_name}.py",
        "random_config": config_root / f"dd_{random_name}.py",
        "selector_work_dir": work_root / f"{selector_name}_subset_100iter",
        "random_work_dir": work_root / f"{random_name}_subset_100iter",
        "selector_eval_work_dir": work_root / f"{selector_name}_subset_100iter_eval_standalone",
        "random_eval_work_dir": work_root / f"{random_name}_subset_100iter_eval_standalone",
        "updated_selector": selector_root / f"{stem}_updated_selector.json",
    }


def _acquire_run_lock(path: Path) -> Any:
    ensure_dir(path.parent)
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another high-frequency Stage 2 run is already using {path}") from exc
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def _initial_state(
    state_path: Path,
    episode_jsonl: Path,
    run_id: str,
    init_selector_path: Path,
    selector_root: Path,
    restart: bool,
) -> dict[str, Any]:
    if state_path.exists() and not restart:
        return read_json(state_path)

    completed = 0
    used_tokens: set[str] = set()
    if episode_jsonl.exists():
        prefix = f"{run_id}_r"
        for row in read_jsonl(episode_jsonl):
            episode_id = str(row.get("episode_id", ""))
            if not episode_id.startswith(prefix):
                continue
            try:
                completed = max(completed, int(episode_id[len(prefix) : len(prefix) + 2]))
            except ValueError:
                continue
            for key in ["selector_selection", "random_same_budget_selection"]:
                path = row.get(key, {}).get("path")
                if path and Path(path).exists():
                    used_tokens.update(read_json(path)["selection"]["selected_tokens"])
    current_selector = (
        selector_root / f"{run_id}_r{completed:02d}_updated_selector.json"
        if completed > 0
        else init_selector_path
    )
    if completed > 0 and not current_selector.exists():
        current_selector = init_selector_path
        completed = 0
        used_tokens.clear()
    return {
        "run_id": run_id,
        "current_selector": str(current_selector),
        "completed_rounds": completed,
        "used_tokens": sorted(used_tokens),
        "rounds": [],
    }


def _write_state(
    state_path: Path,
    run_id: str,
    current_selector: Path,
    used_tokens: set[str],
    completed_rounds: int,
    target_rounds: int,
    selection_count: int,
    round_records: list[dict[str, Any]],
    status: str,
) -> None:
    write_json(
        state_path,
        {
            "schema": "selector_bench.stage2_high_frequency_state.v0",
            "run_id": run_id,
            "status": status,
            "current_selector": str(current_selector),
            "completed_rounds": completed_rounds,
            "target_rounds": target_rounds,
            "selection_count": selection_count,
            "used_tokens": sorted(used_tokens),
            "rounds": round_records,
        },
    )


def _episode_ids(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    return {str(row["episode_id"]) for row in read_jsonl(target)}


def _episode_by_id(path: str | Path, episode_id: str) -> dict[str, Any]:
    for row in read_jsonl(path):
        if str(row.get("episode_id")) == episode_id:
            return row
    raise KeyError(episode_id)


def _diffusiondrive_ref(path: str | Path, diffusiondrive_root: str | Path) -> str:
    target = Path(path).resolve()
    root = Path(diffusiondrive_root).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def _relative_ref(path: Path, base_dir: Path) -> str:
    base = Path(base_dir)
    try:
        return os.path.relpath(path, base.resolve()).replace(os.sep, "/")
    except FileNotFoundError:
        return os.path.relpath(path, base).replace(os.sep, "/")

#!/usr/bin/env python3
"""Create a content-addressed, clean-commit source bundle manifest for 06G."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


SOURCE_FILES = (
    "selector_bench/selector_bench/continual/__init__.py",
    "selector_bench/selector_bench/continual/claim_protocol.py",
    "selector_bench/selector_bench/continual/navsim_protocol.py",
    "selector_bench/selector_bench/continual/drive_opd.py",
    "selector_bench/selector_bench/continual/baselines.py",
    "selector_bench/selector_bench/continual/statistics.py",
    "selector_bench/selector_bench/continual/evaluation_contract.py",
    "selector_bench/scripts/40_build_drive_cl_protocol.py",
    "selector_bench/scripts/41_train_drive_cl_diffusiondrive.py",
    "selector_bench/scripts/42_export_drive_cl_eval_tokens.py",
    "selector_bench/scripts/43_audit_diffusiondrive_context_swap.py",
    "selector_bench/scripts/44_compute_drive_cl_fisher.py",
    "selector_bench/scripts/45_summarize_drive_cl_pdm.py",
    "selector_bench/scripts/46_compare_drive_cl_pdm.py",
    "selector_bench/scripts/47_export_drive_cl_training_manifest.py",
    "selector_bench/scripts/48_reconcile_drive_cl_inventories.py",
    "selector_bench/scripts/49_audit_drive_opd_rollout_schedule.py",
    "selector_bench/scripts/50_apply_drive_cl_global_holm.py",
    "selector_bench/scripts/51_freeze_drive_opd_06g_transfer.py",
    "selector_bench/scripts/52_compare_drive_cl_crossed_pdm.py",
    "selector_bench/scripts/53_audit_drive_cl_claim_protocol.py",
    "selector_bench/scripts/54_run_drive_cl_navsim_evaluator.py",
    "selector_bench/scripts/55_design_drive_cl_confirmatory_seeds.py",
    "selector_bench/scripts/56_build_drive_cl_evidence_inventory.py",
    "selector_bench/scripts/57_audit_drive_cl_p0_attacks.py",
    "selector_bench/scripts/58_register_drive_cl_run.py",
    "selector_bench/configs/drive_opd_native_cl_failure_patch_seed0_v1.json",
    "selector_bench/configs/drive_cl_crossed_comparison_spec.example.json",
    "selector_bench/configs/drive_cl_global_holm_family.example.json",
    "selector_bench/configs/drive_cl_seed_design.example.json",
    "selector_bench/configs/drive_cl_method_registry.v1.json",
    "selector_bench/configs/drive_cl_metric_registry.v1.json",
    "selector_bench/tests/test_drive_cl_baselines.py",
    "selector_bench/tests/test_drive_cl_statistics.py",
    "selector_bench/tests/test_drive_opd_losses.py",
    "selector_bench/tests/test_navsim_continual_protocol.py",
    "selector_bench/tests/test_drive_cl_claim_protocol.py",
)


class FreezeError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination-thread-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    if git(repo, "status", "--porcelain"):
        raise FreezeError("refusing to freeze a transfer manifest from a dirty worktree")
    commit = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FreezeError(f"required source file is absent: {relative}")
        hashes[relative] = sha256(path)
    bundle_rows = [f"{path}\0{digest}" for path, digest in sorted(hashes.items())]
    bundle_sha = hashlib.sha256(("\n".join(bundle_rows) + "\n").encode()).hexdigest()
    payload = {
        "schema": "selector_bench.drive_opd_cross_host_migration.v2",
        "created_at": "2026-08-17",
        "source_host": "07Gkhwang",
        "destination_host": "06G",
        "destination_project": "AutoDrive",
        "destination_thread_id": args.destination_thread_id,
        "source_branch": branch,
        "source_commit": commit,
        "source_worktree_clean": True,
        "source_bundle_sha256": bundle_sha,
        "transfer_mode": "source_only_git_branch",
        "weights_remain_on_origin_host": True,
        "optimizer_states_remain_on_origin_host": True,
        "data_gate": {
            "status": "BLOCKED_UNTIL_06G_REPLAY_RECEIPT_PASSES",
            "required_receipt_schema": "selector_bench.drive_cl_cache_provenance.v2",
            "forbid_claim_bearing_training_before_pass": True,
        },
        "initial_arms": ["sequential", "lwf", "fixed_opd", "ema099_opd"],
        "optimizer_contract": {
            "optimizer": "AdamW",
            "joint_official_and_distillation_loss": True,
            "forward_backward_per_step": 1,
            "optimizer_updates_per_step": 1,
            "teacher_stop_gradient": True,
        },
        "preflight_gates": [
            "record_remote_code_commit_and_dataset_roots",
            "replay_official_cache_selection_and_verify_exact_token_hash",
            "record_gpu_uuid_inventory_without_interrupting_other_users",
            "one_batch_official_loss_equivalence",
            "identical_fixed_teacher_student_distillation_near_zero",
            "matched_lwf_and_opd_teacher_query_budget",
            "one_step_checkpoint_resume_with_optimizer_state",
        ],
        "source_files": hashes,
        "remote_result_contract": {
            "sync_back": [
                "environment_inventory.json",
                "cache_provenance.json",
                "experiment_manifest.json",
                "checkpoint_sha256.json",
                "steps.jsonl",
                "epochs.jsonl",
                "evaluation.csv",
                "summary.json",
                "plots",
            ],
            "do_not_sync_back": [
                "model_weights",
                "optimizer_states",
                "dataset",
                "feature_cache",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

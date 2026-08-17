from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from selector_bench.continual.claim_protocol import (
    CELL_SUMMARY_SCHEMA,
    CROSSED_RECEIPT_SCHEMA,
    EVALUATOR_RUN_SCHEMA,
    GLOBAL_RESULT_SCHEMA,
)
from selector_bench.continual.statistics import PDM_DEFAULT_METRICS
from selector_bench.utils.hashing import hash_jsonable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "selector_bench" / "scripts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_sha256(tokens: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(tokens)) + "\n").encode()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_script(
    script: str, *arguments: object, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "selector_bench")
    return subprocess.run(
        [sys.executable, str(SCRIPT_ROOT / script), *(str(value) for value in arguments)],
        cwd=str(cwd or REPOSITORY_ROOT),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


P0_REJECTION_MARKER = "DRIVE_CL_P0_REJECTION="


def audit_rejected_cli(
    attack_id: str,
    variant: str,
    process: subprocess.CompletedProcess[str],
    claim_output: Path,
) -> None:
    """Expose the mutated CLI decision without weakening the assertions below."""
    payload = {
        "attack_id": attack_id,
        "variant": variant,
        "mutated_cli_exit_code": process.returncode,
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
        "claim_output": str(claim_output),
        "claim_output_exists_before_rejection_assertion": claim_output.exists(),
    }
    print(P0_REJECTION_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def git_commit(root: Path, message: str, *paths: Path) -> str:
    subprocess.run(
        ["git", "-C", str(root), "add", *(str(path.relative_to(root)) for path in paths)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", message], check=True
    )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class ClaimFixture:
    def __init__(self, root: Path, seeds: tuple[int, ...], split: str) -> None:
        self.root = root
        self.seeds = seeds
        self.split = split
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "audit@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Audit Test"], check=True
        )
        self.entrypoint = root / "dummy_evaluator.py"
        self.entrypoint.write_text(
            "import argparse, csv, hashlib\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--checkpoint', required=True)\n"
            "p.add_argument('--tokens', required=True)\n"
            "p.add_argument('--output', required=True)\n"
            "a=p.parse_args()\n"
            "tokens=[x for x in open(a.tokens).read().splitlines() if x]\n"
            "h=int(hashlib.sha256(open(a.checkpoint,'rb').read()).hexdigest()[:8],16)\n"
            "metrics=['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort','ego_progress','driving_direction_compliance']\n"
            "with open(a.output,'w',newline='') as f:\n"
            " w=csv.writer(f); w.writerow(['token','valid',*metrics])\n"
            " for i,t in enumerate(tokens):\n"
            "  base=0.1+(h%700000)/1000000+i/1000\n"
            "  w.writerow([t,'true',*[f'{base+j/10000:.6f}' for j in range(7)]])\n"
        )
        self.command_spec = root / "evaluator_command.json"
        write_json(
            self.command_spec,
            {
                "schema": "selector_bench.drive_cl_evaluator_command.v1",
                "argv": [
                    sys.executable,
                    "{evaluator_entrypoint}",
                    "--checkpoint",
                    "{checkpoint}",
                    "--tokens",
                    "{token_file}",
                    "--output",
                    "{output_csv}",
                ],
            },
        )
        self.source_commit = git_commit(
            root, "freeze evaluator", self.entrypoint, self.command_spec
        )
        self.source_tree = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", f"{self.source_commit}^{{tree}}"],
            text=True,
        ).strip()
        self.environment_lock = root / "environment.lock"
        self.environment_lock.write_text("dummy-evaluator==1\n")
        self.method_registry = root / "method_registry.json"
        self.method_registry.write_bytes(
            (
                REPOSITORY_ROOT
                / "selector_bench/configs/drive_cl_method_registry.v1.json"
            ).read_bytes()
        )
        self.metric_registry = root / "metric_registry.json"
        self.metric_registry.write_bytes(
            (
                REPOSITORY_ROOT
                / "selector_bench/configs/drive_cl_metric_registry.v1.json"
            ).read_bytes()
        )

        self.tokens = ("token-a", "token-b")
        self.token_to_log = {
            "token-a": "2021.05.01.00.00.00_veh-01_00000_00001",
            "token-b": "2021.05.02.00.00.00_veh-02_00000_00001",
        }
        self.protocol_path = root / "protocol.json"
        protocol: dict[str, Any] = {
            "schema": "selector_bench.drive_cl_protocol.v1",
            "protocol_id": "toy_cl",
            "created_at_utc": "2026-08-17T00:00:00+00:00",
            "dataset_identity": {
                "dataset": "navsim-fixture",
                "dataset_version": "v1",
                "dataset_root_metadata_sha256": "d" * 64,
            },
            "stages": [
                {
                    "stage_index": 1,
                    "name": "old_domain",
                    "splits": {
                        split: {
                            "tokens": list(self.tokens),
                            "token_to_log": self.token_to_log,
                        }
                    },
                },
                {"stage_index": 2, "name": "safety_patch", "splits": {}},
            ],
        }
        canonical = dict(protocol)
        canonical.pop("created_at_utc")
        protocol["content_sha256"] = hash_jsonable(canonical)
        write_json(self.protocol_path, protocol)
        self.protocol = protocol
        self.token_file = root / "artifacts" / "evaluation" / "tokens.txt"
        self.token_file.parent.mkdir(parents=True)
        self.token_file.write_text("\n".join(sorted(self.tokens)) + "\n")
        self.evaluation_receipt = self.token_file.parent / "receipt.json"
        write_json(
            self.evaluation_receipt,
            {
                "schema": "selector_bench.drive_cl_eval_cell.v1",
                "protocol_manifest": str(self.protocol_path),
                "protocol_manifest_sha256": sha256(self.protocol_path),
                "protocol_content_sha256": protocol["content_sha256"],
                "dataset_identity": protocol["dataset_identity"],
                "domain_registry": str(self.protocol_path),
                "domain_registry_sha256": sha256(self.protocol_path),
                "stage_index": 1,
                "stage_name": "old_domain",
                "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                "split": split,
                "token_count": 2,
                "log_count": 2,
                "session_count": 2,
                "token_file": str(self.token_file),
                "token_file_sha256": sha256(self.token_file),
                "deterministic_seed_env": "PDM_DETERMINISTIC_GLOBAL_SEED",
                "final_test_policy": (
                    "configuration_must_be_frozen_before_use"
                    if split == "test"
                    else "method_and_trigger_selection_allowed"
                ),
            },
        )
        self.config = root / "experiment.json"
        write_json(
            self.config,
            {"protocol_content_sha256": protocol["content_sha256"]},
        )
        self.artifact_root = root / "artifacts"
        self.protocol_inputs_commit = git_commit(
            root,
            "freeze fixture protocol inputs",
            self.protocol_path,
            self.config,
            self.method_registry,
            self.metric_registry,
            self.environment_lock,
        )
        self.methods: dict[str, dict[str, Any]] = {}
        self.run_paths: dict[tuple[str, int], dict[str, Path]] = {}
        for method_index, (method_id, arm) in enumerate(
            (("sequential", "sequential"), ("drive_opd_fixed", "fixed_opd"))
        ):
            runs = []
            for seed in seeds:
                run_root = self.artifact_root / method_id / f"seed-{seed}"
                run_root.mkdir(parents=True)
                endpoint = run_root / "endpoint.ckpt"
                endpoint.write_bytes(f"{method_id}-{seed}-checkpoint".encode())
                run_id = hashlib.sha256(f"{method_id}:{seed}".encode()).hexdigest()[:32]
                source_checkpoint = root / "stage-start" / f"seed-{seed}.ckpt"
                source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                if not source_checkpoint.exists():
                    source_checkpoint.write_bytes(f"source:{seed}".encode())
                source_hash = sha256(source_checkpoint)
                training_protocol = run_root / "protocol.json"
                training_payload = {
                    "schema": "selector_bench.drive_cl_diffusiondrive_run.v3",
                    "run_id": run_id,
                    "method_id": method_id,
                    "arm": arm,
                    "seed": seed,
                    "stage_index": 2,
                    "dataset_identity": protocol["dataset_identity"],
                    "source_code": {
                        "repository": str(root),
                        "commit": self.source_commit,
                        "tree": self.source_tree,
                        "dirty": False,
                    },
                    "protocol_manifest": str(self.protocol_path),
                    "protocol_manifest_sha256": sha256(self.protocol_path),
                    "protocol_content_sha256": protocol["content_sha256"],
                    "experiment_config": str(self.config),
                    "experiment_config_sha256": sha256(self.config),
                    "method_registry": str(self.method_registry),
                    "method_registry_sha256": sha256(self.method_registry),
                    "environment_lock": str(self.environment_lock),
                    "environment_lock_sha256": sha256(self.environment_lock),
                    "source_checkpoint": str(source_checkpoint),
                    "source_checkpoint_sha256": source_hash,
                    "stage_start_state_sha256": source_hash,
                    "start_epoch": 1,
                    "end_epoch": 2,
                    "max_steps": 0,
                }
                rng_record = run_root / "rng_record.json"
                write_json(
                    rng_record,
                    {
                        "schema": "selector_bench.drive_cl_rng_record.v1",
                        "run_id": run_id,
                        "seed": seed,
                        "python_seed": seed,
                        "numpy_seed": seed,
                        "torch_seed": seed,
                    },
                )
                training_payload["rng_record"] = str(rng_record)
                training_payload["rng_record_sha256"] = sha256(rng_record)
                training_payload["target_budget"] = {
                    "epochs": 2,
                    "optimizer_updates": 10,
                    "current_unique_identities": 2,
                    "old_unique_identities": 2,
                    "current_presentations": 20,
                    "old_presentations": 0,
                    "forward_calls": 10,
                    "backward_calls": 10,
                    "student_queries": 20 if arm == "fixed_opd" else 0,
                    "teacher_queries": 20 if arm == "fixed_opd" else 0,
                }
                registration_path = run_root / "run_registration.json"
                registration_fields = (
                    "run_id",
                    "method_id",
                    "arm",
                    "seed",
                    "stage_index",
                    "dataset_identity",
                    "source_code",
                    "protocol_manifest_sha256",
                    "protocol_content_sha256",
                    "experiment_config_sha256",
                    "method_registry_sha256",
                    "environment_lock_sha256",
                    "source_checkpoint_sha256",
                    "stage_start_state_sha256",
                    "start_epoch",
                    "end_epoch",
                    "max_steps",
                    "target_budget",
                )
                write_json(
                    registration_path,
                    {
                        "schema": "selector_bench.drive_cl_run_registration.v1",
                        **{field: training_payload[field] for field in registration_fields},
                    },
                )
                registration_commit = git_commit(
                    root,
                    f"freeze run registration {method_id} seed {seed}",
                    registration_path,
                )
                training_payload["run_registration_repository"] = str(root)
                training_payload["run_registration"] = str(registration_path)
                training_payload["run_registration_sha256"] = sha256(registration_path)
                training_payload["run_registration_commit"] = registration_commit
                write_json(training_protocol, training_payload)
                training_result = run_root / "result.json"
                optimizer_digest = hashlib.sha256(
                    f"optimizer:{method_id}:{seed}".encode()
                ).hexdigest()
                optimizer_receipt = run_root / "optimizer_state_receipt.json"
                write_json(
                    optimizer_receipt,
                    {
                        "schema": "selector_bench.drive_cl_optimizer_state_receipt.v1",
                        "run_id": run_id,
                        "optimizer": "AdamW",
                        "optimizer_state_sha256": optimizer_digest,
                        "endpoint_sha256": sha256(endpoint),
                    },
                )
                write_json(
                    training_result,
                    {
                        "schema": "selector_bench.drive_cl_diffusiondrive_result.v3",
                        "training_protocol": str(training_protocol),
                        "training_protocol_sha256": sha256(training_protocol),
                        **{
                            key: training_payload[key]
                            for key in (
                                "run_id",
                                "method_id",
                                "arm",
                                "seed",
                                "stage_index",
                                "dataset_identity",
                                "source_code",
                                "protocol_manifest_sha256",
                                "protocol_content_sha256",
                                "experiment_config_sha256",
                                "method_registry_sha256",
                                "environment_lock_sha256",
                                "rng_record_sha256",
                                "run_registration_sha256",
                                "run_registration_commit",
                                "source_checkpoint_sha256",
                                "stage_start_state_sha256",
                                "start_epoch",
                                "end_epoch",
                                "max_steps",
                                "target_budget",
                            )
                        },
                        "completed_epoch": 2,
                        "global_step": 10,
                        "steps_this_invocation": 10,
                        "stopped_early": False,
                        "observed_budget": training_payload["target_budget"],
                        "endpoint": str(endpoint),
                        "endpoint_sha256": sha256(endpoint),
                        "optimizer_state_receipt": str(optimizer_receipt),
                        "optimizer_state_receipt_sha256": sha256(optimizer_receipt),
                        "optimizer_state_sha256": optimizer_digest,
                        "parent_checkpoint_sha256": source_hash,
                        "resources": {
                            "wall_seconds": 1.0,
                            "peak_vram_bytes": 0,
                            "persistent_bytes": endpoint.stat().st_size,
                            "transient_bytes": 0,
                            "host": "cpu-fixture",
                            "execution_device": "cpu_fixture",
                            "gpu_uuid": None,
                        },
                    },
                )
                paths = {
                    "root": run_root,
                    "registration": registration_path,
                    "protocol": training_protocol,
                    "result": training_result,
                    "endpoint": endpoint,
                    "csv": run_root / "pdm.csv",
                    "evaluator": run_root / "evaluator.json",
                    "stdout": run_root / "evaluator.stdout",
                    "stderr": run_root / "evaluator.stderr",
                    "summary": run_root / "summary.json",
                }
                self.run_paths[(method_id, seed)] = paths
                runs.append(
                    {
                        "training_seed": seed,
                        "training_protocol": str(training_protocol),
                        "training_result": str(training_result),
                        "cell_summary": str(paths["summary"]),
                        "run_registration": str(registration_path),
                        "run_registration_sha256": sha256(registration_path),
                        "run_registration_commit": registration_commit,
                    }
                )
            self.methods[method_id] = {
                "method_id": method_id,
                "training_arm": arm,
                "runs": runs,
            }
        self.spec_path = root / "comparison_spec.json"
        self.write_spec("screening_only")

    def write_spec(self, analysis_scope: str) -> None:
        write_json(
            self.spec_path,
            {
                "schema": "selector_bench.drive_cl_crossed_comparison_spec.v2",
                "comparison_id": "main",
                "analysis_scope": analysis_scope,
                "metric_family": list(PDM_DEFAULT_METRICS),
                "expected_seed_ids": list(self.seeds),
                "bootstrap_repetitions": 100,
                "bootstrap_seed": 7,
                "artifact_root": str(self.artifact_root),
                "protocol_manifest": str(self.protocol_path),
                "evaluation_cell_receipt": str(self.evaluation_receipt),
                "checkpoint_stage_index": 2,
                "baseline": self.methods["sequential"],
                "candidate": self.methods["drive_opd_fixed"],
            },
        )

    def evaluate_all(
        self,
        *,
        access_receipt: Path | None = None,
        access_commit: str | None = None,
    ) -> None:
        for method_id in ("sequential", "drive_opd_fixed"):
            for seed in self.seeds:
                paths = self.run_paths[(method_id, seed)]
                arguments: list[object] = [
                    "--training-protocol",
                    paths["protocol"],
                    "--training-result",
                    paths["result"],
                    "--evaluation-cell-receipt",
                    self.evaluation_receipt,
                    "--evaluator-repository",
                    self.root,
                    "--evaluator-source-commit",
                    self.source_commit,
                    "--evaluator-entrypoint",
                    self.entrypoint,
                    "--command-spec",
                    self.command_spec,
                    "--environment-lock",
                    self.environment_lock,
                    "--metric-registry",
                    self.metric_registry,
                    "--output-csv",
                    paths["csv"],
                    "--output-receipt",
                    paths["evaluator"],
                    "--stdout-log",
                    paths["stdout"],
                    "--stderr-log",
                    paths["stderr"],
                    "--deterministic-seed",
                    seed,
                ]
                if access_receipt is not None:
                    arguments.extend(
                        [
                            "--access-repository",
                            self.root,
                            "--access-receipt",
                            access_receipt,
                            "--access-commit",
                            access_commit,
                        ]
                    )
                evaluated = run_script("54_run_drive_cl_navsim_evaluator.py", *arguments)
                if evaluated.returncode != 0:
                    raise AssertionError(evaluated.stderr)
                summary = run_script(
                    "45_summarize_drive_cl_pdm.py",
                    "--evaluator-receipt",
                    paths["evaluator"],
                    "--bootstrap-repetitions",
                    20,
                    "--output",
                    paths["summary"],
                )
                if summary.returncode != 0:
                    raise AssertionError(summary.stderr)

    def freeze_confirmatory_family(
        self,
        *,
        claim_relative: str = "claims/main.crossed.json",
        reverse_hypotheses: bool = False,
    ) -> tuple[Path, str, Path, str]:
        if self.split != "test":
            raise AssertionError("confirmatory fixture requires sealed test split")
        self.write_spec("confirmatory")
        audit_source = self.root / "audit_source_receipt.json"
        write_json(
            audit_source,
            {
                "schema": "selector_bench.drive_cl_audit_source.v1",
                "split": "audit",
                "status": "completed",
            },
        )
        pilot_matrix = self.root / "pilot_matrix.json"
        write_json(
            pilot_matrix,
            {
                "schema": "selector_bench.drive_cl_audit_pilot_matrix.v1",
                "split": "audit",
                "metrics": list(PDM_DEFAULT_METRICS),
                "seed_ids": [0, 1],
                "session_ids": ["pilot-a", "pilot-b"],
                "source_receipts": [str(audit_source)],
                "source_receipt_sha256": [sha256(audit_source)],
                "candidate_minus_baseline": {
                    metric: [[0.0, 0.01], [-0.01, 0.0]]
                    for metric in PDM_DEFAULT_METRICS
                },
            },
        )
        seed_design = self.root / "seed_design.json"
        write_json(
            seed_design,
            {
                "schema": "selector_bench.drive_cl_seed_design.v1",
                "status": "passed",
                "design_split": "audit",
                "designed_seed_ids": list(self.seeds),
                "minimum_relevant_effect": 0.02,
                "target_power": 0.8,
                "estimated_power": 0.85,
                "family_alpha": 0.05,
                "family_metric_count": 7,
                "power_simulation_repetitions": 10000,
                "simulation_seed": 0,
                "minimum_seed_count_for_two_sided_family_tail_resolution": 9,
                "calibration_status": "passed",
                "calibration_method": "heldout_null_crossed_resampling_max_statistic_over_seven_metrics",
                "power_method": "audit_only_crossed_resampling_at_minimum_relevant_effect",
                "pilot_matrix": str(pilot_matrix),
                "pilot_matrix_sha256": sha256(pilot_matrix),
                "audit_source_receipt_sha256": [sha256(audit_source)],
                "candidate_seed_audits": [
                    {
                        "seed_count": len(self.seeds),
                        "minimum_metric_power": 0.85,
                        "metric_power": {
                            metric: 0.85 for metric in PDM_DEFAULT_METRICS
                        },
                        "heldout_null_fwer": 0.05,
                        "null_fwer_tolerance": 0.06,
                        "calibration_simulations": 5000,
                        "evaluation_simulations": 5000,
                    }
                ],
            },
        )
        claim_receipt = self.root / claim_relative
        family = self.root / "family.json"
        hypotheses = []
        for metric in PDM_DEFAULT_METRICS:
            hypotheses.append(
                {
                    "id": f"main::{metric}",
                    "comparison_id": "main",
                    "comparison_spec_repository_path": self.spec_path.relative_to(
                        self.root
                    ).as_posix(),
                    "comparison_spec_sha256": sha256(self.spec_path),
                    "comparison_receipt_repository_path": claim_receipt.relative_to(
                        self.root
                    ).as_posix(),
                    "checkpoint_stage_index": 2,
                    "checkpoint_stage_name": "safety_patch",
                    "evaluation_stage_index": 1,
                    "evaluation_stage_name": "old_domain",
                    "split": "test",
                    "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                    "metric": metric,
                    "direction": "two_sided_candidate_minus_baseline",
                    "baseline_method": "sequential",
                    "baseline_training_arm": "sequential",
                    "candidate_method": "drive_opd_fixed",
                    "candidate_training_arm": "fixed_opd",
                    "sorted_token_sha256": token_sha256(self.tokens),
                }
            )
        if reverse_hypotheses:
            hypotheses.reverse()
        write_json(
            family,
            {
                "schema": "selector_bench.drive_cl_global_holm_family.v4",
                "family_id": "p003_primary",
                "alpha": 0.05,
                "dataset_identity": self.protocol["dataset_identity"],
                "method_registry": str(self.method_registry),
                "method_registry_sha256": sha256(self.method_registry),
                "metric_registry": str(self.metric_registry),
                "metric_registry_sha256": sha256(self.metric_registry),
                "domain_registry": str(self.protocol_path),
                "domain_registry_sha256": sha256(self.protocol_path),
                "metric_family": list(PDM_DEFAULT_METRICS),
                "expected_seed_ids": list(self.seeds),
                "seed_design_receipt": str(seed_design),
                "seed_design_receipt_sha256": sha256(seed_design),
                "claim_receipt_root": "claims",
                "expected_comparisons": [
                    {
                        "comparison_id": "main",
                        "comparison_spec_repository_path": self.spec_path.relative_to(
                            self.root
                        ).as_posix(),
                        "comparison_spec_sha256": sha256(self.spec_path),
                        "comparison_receipt_repository_path": claim_receipt.relative_to(
                            self.root
                        ).as_posix(),
                        "checkpoint_stage_index": 2,
                        "evaluation_stage_index": 1,
                        "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                        "split": "test",
                        "baseline_method": "sequential",
                        "baseline_training_arm": "sequential",
                        "candidate_method": "drive_opd_fixed",
                        "candidate_training_arm": "fixed_opd",
                    }
                ],
                "required_cross_cells": [
                    {
                        "checkpoint_stage_index": 2,
                        "evaluation_stage_index": 1,
                        "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                        "split": "test",
                    }
                ],
                "hypotheses": hypotheses,
            },
        )
        freeze_commit = git_commit(
            self.root,
            "freeze family",
            self.spec_path,
            audit_source,
            pilot_matrix,
            seed_design,
            family,
            self.protocol_path,
            self.method_registry,
            self.metric_registry,
        )
        access = self.root / "sealed_access.json"
        write_json(
            access,
            {
                "schema": "selector_bench.drive_cl_sealed_test_access.v1",
                "status": "authorized",
                "authorized_after_family_freeze": True,
                "access_event_id": hashlib.sha256(
                    f"sealed:{freeze_commit}".encode()
                ).hexdigest()[:32],
                "authorized_at_utc": "2026-08-17T00:01:00+00:00",
                "prior_final_test_access_count": 0,
                "evaluation_cell_receipt_sha256": sha256(self.evaluation_receipt),
                "protocol_content_sha256": self.protocol["content_sha256"],
                "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                "family_spec_repository_path": family.relative_to(self.root).as_posix(),
                "family_spec_sha256": sha256(family),
                "family_freeze_commit": freeze_commit,
            },
        )
        access_commit = git_commit(self.root, "authorize sealed test", access)
        return family, freeze_commit, access, access_commit


class DriveCLClaimProtocolTest(unittest.TestCase):
    def _confirmatory_chain(
        self,
        root: Path,
        *,
        claim_relative: str = "claims/main.crossed.json",
        reverse_hypotheses: bool = False,
    ) -> dict[str, Any]:
        fixture = ClaimFixture(root, tuple(range(9)), "test")
        family, freeze_commit, access, access_commit = fixture.freeze_confirmatory_family(
            claim_relative=claim_relative,
            reverse_hypotheses=reverse_hypotheses,
        )
        fixture.evaluate_all(access_receipt=access, access_commit=access_commit)
        claim = root / claim_relative
        compared = run_script(
            "52_compare_drive_cl_crossed_pdm.py",
            "--comparison-spec",
            fixture.spec_path,
            "--family-repository",
            root,
            "--family-spec",
            family,
            "--family-freeze-commit",
            freeze_commit,
            "--output",
            claim,
        )
        self.assertEqual(compared.returncode, 0, compared.stderr)
        evidence_inventory = root / "evidence_inventory.json"
        inventory = run_script(
            "56_build_drive_cl_evidence_inventory.py",
            "--family-spec",
            family,
            "--repository",
            root,
            "--family-freeze-commit",
            freeze_commit,
            "--output",
            evidence_inventory,
        )
        self.assertEqual(inventory.returncode, 0, inventory.stderr)
        evidence_freeze_commit = git_commit(
            root, "freeze evidence inventory", evidence_inventory
        )
        return {
            "fixture": fixture,
            "family": family,
            "freeze_commit": freeze_commit,
            "claim": claim,
            "evidence_inventory": evidence_inventory,
            "evidence_freeze_commit": evidence_freeze_commit,
        }

    def _run_holm(self, chain: dict[str, Any], output: Path) -> subprocess.CompletedProcess[str]:
        return run_script(
            "50_apply_drive_cl_global_holm.py",
            "--family-spec",
            chain["family"],
            "--repository",
            chain["fixture"].root,
            "--freeze-commit",
            chain["freeze_commit"],
            "--evidence-inventory",
            chain["evidence_inventory"],
            "--evidence-freeze-commit",
            chain["evidence_freeze_commit"],
            "--output",
            output,
        )

    def test_example_specs_register_methods_seeds_and_semantic_axes(self) -> None:
        crossed = json.loads(
            (
                REPOSITORY_ROOT
                / "selector_bench/configs/drive_cl_crossed_comparison_spec.example.json"
            ).read_text()
        )
        self.assertEqual(
            crossed["schema"], "selector_bench.drive_cl_crossed_comparison_spec.v2"
        )
        self.assertEqual(crossed["expected_seed_ids"], list(range(9)))
        self.assertEqual(crossed["candidate"]["method_id"], "drive_opd_fixed")
        self.assertNotIn("evaluation_domain", crossed)
        family = json.loads(
            (
                REPOSITORY_ROOT
                / "selector_bench/configs/drive_cl_global_holm_family.example.json"
            ).read_text()
        )
        self.assertEqual(
            family["schema"], "selector_bench.drive_cl_global_holm_family.v4"
        )
        self.assertEqual(
            [item["metric"] for item in family["hypotheses"]],
            list(PDM_DEFAULT_METRICS),
        )
        self.assertTrue(
            all(
                item["candidate_training_arm"] == "fixed_opd"
                and "comparison_spec_sha256" in item
                for item in family["hypotheses"]
            )
        )

    def test_cpu_run_registration_cli_freezes_target_identity_before_training(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "audit@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Audit Test"],
                check=True,
            )
            protocol = root / "protocol.json"
            write_json(
                protocol,
                {
                    "content_sha256": "c" * 64,
                    "dataset_identity": {
                        "dataset": "NAVSIM",
                        "dataset_version": "fixture-v1",
                        "dataset_root_metadata_sha256": "d" * 64,
                    },
                },
            )
            config = root / "config.json"
            write_json(config, {"protocol_content_sha256": "c" * 64})
            registry = root / "methods.json"
            registry.write_bytes(
                (
                    REPOSITORY_ROOT
                    / "selector_bench/configs/drive_cl_method_registry.v1.json"
                ).read_bytes()
            )
            environment = root / "environment.lock"
            environment.write_text("fixture==1\n")
            checkpoint = root / "stage-start.ckpt"
            checkpoint.write_bytes(b"stage-start")
            budget = root / "target_budget.json"
            write_json(
                budget,
                {
                    "epochs": 1,
                    "optimizer_updates": 2,
                    "current_unique_identities": 2,
                    "old_unique_identities": 0,
                    "current_presentations": 2,
                    "old_presentations": 0,
                    "forward_calls": 2,
                    "backward_calls": 2,
                    "student_queries": 0,
                    "teacher_queries": 0,
                },
            )
            git_commit(
                root,
                "freeze registration inputs",
                protocol,
                config,
                registry,
                environment,
                checkpoint,
                budget,
            )
            output = root / "registration.json"
            registered = run_script(
                "58_register_drive_cl_run.py",
                "--source-repository",
                root,
                "--protocol-manifest",
                protocol,
                "--experiment-config",
                config,
                "--method-registry",
                registry,
                "--environment-lock",
                environment,
                "--source-checkpoint",
                checkpoint,
                "--target-budget",
                budget,
                "--arm",
                "sequential",
                "--seed",
                0,
                "--stage-index",
                2,
                "--start-epoch",
                1,
                "--end-epoch",
                1,
                "--output",
                output,
            )
            self.assertEqual(registered.returncode, 0, registered.stderr)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["schema"], "selector_bench.drive_cl_run_registration.v1")
            self.assertEqual(payload["target_budget"]["optimizer_updates"], 2)
            self.assertFalse(payload["source_code"]["dirty"])

    def test_crossed_screening_binds_real_evaluator_seed_and_off_diagonal_cell(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            output = fixture.root / "screening.json"
            process = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                output,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["schema"], CROSSED_RECEIPT_SCHEMA)
            self.assertEqual(receipt["checkpoint_stage_index"], 2)
            self.assertEqual(receipt["evaluation_stage_index"], 1)
            self.assertEqual(
                receipt["evaluation_domain"],
                "navsim-fixture/v1::toy_cl::stage_1::old_domain",
            )
            self.assertEqual(receipt["baseline"]["training_arm"], "sequential")
            self.assertEqual(receipt["candidate"]["training_arm"], "fixed_opd")
            self.assertEqual(
                len({run["run_id"] for run in receipt["candidate"]["runs"]}), 2
            )
            evaluator = json.loads(
                fixture.run_paths[("drive_opd_fixed", 0)]["evaluator"].read_text()
            )
            self.assertEqual(evaluator["schema"], EVALUATOR_RUN_SCHEMA)
            self.assertIn(str(fixture.run_paths[("drive_opd_fixed", 0)]["endpoint"]), evaluator["executed_argv"])

    def test_p0_01_seed_reuse_is_rejected_before_claim(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            spec = json.loads(fixture.spec_path.read_text())
            spec["baseline"]["runs"][0]["training_result"] = spec["baseline"]["runs"][1][
                "training_result"
            ]
            spec["baseline"]["runs"][0]["cell_summary"] = spec["baseline"]["runs"][1][
                "cell_summary"
            ]
            write_json(fixture.spec_path, spec)
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                fixture.root / "failed.json",
            )
            audit_rejected_cli("P0-01", "seed-reuse", failed, fixture.root / "failed.json")
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((fixture.root / "failed.json").exists())
            self.assertIn("result training protocol path mismatch", failed.stderr)

    def test_p0_03_arm_mismatch_is_rejected_before_claim(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            relabel_spec = json.loads(fixture.spec_path.read_text())
            relabel_spec["candidate"]["method_id"] = "sequential"
            write_json(fixture.spec_path, relabel_spec)
            relabel = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                fixture.root / "failed.json",
            )
            audit_rejected_cli("P0-03", "arm-mismatch", relabel, fixture.root / "failed.json")
            self.assertNotEqual(relabel.returncode, 0)
            self.assertFalse((fixture.root / "failed.json").exists())
            self.assertIn("registered to arm sequential", relabel.stderr)

    def test_p0_04_unknown_domain_is_rejected_before_claim(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            evaluation_payload = json.loads(fixture.evaluation_receipt.read_text())
            evaluation_payload["evaluation_domain"] = "fabricated_unregistered_domain"
            write_json(fixture.evaluation_receipt, evaluation_payload)
            unknown = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                fixture.root / "failed.json",
            )
            audit_rejected_cli("P0-04", "unknown-domain", unknown, fixture.root / "failed.json")
            self.assertNotEqual(unknown.returncode, 0)
            self.assertFalse((fixture.root / "failed.json").exists())
            self.assertIn("evaluation receipt domain mismatch", unknown.stderr)

    def test_p0_02_csv_swap_is_rejected_before_claim(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            baseline_csv = fixture.run_paths[("sequential", 0)]["csv"]
            candidate_paths = fixture.run_paths[("drive_opd_fixed", 0)]
            evaluator = json.loads(candidate_paths["evaluator"].read_text())
            old_csv = evaluator["csv"]
            evaluator["csv"] = str(baseline_csv)
            evaluator["csv_sha256"] = sha256(baseline_csv)
            evaluator["executed_argv"] = [
                str(baseline_csv) if value == old_csv else value
                for value in evaluator["executed_argv"]
            ]
            write_json(candidate_paths["evaluator"], evaluator)
            summary = json.loads(candidate_paths["summary"].read_text())
            summary["evaluator_run_receipt_sha256"] = sha256(candidate_paths["evaluator"])
            summary["csv"] = str(baseline_csv)
            summary["csv_sha256"] = sha256(baseline_csv)
            write_json(candidate_paths["summary"], summary)
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                fixture.root / "failed.json",
            )
            audit_rejected_cli("P0-02", "csv-swap", failed, fixture.root / "failed.json")
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((fixture.root / "failed.json").exists())
            self.assertRegex(failed.stderr, "reuse csv_(path|sha256)")

    def test_rejects_stage_start_checkpoint_content_drift(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            source = fixture.root / "stage-start" / "seed-0.ckpt"
            source.write_bytes(b"tampered-after-training")
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                fixture.root / "failed.json",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("training source checkpoint SHA256 mismatch", failed.stderr)

    def test_claim_clis_reject_metric_override(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dummy = root / "dummy"
            dummy.write_text("x")
            for script, arguments in (
                (
                    "45_summarize_drive_cl_pdm.py",
                    ["--evaluator-receipt", dummy, "--output", root / "summary.json"],
                ),
                (
                    "46_compare_drive_cl_pdm.py",
                    [
                        "--baseline-csv",
                        dummy,
                        "--receipt",
                        dummy,
                        "--candidate",
                        f"method={dummy}",
                        "--output",
                        root / "comparison.json",
                    ],
                ),
            ):
                process = run_script(script, *arguments, "--metric", "score")
                self.assertEqual(process.returncode, 2)
                self.assertIn("unrecognized arguments: --metric score", process.stderr)

    def test_p0_05_suffix_hide_is_rejected_before_claim_decision(self) -> None:
        with TemporaryDirectory() as directory:
            chain = self._confirmatory_chain(Path(directory))
            renamed = chain["claim"].with_name("renamed_without_registered_suffix.bin")
            chain["claim"].rename(renamed)
            failed = run_script(
                "56_build_drive_cl_evidence_inventory.py",
                "--family-spec",
                chain["family"],
                "--repository",
                chain["fixture"].root,
                "--family-freeze-commit",
                chain["freeze_commit"],
                "--output",
                chain["fixture"].root / "attack_inventory.json",
            )
            audit_rejected_cli(
                "P0-05",
                "suffix-hide",
                failed,
                chain["fixture"].root / "attack_inventory.json",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((chain["fixture"].root / "attack_inventory.json").exists())
            self.assertIn("inventory differs from expected comparisons", failed.stderr)

    def test_p0_06_duplicate_receipt_content_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            chain = self._confirmatory_chain(Path(directory))
            duplicate = chain["claim"].parent / "duplicate-arbitrary-name"
            duplicate.write_bytes(chain["claim"].read_bytes())
            failed = run_script(
                "56_build_drive_cl_evidence_inventory.py",
                "--family-spec",
                chain["family"],
                "--repository",
                chain["fixture"].root,
                "--family-freeze-commit",
                chain["freeze_commit"],
                "--output",
                chain["fixture"].root / "attack_inventory.json",
            )
            audit_rejected_cli(
                "P0-06",
                "duplicate-receipt",
                failed,
                chain["fixture"].root / "attack_inventory.json",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((chain["fixture"].root / "attack_inventory.json").exists())
            self.assertIn("byte-identical receipts", failed.stderr)

    def test_p0_07_duplicate_raw_evidence_across_receipts_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            chain = self._confirmatory_chain(Path(directory))
            duplicate = chain["claim"].parent / "different-receipt.json"
            payload = json.loads(chain["claim"].read_text())
            payload["comparison_id"] = "renamed-claim-with-same-raw-evidence"
            write_json(duplicate, payload)
            failed = run_script(
                "56_build_drive_cl_evidence_inventory.py",
                "--family-spec",
                chain["family"],
                "--repository",
                chain["fixture"].root,
                "--family-freeze-commit",
                chain["freeze_commit"],
                "--output",
                chain["fixture"].root / "attack_inventory.json",
            )
            audit_rejected_cli(
                "P0-07",
                "duplicate-raw-evidence",
                failed,
                chain["fixture"].root / "attack_inventory.json",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((chain["fixture"].root / "attack_inventory.json").exists())
            self.assertIn("reuses the same raw evidence bundle", failed.stderr)

    def test_p0_08_missing_cross_cell_is_rejected_at_family_gate(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
            family, _, access, access_commit = fixture.freeze_confirmatory_family()
            fixture.evaluate_all(access_receipt=access, access_commit=access_commit)
            payload = json.loads(family.read_text())
            payload["required_cross_cells"].append(
                {
                    "checkpoint_stage_index": 2,
                    "evaluation_stage_index": 2,
                    "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_2::safety_patch",
                    "split": "test",
                }
            )
            write_json(family, payload)
            mutant_commit = git_commit(fixture.root, "freeze missing-cell attack", family)
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--family-repository",
                fixture.root,
                "--family-spec",
                family,
                "--family-freeze-commit",
                mutant_commit,
                "--output",
                fixture.root / "claims" / "main.crossed.json",
            )
            audit_rejected_cli(
                "P0-08",
                "missing-cross-cell",
                failed,
                fixture.root / "claims" / "main.crossed.json",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((fixture.root / "claims" / "main.crossed.json").exists())
            self.assertRegex(failed.stderr, "unknown evaluation split|missing a required")

    def test_p0_09_row_integrity_mutations_are_rejected(self) -> None:
        for mutation in ("duplicate", "extra", "missing", "invalid"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), (0, 1), "audit")
                fixture.evaluate_all()
                paths = fixture.run_paths[("drive_opd_fixed", 0)]
                lines = paths["csv"].read_text().splitlines()
                if mutation == "duplicate":
                    lines.append(lines[1])
                elif mutation == "extra":
                    lines.append(lines[1].replace("token-a", "token-extra", 1))
                elif mutation == "missing":
                    lines.pop()
                else:
                    fields = lines[1].split(",")
                    fields[1] = "false"
                    lines[1] = ",".join(fields)
                paths["csv"].write_text("\n".join(lines) + "\n")
                evaluator = json.loads(paths["evaluator"].read_text())
                evaluator["csv_sha256"] = sha256(paths["csv"])
                evaluator["csv_row_count"] = len(lines) - 1
                write_json(paths["evaluator"], evaluator)
                summary = json.loads(paths["summary"].read_text())
                summary["evaluator_run_receipt_sha256"] = sha256(paths["evaluator"])
                summary["csv_sha256"] = sha256(paths["csv"])
                summary["csv_row_count"] = len(lines) - 1
                write_json(paths["summary"], summary)
                failed = run_script(
                    "52_compare_drive_cl_crossed_pdm.py",
                    "--comparison-spec",
                    fixture.spec_path,
                    "--output",
                    fixture.root / "failed.json",
                )
                audit_rejected_cli(
                    "P0-09", mutation, failed, fixture.root / "failed.json"
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse((fixture.root / "failed.json").exists())

    def test_p0_10_mutable_config_protocol_and_registries_are_rejected(self) -> None:
        for artifact_name in (
            "config",
            "protocol",
            "method_registry",
            "metric_registry",
            "run_registration",
        ):
            with self.subTest(artifact=artifact_name), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), (0, 1), "audit")
                fixture.evaluate_all()
                path = {
                    "config": fixture.config,
                    "protocol": fixture.protocol_path,
                    "method_registry": fixture.method_registry,
                    "metric_registry": fixture.metric_registry,
                    "run_registration": fixture.run_paths[
                        ("drive_opd_fixed", 0)
                    ]["registration"],
                }[artifact_name]
                path.write_bytes(path.read_bytes() + b" \n")
                failed = run_script(
                    "52_compare_drive_cl_crossed_pdm.py",
                    "--comparison-spec",
                    fixture.spec_path,
                    "--output",
                    fixture.root / "failed.json",
                )
                audit_rejected_cli(
                    "P0-10", artifact_name, failed, fixture.root / "failed.json"
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse((fixture.root / "failed.json").exists())

    def test_p0_11_incomplete_run_budget_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            result_path = fixture.run_paths[("drive_opd_fixed", 0)]["result"]
            result = json.loads(result_path.read_text())
            result["observed_budget"]["optimizer_updates"] -= 1
            write_json(result_path, result)
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                fixture.root / "failed.json",
            )
            audit_rejected_cli("P0-11", "incomplete-run", failed, fixture.root / "failed.json")
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((fixture.root / "failed.json").exists())
            self.assertIn("completed run budget mismatch", failed.stderr)

    def test_p0_12_hidden_extra_claim_object_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            chain = self._confirmatory_chain(Path(directory))
            extra = chain["claim"].parent / "nested" / "unregistered-object.data"
            payload = json.loads(chain["claim"].read_text())
            copied_csv = chain["fixture"].root / "copied-unique.csv"
            source_csv = Path(payload["candidate"]["runs"][0]["csv_path"])
            copied_csv.write_bytes(source_csv.read_bytes() + b"\n")
            payload["candidate"]["runs"][0]["csv_path"] = str(copied_csv)
            payload["candidate"]["runs"][0]["csv_sha256"] = sha256(copied_csv)
            write_json(extra, payload)
            failed = run_script(
                "56_build_drive_cl_evidence_inventory.py",
                "--family-spec",
                chain["family"],
                "--repository",
                chain["fixture"].root,
                "--family-freeze-commit",
                chain["freeze_commit"],
                "--output",
                chain["fixture"].root / "attack_inventory.json",
            )
            audit_rejected_cli(
                "P0-12",
                "hidden-extra-claim",
                failed,
                chain["fixture"].root / "attack_inventory.json",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((chain["fixture"].root / "attack_inventory.json").exists())
            self.assertIn("inventory differs from expected comparisons", failed.stderr)

    def test_positive_family_is_filename_and_hypothesis_order_invariant(self) -> None:
        with TemporaryDirectory() as first_directory, TemporaryDirectory() as second_directory:
            first = self._confirmatory_chain(
                Path(first_directory), claim_relative="claims/a/receipt.weird"
            )
            second = self._confirmatory_chain(
                Path(second_directory),
                claim_relative="claims/z/another-name.bin",
                reverse_hypotheses=True,
            )
            first_output = Path(first_directory) / "holm.json"
            second_output = Path(second_directory) / "holm.json"
            first_holm = self._run_holm(first, first_output)
            second_holm = self._run_holm(second, second_output)
            self.assertEqual(first_holm.returncode, 0, first_holm.stderr)
            self.assertEqual(second_holm.returncode, 0, second_holm.stderr)
            first_result = json.loads(first_output.read_text())
            second_result = json.loads(second_output.read_text())
            self.assertEqual(first_result["schema"], GLOBAL_RESULT_SCHEMA)
            self.assertEqual(first_result["hypothesis_count"], 7)
            self.assertEqual(first_result["results"], second_result["results"])


if __name__ == "__main__":
    unittest.main()

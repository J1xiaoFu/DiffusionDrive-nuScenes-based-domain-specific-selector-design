from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from selector_bench.continual.claim_protocol import (
    CELL_SUMMARY_SCHEMA,
    CROSSED_RECEIPT_SCHEMA,
    EVALUATOR_RUN_SCHEMA,
    GLOBAL_RESULT_SCHEMA,
    load_training_run_contract,
    require_strict_final_test_chronology,
    validate_family_registries_and_cells,
)
from selector_bench.continual.statistics import PDM_DEFAULT_METRICS, StatisticsError
from selector_bench.continual.run_budget import TRANSIENT_STORAGE_SEMANTICS
from selector_bench.utils.hashing import hash_jsonable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "selector_bench" / "scripts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_sha256(tokens: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(tokens)) + "\n").encode()).hexdigest()


def write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n"
    )


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


def next_fixture_commit_epoch(root: Path) -> int:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--count", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    count = int(process.stdout.strip()) if process.returncode == 0 else 0
    return 1_700_000_000 + count * 10


def git_commit(
    root: Path,
    message: str,
    *paths: Path,
    commit_epoch: int | None = None,
) -> str:
    subprocess.run(
        ["git", "-C", str(root), "add", *(str(path.relative_to(root)) for path in paths)],
        check=True,
    )
    epoch = next_fixture_commit_epoch(root) if commit_epoch is None else commit_epoch
    environment = os.environ.copy()
    environment["GIT_AUTHOR_DATE"] = f"{epoch} +0000"
    environment["GIT_COMMITTER_DATE"] = f"{epoch} +0000"
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", message],
        check=True,
        env=environment,
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
            (
                ("sequential", "sequential"),
                ("drive_opd_fixed", "fixed_opd"),
                ("lwf", "lwf"),
            )
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
                    "transient_storage_semantics": TRANSIENT_STORAGE_SEMANTICS,
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
                    "student_queries": 20 if arm in {"fixed_opd", "lwf"} else 0,
                    "teacher_queries": 20 if arm in {"fixed_opd", "lwf"} else 0,
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
                            "transient_bytes": 137,
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
        self.lwf_spec_path = root / "comparison_spec_lwf.json"
        self.write_spec("screening_only")

    def write_spec(self, analysis_scope: str) -> None:
        for path, comparison_id, candidate_id in (
            (self.spec_path, "main", "drive_opd_fixed"),
            (self.lwf_spec_path, "lwf_vs_sequential", "lwf"),
        ):
            write_json(
                path,
                {
                "schema": "selector_bench.drive_cl_crossed_comparison_spec.v2",
                "comparison_id": comparison_id,
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
                "candidate": self.methods[candidate_id],
                },
            )

    def evaluate_all(
        self,
        *,
        access_receipt: Path | None = None,
        access_commit: str | None = None,
    ) -> None:
        for method_id in ("sequential", "drive_opd_fixed", "lwf"):
            for seed in self.seeds:
                evaluated = self.evaluate_one(
                    method_id,
                    seed,
                    access_receipt=access_receipt,
                    access_commit=access_commit,
                )
                if evaluated.returncode != 0:
                    raise AssertionError(evaluated.stderr)
                paths = self.run_paths[(method_id, seed)]
                summary = run_script(
                    "45_summarize_drive_cl_pdm.py",
                    "--evaluator-receipt",
                    paths["evaluator"],
                    "--bootstrap-repetitions",
                    20,
                    *(
                        (
                            "--family-repository",
                            self.root,
                            "--family-spec",
                            self.root / "family.json",
                            "--family-freeze-commit",
                            json.loads(paths["evaluator"].read_text())["sealed_test_access"][
                                "family_freeze_commit"
                            ],
                        )
                        if access_receipt is not None
                        else ()
                    ),
                    "--output",
                    paths["summary"],
                )
                if summary.returncode != 0:
                    raise AssertionError(summary.stderr)

    def evaluate_one(
        self,
        method_id: str,
        seed: int,
        *,
        access_receipt: Path | None = None,
        access_commit: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
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
        return run_script("54_run_drive_cl_navsim_evaluator.py", *arguments)

    def freeze_confirmatory_family(
        self,
        *,
        claim_relative: str = "claims/main.crossed.json",
        second_claim_relative: str | None = None,
        reverse_hypotheses: bool = False,
        access_variant: str = "valid",
        seed_design_variant: str = "valid",
    ) -> tuple[Path, str, Path, str]:
        if self.split != "test":
            raise AssertionError("confirmatory fixture requires sealed test split")
        self.write_spec("confirmatory")
        if second_claim_relative is None:
            first_path = Path(claim_relative)
            second_claim_relative = str(
                first_path.parent / ("lwf-" + first_path.name)
            )
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
                "seed_ids": [0, 1, 2, 3],
                "session_ids": [f"pilot-{index}" for index in range(8)],
                "source_receipts": [str(audit_source)],
                "source_receipt_sha256": [sha256(audit_source)],
                "candidate_minus_baseline": {
                    metric: [
                        [
                            (seed_index - 1.5) * 0.01
                            + (session_index - 3.5) * 0.002
                            for session_index in range(8)
                        ]
                        for seed_index in range(4)
                    ]
                    for metric in PDM_DEFAULT_METRICS
                },
            },
        )
        seed_design = self.root / "seed_design.json"
        program_root = self.root / "seed_design_program"
        generator_module = program_root / "seed_design.py"
        generator_entrypoint = program_root / "55_design_drive_cl_confirmatory_seeds.py"
        generator_module.parent.mkdir(parents=True, exist_ok=True)
        generator_module.write_bytes(
            (
                REPOSITORY_ROOT
                / "selector_bench/selector_bench/continual/seed_design.py"
            ).read_bytes()
        )
        generator_entrypoint.write_bytes(
            (SCRIPT_ROOT / "55_design_drive_cl_confirmatory_seeds.py").read_bytes()
        )
        designed = run_script(
            "55_design_drive_cl_confirmatory_seeds.py",
            "--pilot-matrix",
            pilot_matrix,
            "--candidate-seed-ids",
            ",".join(str(seed) for seed in self.seeds),
            "--minimum-relevant-effect",
            0.1,
            "--target-power",
            0.8,
            "--family-alpha",
            0.05,
            "--simulation-repetitions",
            10000,
            "--simulation-seed",
            0,
            "--repository",
            self.root,
            "--generator-module-source",
            generator_module,
            "--generator-entrypoint-source",
            generator_entrypoint,
            "--output",
            seed_design,
        )
        if designed.returncode != 0:
            raise AssertionError(designed.stderr or designed.stdout)
        seed_payload = json.loads(seed_design.read_text())
        if seed_design_variant == "handwritten_output":
            seed_payload["estimated_power"] = 0.812345
        elif seed_design_variant == "altered_seed_or_repetitions":
            seed_payload["normalized_invocation"]["simulation_seed"] = 17
        elif seed_design_variant == "altered_effect_or_pilot":
            seed_payload["normalized_invocation"]["minimum_relevant_effect"] = 0.2
        elif seed_design_variant == "altered_program_identity":
            seed_payload["generator_module_sha256"] = "0" * 64
        elif seed_design_variant == "altered_replayed_statistic":
            seed_payload["candidate_seed_audits"][0]["critical_value"] += 0.5
        elif seed_design_variant.startswith("nonfinite_"):
            _, location, value_name = seed_design_variant.split("_", 2)
            value = {
                "nan": float("nan"),
                "posinf": float("inf"),
                "neginf": float("-inf"),
            }[value_name]
            if location == "minimum-effect":
                seed_payload["minimum_relevant_effect"] = value
            elif location == "target-power":
                seed_payload["target_power"] = value
            elif location == "critical-value":
                seed_payload["candidate_seed_audits"][0]["critical_value"] = value
            elif location == "metric-power":
                seed_payload["candidate_seed_audits"][0]["metric_power"]["score"] = value
            elif location == "heldout-fwer":
                seed_payload["candidate_seed_audits"][0]["heldout_null_fwer"] = value
            else:
                raise AssertionError(f"unknown non-finite location: {location}")
        elif seed_design_variant == "bool_designed_seed":
            seed_payload["designed_seed_ids"][0] = False
        elif seed_design_variant == "bool_candidate_seed":
            seed_payload["candidate_seed_ids"][0] = False
            seed_payload["normalized_invocation"]["candidate_seed_ids"][0] = False
        elif seed_design_variant == "bool_family_metric_count":
            seed_payload["family_metric_count"] = True
        elif seed_design_variant == "bool_simulation_seed":
            seed_payload["simulation_seed"] = False
            seed_payload["normalized_invocation"]["simulation_seed"] = False
        elif seed_design_variant.startswith("family-alpha_"):
            pass
        elif seed_design_variant != "valid":
            raise AssertionError(f"unknown seed-design fixture variant: {seed_design_variant}")
        if seed_design_variant != "valid":
            write_json(seed_design, seed_payload, allow_nan=True)
        claim_receipt = self.root / claim_relative
        second_claim_receipt = self.root / second_claim_relative
        family = self.root / "family.json"
        access_ledger = self.root / "sealed_test_access_ledger.json"
        freeze_events = (
            [{"unregistered_prior_access": True}]
            if access_variant == "nonempty_freeze_ledger"
            else []
        )
        write_json(
            access_ledger,
            {
                "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                "family_id": "p003_primary",
                "events": freeze_events,
            },
        )
        comparisons = (
            (
                "main",
                self.spec_path,
                claim_receipt,
                "drive_opd_fixed",
                "fixed_opd",
            ),
            (
                "lwf_vs_sequential",
                self.lwf_spec_path,
                second_claim_receipt,
                "lwf",
                "lwf",
            ),
        )
        hypotheses = []
        expected_comparisons = []
        for comparison_id, spec_path, receipt_path, candidate_method, candidate_arm in comparisons:
            expected_comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "comparison_spec_repository_path": spec_path.relative_to(
                        self.root
                    ).as_posix(),
                    "comparison_spec_sha256": sha256(spec_path),
                    "comparison_receipt_repository_path": receipt_path.relative_to(
                        self.root
                    ).as_posix(),
                    "checkpoint_stage_index": 2,
                    "evaluation_stage_index": 1,
                    "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                    "split": "test",
                    "baseline_method": "sequential",
                    "baseline_training_arm": "sequential",
                    "candidate_method": candidate_method,
                    "candidate_training_arm": candidate_arm,
                }
            )
            for metric in PDM_DEFAULT_METRICS:
                hypotheses.append(
                    {
                        "id": f"{comparison_id}::{metric}",
                        "comparison_id": comparison_id,
                        "comparison_spec_repository_path": spec_path.relative_to(
                            self.root
                        ).as_posix(),
                        "comparison_spec_sha256": sha256(spec_path),
                        "comparison_receipt_repository_path": receipt_path.relative_to(
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
                        "candidate_method": candidate_method,
                        "candidate_training_arm": candidate_arm,
                        "sorted_token_sha256": token_sha256(self.tokens),
                    }
                )
        if reverse_hypotheses:
            hypotheses.reverse()
        family_payload = {
                "schema": "selector_bench.drive_cl_global_holm_family.v5",
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
                "sealed_test_access_ledger_repository_path": access_ledger.relative_to(
                    self.root
                ).as_posix(),
                "sealed_test_access_ledger_empty_sha256": sha256(access_ledger),
                "claim_receipt_root": "claims",
                "expected_comparisons": expected_comparisons,
                "required_cross_cells": [
                    {
                        "checkpoint_stage_index": 2,
                        "evaluation_stage_index": 1,
                        "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                        "split": "test",
                    }
                ],
                "shared_confirmatory_cells": [
                    {
                        "checkpoint_stage_index": 2,
                        "evaluation_stage_index": 1,
                        "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                        "split": "test",
                        "common_baseline_method": "sequential",
                        "common_baseline_training_arm": "sequential",
                        "minimum_distinct_candidate_count": 2,
                        "candidates": [
                            {"method": "drive_opd_fixed", "training_arm": "fixed_opd"},
                            {"method": "lwf", "training_arm": "lwf"},
                        ],
                    }
                ],
                "hypotheses": hypotheses,
            }
        if seed_design_variant.startswith("family-alpha_"):
            family_payload["alpha"] = {
                "nan": float("nan"),
                "posinf": float("inf"),
                "neginf": float("-inf"),
                "bool": True,
            }[seed_design_variant.removeprefix("family-alpha_")]
        write_json(family, family_payload, allow_nan=True)
        freeze_commit = git_commit(
            self.root,
            "freeze family",
            self.spec_path,
            self.lwf_spec_path,
            audit_source,
            pilot_matrix,
            seed_design,
            generator_module,
            generator_entrypoint,
            access_ledger,
            family,
            self.protocol_path,
            self.method_registry,
            self.metric_registry,
        )
        access = self.root / "sealed_access.json"
        family_epoch = int(
            subprocess.check_output(
                ["git", "-C", str(self.root), "show", "-s", "--format=%ct", freeze_commit],
                text=True,
            ).strip()
        )
        authorized_at = datetime.fromtimestamp(
            family_epoch + 1, tz=timezone.utc
        ).isoformat()
        if access_variant == "authorization_year_1900":
            authorized_at = "1900-01-01T00:00:00+00:00"
        elif access_variant == "future_authorization":
            authorized_at = "2999-01-01T00:00:00+00:00"
        elif access_variant == "family_equals_authorization":
            authorized_at = datetime.fromtimestamp(
                family_epoch, tz=timezone.utc
            ).isoformat()
        elif access_variant == "authorization_equals_access":
            authorized_at = datetime.fromtimestamp(
                next_fixture_commit_epoch(self.root), tz=timezone.utc
            ).isoformat()
        event_id = hashlib.sha256(f"sealed:{freeze_commit}".encode()).hexdigest()[:32]
        event = {
            "access_event_id": event_id,
            "authorized_at_utc": authorized_at,
            "family_id": "p003_primary",
            "family_spec_sha256": sha256(family),
            "evaluation_cell_receipt_sha256": sha256(self.evaluation_receipt),
            "protocol_content_sha256": self.protocol["content_sha256"],
            "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
        }
        if access_variant == "intermediate_event_rewrite":
            prior_event = dict(event)
            prior_event["access_event_id"] = "f" * 32
            write_json(
                access_ledger,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_primary",
                    "events": [prior_event],
                },
            )
            git_commit(self.root, "record unauthorized prior event", access_ledger)
            write_json(
                access_ledger,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_primary",
                    "events": [],
                },
            )
            git_commit(self.root, "delete unauthorized prior event", access_ledger)
        if access_variant == "future_intermediate_then_earlier_access":
            intermediate = self.root / "chronology_intermediate.txt"
            intermediate.write_text("future intermediate\n")
            git_commit(
                self.root,
                "future-dated intermediate",
                intermediate,
                commit_epoch=family_epoch + 100,
            )
        final_events = [event, dict(event)] if access_variant == "duplicate_event" else [event]
        write_json(
            access_ledger,
            {
                "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                "family_id": "p003_primary",
                "events": final_events,
            },
        )
        receipt_ledger = access_ledger
        extra_commit_paths: list[Path] = []
        if access_variant == "wrong_ledger_path_or_family":
            receipt_ledger = self.root / "alternate_access_ledger.json"
            receipt_ledger.write_bytes(access_ledger.read_bytes())
            extra_commit_paths.append(receipt_ledger)
        write_json(
            access,
            {
                "schema": "selector_bench.drive_cl_sealed_test_access.v2",
                "status": "authorized",
                "family_id": "p003_primary",
                "family_spec_repository_path": family.relative_to(self.root).as_posix(),
                "family_spec_sha256": sha256(family),
                "family_freeze_commit": freeze_commit,
                "access_ledger_repository_path": receipt_ledger.relative_to(
                    self.root
                ).as_posix(),
                "access_ledger_sha256": sha256(receipt_ledger),
                "access_event_id": event_id,
                "access_event_index": 0,
                "authorized_at_utc": authorized_at,
                "evaluation_cell_receipt_sha256": sha256(self.evaluation_receipt),
                "protocol_content_sha256": self.protocol["content_sha256"],
                "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
            },
        )
        if access_variant == "asserted_count_substitution":
            access_payload = json.loads(access.read_text())
            access_payload["prior_final_test_access_count"] = 0
            write_json(access, access_payload)
        access_epoch = None
        if access_variant == "future_intermediate_then_earlier_access":
            access_epoch = family_epoch + 20
        elif access_variant == "reversed_endpoint_commit_times":
            access_epoch = family_epoch - 10
        access_commit = git_commit(
            self.root,
            "authorize sealed test",
            access_ledger,
            access,
            *extra_commit_paths,
            commit_epoch=access_epoch,
        )
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
        second_claim = claim.parent / ("lwf-" + claim.name)
        for comparison_spec, output in (
            (fixture.spec_path, claim),
            (fixture.lwf_spec_path, second_claim),
        ):
            compared = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                comparison_spec,
                "--family-repository",
                root,
                "--family-spec",
                family,
                "--family-freeze-commit",
                freeze_commit,
                "--output",
                output,
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
            "second_claim": second_claim,
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
            family["schema"], "selector_bench.drive_cl_global_holm_family.v5"
        )
        self.assertEqual(
            [item["metric"] for item in family["hypotheses"]],
            list(PDM_DEFAULT_METRICS) * 2,
        )
        self.assertEqual(
            family["shared_confirmatory_cells"][0]["candidates"],
            [
                {"method": "drive_opd_fixed", "training_arm": "fixed_opd"},
                {"method": "lwf", "training_arm": "lwf"},
            ],
        )

    def test_common_evaluation_cell_supports_multiple_registered_methods(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
            family_path, freeze_commit, _, _ = fixture.freeze_confirmatory_family()
            family = json.loads(family_path.read_text())
            validated = validate_family_registries_and_cells(
                family,
                family_path=family_path,
                repository=fixture.root,
                freeze_commit=freeze_commit,
            )
            self.assertEqual(len(validated["expected_comparisons"]), 2)
            self.assertEqual(len(validated["required_cross_cells"]), 1)
            self.assertEqual(
                len(next(iter(validated["shared_confirmatory_cells"].values()))["candidates"]),
                2,
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

    def test_claim_eligible_run_requires_measured_nonzero_transient_storage(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            paths = fixture.run_paths[("drive_opd_fixed", 0)]
            result = json.loads(paths["result"].read_text())
            result["resources"]["transient_bytes"] = 0
            write_json(paths["result"], result)
            with self.assertRaisesRegex(
                StatisticsError, "lacks measured nonzero transient storage"
            ):
                load_training_run_contract(paths["protocol"], paths["result"])

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

    def test_p0_13_noncanonical_final_test_chronology_is_rejected(self) -> None:
        variants = (
            "authorization_year_1900",
            "future_authorization",
            "nonempty_freeze_ledger",
            "duplicate_event",
            "wrong_ledger_path_or_family",
            "asserted_count_substitution",
            "intermediate_event_rewrite",
        )
        for variant in variants:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
                _, _, access, access_commit = fixture.freeze_confirmatory_family(
                    access_variant=variant
                )
                claim_output = fixture.run_paths[("sequential", 0)]["evaluator"]
                failed = fixture.evaluate_one(
                    "sequential",
                    0,
                    access_receipt=access,
                    access_commit=access_commit,
                )
                audit_rejected_cli("P0-13", variant, failed, claim_output)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(claim_output.exists())

    def test_p0_14_loader_delivery_budget_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), (0, 1), "audit")
            fixture.evaluate_all()
            result_path = fixture.run_paths[("drive_opd_fixed", 0)]["result"]
            result = json.loads(result_path.read_text())
            result["observed_budget"]["current_unique_identities"] -= 1
            write_json(result_path, result)
            claim_output = fixture.root / "failed.json"
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--output",
                claim_output,
            )
            audit_rejected_cli("P0-14", "loader-delivery-mismatch", failed, claim_output)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(claim_output.exists())
            self.assertIn("completed run budget mismatch", failed.stderr)

    def test_p0_15_seed_design_must_be_exact_executable_replay(self) -> None:
        variants = (
            "handwritten_output",
            "altered_seed_or_repetitions",
            "altered_effect_or_pilot",
            "altered_program_identity",
            "altered_replayed_statistic",
        )
        for variant in variants:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
                family, freeze_commit, _, _ = fixture.freeze_confirmatory_family(
                    seed_design_variant=variant
                )
                claim_output = fixture.root / "failed.json"
                failed = run_script(
                    "52_compare_drive_cl_crossed_pdm.py",
                    "--comparison-spec",
                    fixture.spec_path,
                    "--family-repository",
                    fixture.root,
                    "--family-spec",
                    family,
                    "--family-freeze-commit",
                    freeze_commit,
                    "--output",
                    claim_output,
                )
                audit_rejected_cli("P0-15", variant, failed, claim_output)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(claim_output.exists())

    def test_p0_16_strict_raw_git_chronology_is_required(self) -> None:
        variants = (
            "family_equals_authorization",
            "authorization_equals_access",
            "future_intermediate_then_earlier_access",
            "reversed_endpoint_commit_times",
        )
        for variant in variants:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
                _, _, access, access_commit = fixture.freeze_confirmatory_family(
                    access_variant=variant
                )
                claim_output = fixture.run_paths[("sequential", 0)]["evaluator"]
                failed = fixture.evaluate_one(
                    "sequential",
                    0,
                    access_receipt=access,
                    access_commit=access_commit,
                )
                audit_rejected_cli("P0-16", variant, failed, claim_output)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(claim_output.exists())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            claim_output = root / "forbidden.json"
            code = (
                "from datetime import datetime, timezone, timedelta; "
                "from selector_bench.continual.claim_protocol import "
                "require_strict_final_test_chronology; "
                "a=datetime(2026,1,1,tzinfo=timezone.utc); "
                "b=a+timedelta(seconds=1); c=b+timedelta(seconds=1); "
                "require_strict_final_test_chronology(a,b,c,c)"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "selector_bench")
            failed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            audit_rejected_cli(
                "P0-16", "access_equals_evaluator", failed, claim_output
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(claim_output.exists())

        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
            family_a, _, access_a, access_commit_a = fixture.freeze_confirmatory_family()
            fixture.evaluate_all(access_receipt=access_a, access_commit=access_commit_a)
            family_bad = fixture.root / "family_bad_history.json"
            ledger_bad = fixture.root / "sealed_test_access_ledger_bad_history.json"
            family_payload = json.loads(family_a.read_text())
            family_payload["family_id"] = "p003_bad_history"
            family_payload["claim_receipt_root"] = "claims_bad_history"
            for comparison in family_payload["expected_comparisons"]:
                comparison["comparison_receipt_repository_path"] = (
                    "claims_bad_history/"
                    + Path(comparison["comparison_receipt_repository_path"]).name
                )
            for hypothesis in family_payload["hypotheses"]:
                hypothesis["comparison_receipt_repository_path"] = (
                    "claims_bad_history/"
                    + Path(hypothesis["comparison_receipt_repository_path"]).name
                )
            write_json(
                ledger_bad,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_bad_history",
                    "events": [],
                },
            )
            family_payload["sealed_test_access_ledger_repository_path"] = (
                ledger_bad.relative_to(fixture.root).as_posix()
            )
            family_payload["sealed_test_access_ledger_empty_sha256"] = sha256(ledger_bad)
            write_json(family_bad, family_payload)
            freeze_bad = git_commit(
                fixture.root, "freeze bad-history family", family_bad, ledger_bad
            )
            family_epoch = int(
                subprocess.check_output(
                    ["git", "-C", str(fixture.root), "show", "-s", "--format=%ct", freeze_bad],
                    text=True,
                ).strip()
            )
            authorized_at = datetime.fromtimestamp(
                family_epoch + 1, tz=timezone.utc
            ).isoformat()
            event_id = hashlib.sha256(f"bad:{freeze_bad}".encode()).hexdigest()[:32]
            event = {
                "access_event_id": event_id,
                "authorized_at_utc": authorized_at,
                "family_id": "p003_bad_history",
                "family_spec_sha256": sha256(family_bad),
                "evaluation_cell_receipt_sha256": sha256(fixture.evaluation_receipt),
                "protocol_content_sha256": fixture.protocol["content_sha256"],
                "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
            }
            prior = dict(event)
            prior["access_event_id"] = "e" * 32
            write_json(
                ledger_bad,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_bad_history",
                    "events": [prior],
                },
            )
            git_commit(
                fixture.root,
                "bad intermediate ledger event",
                ledger_bad,
                commit_epoch=family_epoch + 20,
            )
            write_json(
                ledger_bad,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_bad_history",
                    "events": [],
                },
            )
            git_commit(
                fixture.root,
                "delete bad intermediate event",
                ledger_bad,
                commit_epoch=family_epoch + 30,
            )
            write_json(
                ledger_bad,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_bad_history",
                    "events": [event],
                },
            )
            access_bad = fixture.root / "sealed_access_bad_history.json"
            write_json(
                access_bad,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access.v2",
                    "status": "authorized",
                    "family_id": "p003_bad_history",
                    "family_spec_repository_path": family_bad.relative_to(
                        fixture.root
                    ).as_posix(),
                    "family_spec_sha256": sha256(family_bad),
                    "family_freeze_commit": freeze_bad,
                    "access_ledger_repository_path": ledger_bad.relative_to(
                        fixture.root
                    ).as_posix(),
                    "access_ledger_sha256": sha256(ledger_bad),
                    "access_event_id": event_id,
                    "access_event_index": 0,
                    "authorized_at_utc": authorized_at,
                    "evaluation_cell_receipt_sha256": sha256(
                        fixture.evaluation_receipt
                    ),
                    "protocol_content_sha256": fixture.protocol["content_sha256"],
                    "evaluation_domain": "navsim-fixture/v1::toy_cl::stage_1::old_domain",
                },
            )
            access_commit_bad = git_commit(
                fixture.root,
                "authorize retargeted bad history",
                ledger_bad,
                access_bad,
                commit_epoch=family_epoch + 40,
            )
            family_time = datetime.fromtimestamp(family_epoch, tz=timezone.utc).isoformat()
            access_time = datetime.fromtimestamp(
                family_epoch + 40, tz=timezone.utc
            ).isoformat()
            for paths in fixture.run_paths.values():
                evaluator = json.loads(paths["evaluator"].read_text())
                original_access = evaluator["sealed_test_access"]
                evaluator["sealed_test_access"] = {
                    **original_access,
                    "access_repository": str(fixture.root),
                    "access_repository_path": access_bad.relative_to(
                        fixture.root
                    ).as_posix(),
                    "access_receipt": str(access_bad),
                    "access_receipt_sha256": sha256(access_bad),
                    "access_commit": access_commit_bad,
                    "access_receipt_repository_path": access_bad.relative_to(
                        fixture.root
                    ).as_posix(),
                    "family_spec": str(family_bad),
                    "family_spec_repository_path": family_bad.relative_to(
                        fixture.root
                    ).as_posix(),
                    "family_spec_sha256": sha256(family_bad),
                    "family_freeze_commit": freeze_bad,
                    "family_id": "p003_bad_history",
                    "access_ledger": str(ledger_bad),
                    "access_ledger_repository_path": ledger_bad.relative_to(
                        fixture.root
                    ).as_posix(),
                    "sealed_test_access_ledger_empty_sha256": family_payload[
                        "sealed_test_access_ledger_empty_sha256"
                    ],
                    "access_ledger_sha256": sha256(ledger_bad),
                    "access_event_index": 0,
                    "access_event_id": event_id,
                    "authorized_at_utc": authorized_at,
                    "family_commit_time_utc": family_time,
                    "access_commit_time_utc": access_time,
                }
                write_json(paths["evaluator"], evaluator)
                summary = json.loads(paths["summary"].read_text())
                summary["evaluator_run_receipt_sha256"] = sha256(paths["evaluator"])
                write_json(paths["summary"], summary)
            claim_output = fixture.root / "claims_bad_history" / "main.crossed.json"
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--family-repository",
                fixture.root,
                "--family-spec",
                family_bad,
                "--family-freeze-commit",
                freeze_bad,
                "--output",
                claim_output,
            )
            audit_rejected_cli(
                "P0-16", "claim-stage-add-delete-add-history", failed, claim_output
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(claim_output.exists())

    def test_p0_17_test_evidence_is_bound_to_active_family(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
            family, freeze_commit, access, access_commit = fixture.freeze_confirmatory_family()
            fixture.evaluate_all(access_receipt=access, access_commit=access_commit)
            evaluator_path = fixture.run_paths[("sequential", 0)]["evaluator"]
            original = evaluator_path.read_bytes()
            mutations: dict[str, tuple[str, object]] = {
                "family-id": ("family_id", "transplanted-family"),
                "family-spec-path": ("family_spec_repository_path", "other-family.json"),
                "family-spec-sha": ("family_spec_sha256", "0" * 64),
                "family-freeze-commit": ("family_freeze_commit", "0" * 40),
                "ledger-path": ("access_ledger_repository_path", "other-ledger.json"),
                "empty-ledger-sha": (
                    "sealed_test_access_ledger_empty_sha256",
                    "0" * 64,
                ),
                "access-commit": ("access_commit", "0" * 40),
                "access-receipt-sha": ("access_receipt_sha256", "0" * 64),
                "access-ledger-sha": ("access_ledger_sha256", "0" * 64),
                "access-event-id": ("access_event_id", "f" * 32),
                "access-event-index-bool": ("access_event_index", False),
                "evaluation-cell-sha": (
                    "evaluation_cell_receipt_sha256",
                    "0" * 64,
                ),
                "protocol-sha": ("protocol_content_sha256", "0" * 64),
                "evaluation-domain": ("evaluation_domain", "transplanted/domain"),
                "split": ("split", "audit"),
                "evaluator-time": (
                    "evaluator_started_at_utc",
                    "1900-01-01T00:00:00+00:00",
                ),
                "noncanonical-offset-time": (
                    "evaluator_started_at_utc",
                    "2026-08-18T00:00:00+08:00",
                ),
            }
            for variant, (field, value) in mutations.items():
                with self.subTest(variant=variant):
                    payload = json.loads(original)
                    payload["sealed_test_access"][field] = value
                    write_json(evaluator_path, payload)
                    claim_output = fixture.root / "claims" / f"attack-{variant}.json"
                    failed = run_script(
                        "52_compare_drive_cl_crossed_pdm.py",
                        "--comparison-spec",
                        fixture.spec_path,
                        "--family-repository",
                        fixture.root,
                        "--family-spec",
                        family,
                        "--family-freeze-commit",
                        freeze_commit,
                        "--output",
                        claim_output,
                    )
                    audit_rejected_cli("P0-17", variant, failed, claim_output)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertFalse(claim_output.exists())
                    evaluator_path.write_bytes(original)

            family_b = fixture.root / "family_b.json"
            ledger_b = fixture.root / "sealed_test_access_ledger_b.json"
            write_json(
                ledger_b,
                {
                    "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                    "family_id": "p003_transplant_target",
                    "events": [],
                },
            )
            family_payload = json.loads(family.read_text())
            family_payload["family_id"] = "p003_transplant_target"
            family_payload["sealed_test_access_ledger_repository_path"] = ledger_b.relative_to(
                fixture.root
            ).as_posix()
            family_payload["sealed_test_access_ledger_empty_sha256"] = sha256(ledger_b)
            family_payload["claim_receipt_root"] = "claims_b"
            for comparison in family_payload["expected_comparisons"]:
                comparison["comparison_receipt_repository_path"] = (
                    "claims_b/" + Path(comparison["comparison_receipt_repository_path"]).name
                )
            for hypothesis in family_payload["hypotheses"]:
                hypothesis["comparison_receipt_repository_path"] = (
                    "claims_b/" + Path(hypothesis["comparison_receipt_repository_path"]).name
                )
            write_json(family_b, family_payload)
            freeze_b = git_commit(
                fixture.root, "freeze transplant target family", family_b, ledger_b
            )
            claim_output = fixture.root / "claims_b" / "main.crossed.json"
            failed = run_script(
                "52_compare_drive_cl_crossed_pdm.py",
                "--comparison-spec",
                fixture.spec_path,
                "--family-repository",
                fixture.root,
                "--family-spec",
                family_b,
                "--family-freeze-commit",
                freeze_b,
                "--output",
                claim_output,
            )
            audit_rejected_cli("P0-17", "cross-family-transplant", failed, claim_output)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(claim_output.exists())

    def test_p0_18_shared_cell_requires_two_candidates_one_baseline(self) -> None:
        variants = (
            "single-candidate",
            "duplicate-candidate",
            "different-baselines",
            "undeclared-candidate",
        )
        for variant in variants:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
                family, _, _, _ = fixture.freeze_confirmatory_family()
                family_payload = json.loads(family.read_text())
                ledger = fixture.root / "sealed_test_access_ledger.json"
                write_json(
                    ledger,
                    {
                        "schema": "selector_bench.drive_cl_sealed_test_access_ledger.v1",
                        "family_id": "p003_primary",
                        "events": [],
                    },
                )
                family_payload["sealed_test_access_ledger_empty_sha256"] = sha256(ledger)
                shared = family_payload["shared_confirmatory_cells"][0]
                if variant == "single-candidate":
                    family_payload["expected_comparisons"] = family_payload[
                        "expected_comparisons"
                    ][:1]
                    family_payload["hypotheses"] = [
                        item
                        for item in family_payload["hypotheses"]
                        if item["comparison_id"] == "main"
                    ]
                    shared["minimum_distinct_candidate_count"] = 1
                    shared["candidates"] = shared["candidates"][:1]
                elif variant == "duplicate-candidate":
                    shared["candidates"][1] = dict(shared["candidates"][0])
                elif variant == "different-baselines":
                    family_payload["expected_comparisons"][1][
                        "baseline_method"
                    ] = "planner_only"
                    family_payload["expected_comparisons"][1][
                        "baseline_training_arm"
                    ] = "planner_only"
                    for item in family_payload["hypotheses"]:
                        if item["comparison_id"] == "lwf_vs_sequential":
                            item["baseline_method"] = "planner_only"
                            item["baseline_training_arm"] = "planner_only"
                else:
                    shared["candidates"][1] = {
                        "method": "planner_only",
                        "training_arm": "planner_only",
                    }
                write_json(family, family_payload)
                mutant_commit = git_commit(
                    fixture.root, f"freeze {variant} family", family, ledger
                )
                claim_output = fixture.root / "claims" / "main.crossed.json"
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
                    claim_output,
                )
                audit_rejected_cli("P0-18", variant, failed, claim_output)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(claim_output.exists())

    def test_p0_19_seed_design_rejects_nonfinite_and_boolean_numerics(self) -> None:
        receipt_variants = [
            f"nonfinite_{location}_{value}"
            for location in (
                "minimum-effect",
                "target-power",
                "critical-value",
                "metric-power",
                "heldout-fwer",
            )
            for value in ("nan", "posinf", "neginf")
        ] + [
            f"family-alpha_{value}"
            for value in ("nan", "posinf", "neginf", "bool")
        ] + [
            "bool_designed_seed",
            "bool_candidate_seed",
            "bool_family_metric_count",
            "bool_simulation_seed",
        ]
        for variant in receipt_variants:
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                fixture = ClaimFixture(Path(directory), tuple(range(9)), "test")
                family, freeze_commit, _, _ = fixture.freeze_confirmatory_family(
                    seed_design_variant=variant
                )
                claim_output = fixture.root / "failed.json"
                failed = run_script(
                    "52_compare_drive_cl_crossed_pdm.py",
                    "--comparison-spec",
                    fixture.spec_path,
                    "--family-repository",
                    fixture.root,
                    "--family-spec",
                    family,
                    "--family-freeze-commit",
                    freeze_commit,
                    "--output",
                    claim_output,
                )
                audit_rejected_cli("P0-19", variant, failed, claim_output)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(claim_output.exists())

        for value_name, value in (
            ("nan", float("nan")),
            ("posinf", float("inf")),
            ("neginf", float("-inf")),
            ("bool", True),
        ):
            with self.subTest(variant=f"pilot-{value_name}"), TemporaryDirectory() as directory:
                root = Path(directory)
                pilot = root / "pilot.json"
                matrix = [[0.00, 0.01], [0.02, 0.03]]
                payload = {
                    "schema": "selector_bench.drive_cl_audit_pilot_matrix.v1",
                    "split": "audit",
                    "metrics": list(PDM_DEFAULT_METRICS),
                    "seed_ids": [0, 1],
                    "session_ids": ["session-a", "session-b"],
                    "source_receipts": ["audit.json"],
                    "source_receipt_sha256": ["0" * 64],
                    "candidate_minus_baseline": {
                        metric: [list(row) for row in matrix]
                        for metric in PDM_DEFAULT_METRICS
                    },
                }
                payload["candidate_minus_baseline"]["score"][0][0] = value
                write_json(pilot, payload, allow_nan=True)
                claim_output = root / "seed_design.json"
                failed = run_script(
                    "55_design_drive_cl_confirmatory_seeds.py",
                    "--pilot-matrix",
                    pilot,
                    "--candidate-seed-ids",
                    "0,1,2,3,4,5,6,7,8",
                    "--minimum-relevant-effect",
                    0.1,
                    "--target-power",
                    0.8,
                    "--family-alpha",
                    0.05,
                    "--simulation-repetitions",
                    10000,
                    "--simulation-seed",
                    0,
                    "--repository",
                    REPOSITORY_ROOT,
                    "--output",
                    claim_output,
                )
                audit_rejected_cli(
                    "P0-19", f"pilot-{value_name}", failed, claim_output
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(claim_output.exists())

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
            self.assertEqual(first_result["hypothesis_count"], 14)
            self.assertEqual(first_result["comparison_receipt_count"], 2)
            self.assertEqual(first_result["results"], second_result["results"])
            for chain in (first, second):
                for paths in chain["fixture"].run_paths.values():
                    access = json.loads(paths["evaluator"].read_text())[
                        "sealed_test_access"
                    ]
                    instants = [
                        datetime.fromisoformat(access[field])
                        for field in (
                            "family_commit_time_utc",
                            "authorized_at_utc",
                            "access_commit_time_utc",
                            "evaluator_started_at_utc",
                        )
                    ]
                    self.assertEqual(len(set(instants)), 4)
                    require_strict_final_test_chronology(*instants)

    def test_claim_path_json_emissions_disable_nan(self) -> None:
        paths = [
            REPOSITORY_ROOT
            / "selector_bench"
            / "selector_bench"
            / "continual"
            / name
            for name in ("claim_protocol.py", "seed_design.py")
        ]
        paths.extend(sorted(SCRIPT_ROOT.glob("[4-5][0-9]_*.py")))
        emission_count = 0
        missing: list[str] = []
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"
                    and node.func.attr in {"dump", "dumps"}
                ):
                    continue
                emission_count += 1
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                allow_nan = keywords.get("allow_nan")
                if not (
                    isinstance(allow_nan, ast.Constant)
                    and allow_nan.value is False
                ):
                    missing.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
        self.assertGreater(emission_count, 0)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Emit a clean-commit CPU audit receipt for the Drive-CL claim protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


AUDITED_SOURCES = (
    "selector_bench/selector_bench/continual/claim_protocol.py",
    "selector_bench/selector_bench/continual/drive_opd.py",
    "selector_bench/selector_bench/continual/evaluation_contract.py",
    "selector_bench/selector_bench/continual/navsim_protocol.py",
    "selector_bench/selector_bench/continual/run_budget.py",
    "selector_bench/selector_bench/continual/seed_design.py",
    "selector_bench/selector_bench/continual/statistics.py",
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
    "selector_bench/configs/drive_cl_crossed_comparison_spec.example.json",
    "selector_bench/configs/drive_cl_global_holm_family.example.json",
    "selector_bench/configs/drive_cl_seed_design.example.json",
    "selector_bench/configs/drive_cl_sealed_test_access_ledger.example.json",
    "selector_bench/configs/drive_cl_method_registry.v1.json",
    "selector_bench/configs/drive_cl_metric_registry.v1.json",
    "selector_bench/tests/test_navsim_continual_protocol.py",
    "selector_bench/tests/test_drive_cl_baselines.py",
    "selector_bench/tests/test_drive_opd_losses.py",
    "selector_bench/tests/test_drive_cl_statistics.py",
    "selector_bench/tests/test_drive_cl_claim_protocol.py",
    "selector_bench/tests/test_drive_cl_run_budget.py",
)

TEST_FILES = (
    "selector_bench/tests/test_navsim_continual_protocol.py",
    "selector_bench/tests/test_drive_opd_losses.py",
    "selector_bench/tests/test_drive_cl_baselines.py",
    "selector_bench/tests/test_drive_cl_statistics.py",
    "selector_bench/tests/test_drive_cl_claim_protocol.py",
    "selector_bench/tests/test_drive_cl_run_budget.py",
)


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode != 0:
        raise AuditError(f"Git command failed: {' '.join(arguments)}\n{process.stdout}")
    return process.stdout.strip()


def command_receipt(command: list[str], *, repo: Path, environment: dict[str, str]) -> dict[str, object]:
    process = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "exit_code": process.returncode,
        "output": process.stdout,
        "output_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    python = args.python.resolve()
    if not python.is_file():
        raise AuditError(f"Python interpreter does not exist: {python}")
    dirty = git(repo, "status", "--porcelain")
    if dirty:
        raise AuditError("claim-protocol audit requires a clean source commit")
    branch = git(repo, "branch", "--show-current")
    commit = git(repo, "rev-parse", "HEAD")
    sources: dict[str, str] = {}
    for relative in AUDITED_SOURCES:
        path = repo / relative
        if not path.is_file():
            raise AuditError(f"missing audited source: {relative}")
        sources[relative] = sha256(path)
    for relative in (
        "selector_bench/configs/drive_cl_crossed_comparison_spec.example.json",
        "selector_bench/configs/drive_cl_global_holm_family.example.json",
        "selector_bench/configs/drive_cl_seed_design.example.json",
        "selector_bench/configs/drive_cl_sealed_test_access_ledger.example.json",
        "selector_bench/configs/drive_cl_method_registry.v1.json",
        "selector_bench/configs/drive_cl_metric_registry.v1.json",
    ):
        json.loads((repo / relative).read_text())

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "selector_bench")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    compile_receipt = command_receipt(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; import sys; "
                "[(compile(Path(p).read_text(), p, 'exec')) for p in sys.argv[1:]]"
            ),
            *(relative for relative in AUDITED_SOURCES if relative.endswith(".py")),
        ],
        repo=repo,
        environment=environment,
    )
    test_receipt = command_receipt(
        [str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST_FILES],
        repo=repo,
        environment=environment,
    )
    attack_output = args.output.parent / "p0_attack_audit.json"
    attack_receipt = command_receipt(
        [
            str(python),
            "selector_bench/scripts/57_audit_drive_cl_p0_attacks.py",
            "--repo",
            str(repo),
            "--python",
            str(python),
            "--output",
            str(attack_output),
        ],
        repo=repo,
        environment=environment,
    )
    diff_check = command_receipt(
        ["git", "diff", "--check"], repo=repo, environment=environment
    )
    status = (
        "PASS"
        if compile_receipt["exit_code"] == 0
        and test_receipt["exit_code"] == 0
        and attack_receipt["exit_code"] == 0
        and diff_check["exit_code"] == 0
        else "FAIL"
    )
    payload = {
        "schema": "selector_bench.drive_cl_claim_protocol_audit.v1",
        "status": status,
        "source_branch": branch,
        "source_commit": commit,
        "source_worktree_clean_before_audit": True,
        "python": str(python),
        "source_sha256": sources,
        "json_examples_parse": True,
        "compile": compile_receipt,
        "tests": test_receipt,
        "p0_attack_audit": {
            **attack_receipt,
            "receipt": str(attack_output),
            "receipt_sha256": sha256(attack_output) if attack_output.is_file() else None,
        },
        "git_diff_check": diff_check,
        "evidence_boundary": (
            "CPU evidence for identity, multiplicity and adapter-counter contracts only; "
            "no official-model batch, cache, training or NAVSIM performance was executed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

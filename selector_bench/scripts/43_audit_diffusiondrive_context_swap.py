#!/usr/bin/env python3
"""Causal first gate for perception drift via four DiffusionDrive context swaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AuditError(RuntimeError):
    pass


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(value) for value in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value


class TokenDataset(Dataset):
    def __init__(self, dataset: Any, tokens: list[str]) -> None:
        lookup = {token: index for index, token in enumerate(dataset.tokens)}
        missing = sorted(set(tokens) - set(lookup))
        if missing:
            raise AuditError(f"audit tokens absent from cache: {missing[:3]}")
        self.dataset = dataset
        self.tokens = tokens
        self.indices = [lookup[token] for token in tokens]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[str, Any, Any]:
        features, targets = self.dataset[self.indices[index]]
        return self.tokens[index], features, targets


def build_agent(rap_root: Path, checkpoint: Path) -> Any:
    sys.path.insert(0, str(rap_root.resolve()))
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    import navsim.agents.diffusiondrive.transfuser_config as config_module
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

    config = TransfuserConfig(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5)
    )
    config.waymo = False
    config.distill_feature = True
    config.latent = True
    config.plan_anchor_path = str(
        Path(config_module.__file__).with_name("kmeans_navsim_traj_20.npy")
    )
    agent = TransfuserAgent(config=config, lr=1.5e-4, checkpoint_path=None)
    payload = torch.load(checkpoint, map_location="cpu")
    state = {
        name.removeprefix("agent."): value for name, value in payload["state_dict"].items()
    }
    incompatible = agent.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AuditError(f"checkpoint mismatch: {incompatible}")
    return agent


def response_metrics(
    candidate: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from selector_bench.continual.drive_opd import bernoulli_mode_forward_kl

    response, logits = candidate
    reference_response, reference_logits = reference
    elementwise_mode_kl = bernoulli_mode_forward_kl(
        reference_logits, logits, reduction="none"
    )
    return {
        "response_rmse": (response - reference_response).square().flatten(1).mean(1).sqrt(),
        "mode_bernoulli_forward_kl": elementwise_mode_kl.flatten(1).mean(1),
        # Argmax is retained only as a selected-anchor consistency audit; the
        # underlying sigmoid scores are independent Bernoulli outputs.
        "mode_selected_anchor_changed": (
            logits.argmax(-1) != reference_logits.argmax(-1)
        ).float(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--split", choices=("train", "audit"), default="audit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rap-root", type=Path, default=Path("/home/rguo/RAP-main/RAP-main"))
    parser.add_argument(
        "--cache", type=Path, default=Path("/home/rguo/rap_workspace/exp/training_cache")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AuditError("bind exactly one GPU through gpu-run")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.rap_root.resolve()))
    from navsim.planning.training.dataset import CacheOnlyDataset
    from selector_bench.continual.drive_opd import (
        DiffusionDriveOPDAdapter,
        PlanningDistillationConfig,
    )

    manifest = json.loads(args.protocol_manifest.read_text())
    stage = next(
        item for item in manifest["stages"] if item["stage_index"] == args.stage_index
    )
    tokens = sorted(stage["splits"][args.split]["tokens"])
    if args.max_tokens:
        tokens = sorted(tokens, key=lambda token: stable_seed(args.seed, token))[: args.max_tokens]
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    teacher = build_agent(args.rap_root, args.teacher_checkpoint).to(device).eval()
    student = build_agent(args.rap_root, args.student_checkpoint).to(device).eval()
    adapter = DiffusionDriveOPDAdapter(student, teacher)
    full = CacheOnlyDataset(
        args.cache,
        student.get_feature_builders(),
        student.get_target_builders(),
        use_cache_list=True,
    )
    dataset = TokenDataset(full, tokens)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    rows: list[dict[str, Any]] = []
    planning = PlanningDistillationConfig()
    for batch_index, (batch_tokens, features, targets) in enumerate(loader):
        features = move(features, device)
        targets = move(targets, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(args.seed, args.stage_index, batch_index))
        state = adapter.exogenous_states(targets, planning, generator=generator)[0]
        with torch.no_grad():
            responses = adapter.context_swap_responses(features, state, timestep=10)
        reference = responses["teacher_perception_teacher_planner"]
        comparisons = {
            name: response_metrics(value, reference)
            for name, value in responses.items()
            if name != "teacher_perception_teacher_planner"
        }
        for sample_index, token in enumerate(batch_tokens):
            row: dict[str, Any] = {"token": token}
            for name, metrics in comparisons.items():
                for metric, value in metrics.items():
                    row[f"{name}.{metric}"] = float(value[sample_index].item())
            rows.append(row)

    metric_names = sorted(set(rows[0]) - {"token"})
    summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "median": float(np.median([row[metric] for row in rows])),
            "p95": float(np.percentile([row[metric] for row in rows], 95)),
        }
        for metric in metric_names
    }
    result = {
        "schema": "selector_bench.diffusiondrive_context_swap_audit.v2",
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_sha256": sha256(args.teacher_checkpoint),
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "student_sha256": sha256(args.student_checkpoint),
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "protocol_content_sha256": manifest["content_sha256"],
        "stage_index": args.stage_index,
        "split": args.split,
        "sample_count": len(rows),
        "common_state": "gt_trajectory_plus_shared_noise_at_t10",
        "mode_score_semantics": "independent_per_anchor_sigmoid_bernoulli",
        "reference": "teacher_perception_teacher_planner",
        "summary": summary,
        "rows": rows,
        "evidence_boundary": (
            "planning-response causality gate only; a material shift requires a subsequent "
            "full PDMS context-swap evaluation before claiming metric causality"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "result.json.tmp"
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output_dir / "result.json")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

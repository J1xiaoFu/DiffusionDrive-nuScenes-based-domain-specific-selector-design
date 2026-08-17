#!/usr/bin/env python3
"""Single-update DiffusionDrive continual-learning runner for the Seed-0 funnel.

Every arm uses the same official NAVSIM loss, data order, optimizer and number
of optimizer steps.  OPD/LwF add their auxiliary scalar before the one backward
and AdamW step; they never take an alternating repair step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ARMS = (
    "sequential",
    "planner_only",
    "ewc",
    "agem",
    "aler_drive",
    "step_matched_replay",
    "full_exposure_replay",
    "lwf",
    "fixed_opd",
    "ema099_opd",
    "perception_only",
    "planning_only",
)
CHECKPOINT_SCHEMA = "selector_bench.drive_cl_diffusiondrive_checkpoint.v1"


class RunnerError(RuntimeError):
    pass


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def slice_batch(value: Any, count: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[:count]
    if isinstance(value, Mapping):
        return {key: slice_batch(item, count) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(slice_batch(item, count) for item in value)
    if isinstance(value, list):
        return value[:count]
    return value


class TokenDataset(Dataset):
    def __init__(self, dataset: Any, tokens: Sequence[str]) -> None:
        token_to_index = {token: index for index, token in enumerate(dataset.tokens)}
        missing = sorted(set(tokens) - set(token_to_index))
        if missing:
            raise RunnerError(f"{len(missing)} manifest tokens are absent from cache: {missing[:3]}")
        self.dataset = dataset
        self.tokens = list(tokens)
        self.indices = [token_to_index[token] for token in self.tokens]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.dataset[self.indices[index]]


def build_agent(rap_root: Path, learning_rate: float) -> Any:
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
    return TransfuserAgent(config=config, lr=learning_rate, checkpoint_path=None)


def build_optimizer_scheduler(agent: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR

    optimizer = agent.get_optimizers()["optimizer"]
    scheduler = WarmupCosLR(
        optimizer,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        epochs=args.schedule_epochs,
        warmup_epochs=args.warmup_epochs,
    )
    return optimizer, scheduler


def _agent_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = payload["state_dict"]
    if all(key.startswith("agent.") for key in state):
        return {key.removeprefix("agent."): value for key, value in state.items()}
    return dict(state)


def restore_training_state(
    checkpoint: Path,
    agent: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, int]:
    payload = torch.load(checkpoint, map_location="cpu")
    incompatible = agent.load_state_dict(_agent_state(payload), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RunnerError(f"checkpoint state mismatch: {incompatible}")
    optimizer_states = payload.get("optimizer_states", [])
    scheduler_states = payload.get("lr_schedulers", [])
    if optimizer_states:
        if len(optimizer_states) != 1:
            raise RunnerError("checkpoint must contain at most one optimizer")
        optimizer.load_state_dict(optimizer_states[0])
    if scheduler_states:
        if len(scheduler_states) != 1:
            raise RunnerError("checkpoint must contain at most one scheduler")
        scheduler.load_state_dict(scheduler_states[0])
    completed_epoch = (
        int(payload["completed_epoch"])
        if "completed_epoch" in payload
        else int(payload.get("epoch", -1)) + 1
    )
    return completed_epoch, int(payload.get("global_step", 0))


def checkpoint_payload(
    agent: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    completed_epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "completed_epoch": completed_epoch,
        "epoch": completed_epoch - 1,
        "global_step": global_step,
        "state_dict": {
            f"agent.{name}": value.detach().cpu() for name, value in agent.state_dict().items()
        },
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        "experiment": dict(protocol),
    }


def atomic_checkpoint(payload: Mapping[str, Any], path: Path) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def stage_tokens(manifest: Mapping[str, Any], stage_index: int, split: str) -> list[str]:
    for stage in manifest["stages"]:
        if int(stage["stage_index"]) == stage_index:
            return list(stage["splits"][split]["tokens"])
    raise RunnerError(f"stage {stage_index} is absent from protocol")


def training_tokens(
    manifest: Mapping[str, Any],
    arm: str,
    stage_index: int,
    seed: int,
    batch_size: int = 64,
) -> list[str]:
    current = stage_tokens(manifest, stage_index, "train")
    old = [
        token
        for prior_stage in range(1, stage_index)
        for token in stage_tokens(manifest, prior_stage, "train")
    ]
    if arm == "full_exposure_replay" and old:
        return current + old
    if arm == "step_matched_replay" and old:
        from selector_bench.continual.baselines import deterministic_replay_choice

        result: list[str] = []
        batch_size = min(batch_size, len(current))
        for step in range(math.ceil(len(current) / batch_size)):
            new, replay = deterministic_replay_choice(
                current,
                old,
                step=step,
                seed=seed,
                replay_fraction=0.5,
                batch_size=batch_size,
            )
            result.extend(new + replay)
        return result[: len(current)]
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rap-root", type=Path, default=Path("/home/rguo/RAP-main/RAP-main"))
    parser.add_argument(
        "--cache", type=Path, default=Path("/home/rguo/rap_workspace/exp/training_cache")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--schedule-epochs", type=int, default=180)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--opd-microbatch", type=int, default=8)
    parser.add_argument("--lambda-perception", type=float, default=1.0)
    parser.add_argument("--lambda-planning", type=float, default=1.0)
    parser.add_argument("--agent-confidence", type=float, default=0.7)
    parser.add_argument("--bev-confidence", type=float, default=0.7)
    parser.add_argument("--ewc-fisher", type=Path)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    parser.add_argument("--agem-buffer-size", type=int, default=1024)
    parser.add_argument("--agem-replay-batch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if args.stage_index > 1 and args.source_checkpoint is None:
        parser.error("stage 2/3 requires the preceding stage endpoint checkpoint")
    for label, path in (
        ("protocol manifest", args.protocol_manifest),
        ("RAP root", args.rap_root),
        ("cache", args.cache),
    ):
        if not path.exists():
            parser.error(f"{label} is missing: {path}")
    if args.source_checkpoint is not None and not args.source_checkpoint.is_file():
        parser.error(f"source checkpoint is missing: {args.source_checkpoint}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.opd_microbatch <= 0:
        parser.error("epochs and batch sizes must be positive")
    if args.max_steps < 0 or args.workers < 0:
        parser.error("max steps and workers must be non-negative")
    if args.arm == "ewc" and (args.ewc_fisher is None or not args.ewc_fisher.is_file()):
        parser.error("EWC requires --ewc-fisher from the qualified stage-start checkpoint")
    if args.ewc_lambda < 0 or args.agem_buffer_size <= 0 or args.agem_replay_batch <= 0:
        parser.error("invalid EWC/A-GEM hyperparameter")
    return args


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.rap_root.resolve()))
    from navsim.planning.training.dataset import CacheOnlyDataset
    from selector_bench.continual.drive_opd import (
        DriveOPDConfig,
        DiffusionDriveOPDAdapter,
        PerceptionDistillationConfig,
        PlanningDistillationConfig,
        clone_frozen_teacher,
        perception_distillation_loss,
        update_ema_teacher_,
    )
    from selector_bench.continual.baselines import (
        EWCState,
        aler_adversarial_latent_search,
        aler_repair_loss,
        capture_gradient,
        nearest_manifold_distance,
        project_agem_gradient_,
        set_gradient_,
        trainable_parameters,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RunnerError("bind exactly one idle GPU through /home/rguo/bin/gpu-run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = os.statvfs(args.output_dir).f_bavail * os.statvfs(args.output_dir).f_frsize
    if not args.no_save and free_bytes < 4 * 1024**3:
        raise RunnerError("less than 4 GiB free before checkpointed run")
    manifest = json.loads(args.protocol_manifest.read_text())
    tokens = training_tokens(
        manifest, args.arm, args.stage_index, args.seed, batch_size=args.batch_size
    )
    if not tokens:
        raise RunnerError("training token list is empty")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    agent = build_agent(args.rap_root, args.learning_rate).to(device)
    optimizer, scheduler = build_optimizer_scheduler(agent, args)
    completed_epoch = 0
    global_step = 0
    if args.source_checkpoint is not None:
        completed_epoch, global_step = restore_training_state(
            args.source_checkpoint, agent, optimizer, scheduler
        )

    if args.arm == "planner_only":
        for parameter in agent.parameters():
            parameter.requires_grad_(False)
        for parameter in agent._transfuser_model._trajectory_head.parameters():
            parameter.requires_grad_(True)

    ewc_state = None
    if args.arm == "ewc":
        fisher_payload = torch.load(args.ewc_fisher, map_location="cpu")
        expected_source = sha256_file(args.source_checkpoint)
        if fisher_payload.get("source_checkpoint_sha256") != expected_source:
            raise RunnerError("EWC Fisher was not computed from this stage-start checkpoint")
        ewc_state = EWCState.from_model_and_fisher(
            agent,
            fisher_payload["fisher"],
            int(fisher_payload["sample_count"]),
        )

    distillation_arms = {
        "lwf",
        "fixed_opd",
        "ema099_opd",
        "perception_only",
        "planning_only",
        "aler_drive",
    }
    adapter = None
    teacher = None
    if args.arm in distillation_arms:
        if args.source_checkpoint is None:
            raise RunnerError("distillation requires a qualified stage-start teacher")
        teacher = clone_frozen_teacher(agent)
        adapter = DiffusionDriveOPDAdapter(agent, teacher)

    full_dataset = CacheOnlyDataset(
        args.cache,
        agent.get_feature_builders(),
        agent.get_target_builders(),
        use_cache_list=True,
    )
    dataset = TokenDataset(full_dataset, tokens)
    agem_dataset = None
    agem_buffer_tokens: list[str] = []
    if args.arm == "agem":
        old_tokens = [
            token
            for prior_stage in range(1, args.stage_index)
            for token in stage_tokens(manifest, prior_stage, "train")
        ]
        if not old_tokens:
            raise RunnerError("A-GEM requires at least one previously learned stage")
        agem_buffer_tokens = sorted(
            old_tokens,
            key=lambda token: hashlib.sha256(
                f"{args.seed}\0agem-buffer\0{token}".encode()
            ).digest(),
        )[: args.agem_buffer_size]
        agem_dataset = TokenDataset(full_dataset, agem_buffer_tokens)
    start_epoch = completed_epoch + 1
    end_epoch = completed_epoch + args.epochs
    protocol = {
        "schema": "selector_bench.drive_cl_diffusiondrive_run.v1",
        "arm": args.arm,
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "protocol_manifest_sha256": sha256_file(args.protocol_manifest),
        "protocol_content_sha256": manifest.get("content_sha256"),
        "stage_index": args.stage_index,
        "source_checkpoint": str(args.source_checkpoint.resolve()) if args.source_checkpoint else None,
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint) if args.source_checkpoint else None,
        "teacher_policy": (
            "ema_m0.99" if args.arm == "ema099_opd" else "fixed_stage_start"
            if args.arm in distillation_arms
            else None
        ),
        "distillation_support": "exogenous_gt_noise" if args.arm == "lwf" else "adversarial_exogenous_latent"
        if args.arm == "aler_drive"
        else "student_rollout"
        if args.arm in distillation_arms
        else None,
        "loss_integration": "one_composite_loss_one_backward_one_adamw_step",
        "perception_and_planning_trainable": args.arm != "planner_only",
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "train_tokens": len(tokens),
        "batch_size": args.batch_size,
        "optimizer": type(optimizer).__name__,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "schedule_epochs": args.schedule_epochs,
        "warmup_epochs": args.warmup_epochs,
        "opd_microbatch": args.opd_microbatch,
        "lambda_perception": args.lambda_perception,
        "lambda_planning": args.lambda_planning,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "checkpoint_policy": "stage_endpoint_plus_one_rolling_optimizer_state",
        "ewc_fisher": str(args.ewc_fisher.resolve()) if args.ewc_fisher else None,
        "ewc_lambda": args.ewc_lambda if args.arm == "ewc" else None,
        "agem_buffer_tokens": len(agem_buffer_tokens) if args.arm == "agem" else None,
        "agem_replay_batch": args.agem_replay_batch if args.arm == "agem" else None,
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps({"event": "protocol", **protocol}, sort_keys=True), flush=True)

    distill_config = DriveOPDConfig(
        lambda_perception=(
            0.0 if args.arm == "planning_only" else args.lambda_perception
        ),
        lambda_planning=(
            0.0 if args.arm == "perception_only" else args.lambda_planning
        ),
        perception=PerceptionDistillationConfig(
            agent_confidence=args.agent_confidence,
            bev_confidence=args.bev_confidence,
        ),
        planning=PlanningDistillationConfig(),
    )
    step_log = args.output_dir / "steps.jsonl"
    epoch_log = args.output_dir / "epochs.jsonl"
    steps_this_invocation = 0
    stopped_early = False
    for epoch in range(start_epoch, end_epoch + 1):
        order_generator = torch.Generator()
        order_generator.manual_seed(stable_seed("order", args.seed, args.stage_index, epoch))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=order_generator,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.workers > 0,
        )
        agem_iterator = None
        agem_loader = None
        if agem_dataset is not None:
            replay_generator = torch.Generator()
            replay_generator.manual_seed(
                stable_seed("agem-order", args.seed, args.stage_index, epoch)
            )
            agem_loader = DataLoader(
                agem_dataset,
                batch_size=args.agem_replay_batch,
                shuffle=True,
                generator=replay_generator,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=args.workers > 0,
            )
            agem_iterator = iter(agem_loader)
        agent.train()
        epoch_start = time.monotonic()
        sums: dict[str, float] = {}
        samples = 0
        batches = 0
        for batch_index, (features, targets) in enumerate(loader):
            if args.max_steps and steps_this_invocation >= args.max_steps:
                stopped_early = True
                break
            seed = stable_seed("stock", args.seed, args.stage_index, epoch, batch_index)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            features = move_to_device(features, device)
            targets = move_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            batch_start = time.monotonic()
            distill_metrics: dict[str, float] = {}
            distillation = None
            if adapter is not None:
                micro = min(args.opd_microbatch, int(next(iter(features.values())).shape[0]))
                opd_generator = torch.Generator(device=device)
                opd_generator.manual_seed(
                    stable_seed("distill", args.seed, args.stage_index, epoch, batch_index)
                )
                distill_features = slice_batch(features, micro)
                distill_targets = slice_batch(targets, micro)
                if args.arm == "aler_drive":
                    student_context, teacher_context = adapter.deterministic_contexts(
                        distill_features
                    )
                    perception, perception_metrics = perception_distillation_loss(
                        student_context, teacher_context, distill_config.perception
                    )
                    real_states = adapter.exogenous_states(
                        distill_targets, distill_config.planning, generator=opd_generator
                    )
                    timestep = distill_config.planning.rollout_timesteps[0]
                    # Latent ascent must not consume the perception graph that
                    # will later receive the repair/perception gradients.
                    search_context = student_context.detached()
                    search_student_query = lambda state, time: adapter.query(
                        adapter.student, search_context, state, time
                    )
                    repair_student_query = lambda state, time: adapter.query(
                        adapter.student, student_context, state, time
                    )
                    teacher_query = lambda state, time: adapter.query(
                        adapter.teacher, teacher_context, state, time
                    )
                    with adapter.deterministic_planner_queries():
                        searched, search_metrics = aler_adversarial_latent_search(
                            real_states[0],
                            search_student_query,
                            teacher_query,
                            timestep,
                            search_steps=1,
                            step_size=0.05,
                            radius=0.1,
                        )
                        repair, repair_metrics = aler_repair_loss(
                            searched, repair_student_query, teacher_query, timestep
                        )
                    manifold = nearest_manifold_distance(searched, real_states[0])
                    distillation = (
                        distill_config.lambda_perception * perception
                        + distill_config.lambda_planning * repair
                    )
                    distill_metrics = {
                        "distillation_total": float(distillation.detach().item()),
                        "distillation_support_student": 0.0,
                        **perception_metrics,
                        **search_metrics,
                        **repair_metrics,
                        "aler_teacher_queries": 2.0,
                        "student_denoiser_queries": 2.0,
                        "teacher_denoiser_queries": 2.0,
                        "aler_manifold_l2_mean": float(manifold.mean().item()),
                    }
                else:
                    distillation, distill_metrics = adapter.distillation_loss(
                        distill_features,
                        distill_targets,
                        distill_config,
                        support="exogenous" if args.arm == "lwf" else "student",
                        generator=opd_generator,
                    )
            # Distillation is evaluated before the official train-mode forward.
            # Otherwise the student's current-batch BN buffer update is falsely
            # measured against the unchanged stage-start teacher as historical
            # perception drift even when both checkpoints are initially equal.
            predictions = agent.forward(features, targets)
            stock_terms = agent.compute_loss(features, targets, predictions)
            stock_loss = stock_terms["loss"]
            total_loss = stock_loss
            if distillation is not None:
                total_loss = total_loss + distillation
            if ewc_state is not None:
                ewc_penalty = ewc_state.penalty(agent)
                total_loss = total_loss + args.ewc_lambda * ewc_penalty
                distill_metrics["ewc_penalty"] = float(ewc_penalty.detach().item())
            total_loss.backward()
            if agem_iterator is not None:
                parameters = trainable_parameters(agent)
                current_gradient = capture_gradient(parameters)
                optimizer.zero_grad(set_to_none=True)
                try:
                    replay_features, replay_targets = next(agem_iterator)
                except StopIteration:
                    agem_iterator = iter(agem_loader)
                    replay_features, replay_targets = next(agem_iterator)
                replay_seed = stable_seed(
                    "agem-reference", args.seed, args.stage_index, epoch, batch_index
                )
                random.seed(replay_seed)
                np.random.seed(replay_seed)
                torch.manual_seed(replay_seed)
                torch.cuda.manual_seed_all(replay_seed)
                replay_features = move_to_device(replay_features, device)
                replay_targets = move_to_device(replay_targets, device)
                replay_predictions = agent.forward(replay_features, replay_targets)
                replay_terms = agent.compute_loss(
                    replay_features, replay_targets, replay_predictions
                )
                reference_loss = replay_terms["loss"]
                reference_loss.backward()
                reference_gradient = capture_gradient(parameters)
                set_gradient_(parameters, current_gradient)
                del current_gradient
                distill_metrics.update(
                    project_agem_gradient_(parameters, reference_gradient)
                )
                del reference_gradient
                distill_metrics["agem_reference_loss"] = float(
                    reference_loss.detach().item()
                )
            gradient_squares = [
                parameter.grad.detach().float().square().sum()
                for parameter in agent.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradient_squares:
                raise RunnerError("training produced no gradient")
            gradient_norm = float(torch.stack(gradient_squares).sum().sqrt().item())
            if not math.isfinite(gradient_norm) or not math.isfinite(float(total_loss.detach())):
                raise RunnerError("non-finite loss or gradient")
            optimizer.step()
            if args.arm == "ema099_opd" and teacher is not None:
                update_ema_teacher_(teacher, agent, momentum=0.99)
            global_step += 1
            steps_this_invocation += 1
            current_batch = int(next(iter(features.values())).shape[0])
            samples += current_batch
            batches += 1
            row = {
                "event": "step",
                "arm": args.arm,
                "stage_index": args.stage_index,
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "batch_size": current_batch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "stock_loss": float(stock_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "gradient_norm": gradient_norm,
                "peak_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
                "step_seconds": time.monotonic() - batch_start,
                **distill_metrics,
            }
            for name, value in row.items():
                if isinstance(value, (int, float)) and name not in {
                    "epoch",
                    "batch_index",
                    "global_step",
                    "batch_size",
                    "stage_index",
                }:
                    sums[name] = sums.get(name, 0.0) + float(value)
            if batch_index % args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            append_jsonl(step_log, row)
        if stopped_early:
            break
        scheduler.step()
        completed_epoch = epoch
        epoch_row = {
            "event": "epoch",
            "arm": args.arm,
            "stage_index": args.stage_index,
            "completed_epoch": epoch,
            "global_step": global_step,
            "steps": batches,
            "samples": samples,
            "wall_seconds": time.monotonic() - epoch_start,
            "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            **{name: value / max(batches, 1) for name, value in sums.items()},
        }
        if not args.no_save:
            rolling = args.output_dir / "resume_latest.ckpt"
            digest = atomic_checkpoint(
                checkpoint_payload(
                    agent,
                    optimizer,
                    scheduler,
                    completed_epoch=completed_epoch,
                    global_step=global_step,
                    protocol=protocol,
                ),
                rolling,
            )
            epoch_row["checkpoint"] = str(rolling.resolve())
            epoch_row["checkpoint_sha256"] = digest
        append_jsonl(epoch_log, epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)

    endpoint: str | None = None
    endpoint_sha256: str | None = None
    if not args.no_save and not stopped_early:
        rolling = args.output_dir / "resume_latest.ckpt"
        endpoint_path = args.output_dir / f"stage_{args.stage_index}_endpoint.ckpt"
        if not endpoint_path.exists():
            os.link(rolling, endpoint_path)
        endpoint = str(endpoint_path.resolve())
        endpoint_sha256 = sha256_file(endpoint_path)
    result = {
        "schema": "selector_bench.drive_cl_diffusiondrive_result.v1",
        "arm": args.arm,
        "stage_index": args.stage_index,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "steps_this_invocation": steps_this_invocation,
        "stopped_early": stopped_early,
        "endpoint": endpoint,
        "endpoint_sha256": endpoint_sha256,
    }
    atomic_json(args.output_dir / "result.json", result)
    print(json.dumps({"event": "result", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Driving-native perception and student-support planning distillation.

This module deliberately keeps optimization policy out of the loss: callers
form one composite scalar, call backward once, and take one AdamW step.  OPD
and LwF share every code path except construction of the queried diffusion
states, which makes the student-support hypothesis directly testable.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


TensorMap = Mapping[str, torch.Tensor]
QueryFunction = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class DriveOPDError(RuntimeError):
    """Raised when an architecture or loss invariant is violated."""


@dataclass(frozen=True)
class DiffusionContext:
    ego_query: torch.Tensor
    agents_query: torch.Tensor
    bev_feature: torch.Tensor
    bev_spatial_shape: tuple[int, int]
    status_encoding: torch.Tensor
    bev_semantic_map: torch.Tensor
    agent_states: torch.Tensor
    agent_logits: torch.Tensor

    def detached(self) -> "DiffusionContext":
        return DiffusionContext(
            ego_query=self.ego_query.detach(),
            agents_query=self.agents_query.detach(),
            bev_feature=self.bev_feature.detach(),
            bev_spatial_shape=self.bev_spatial_shape,
            status_encoding=self.status_encoding.detach(),
            bev_semantic_map=self.bev_semantic_map.detach(),
            agent_states=self.agent_states.detach(),
            agent_logits=self.agent_logits.detach(),
        )


@dataclass(frozen=True)
class PerceptionDistillationConfig:
    agent_confidence: float = 0.7
    bev_confidence: float = 0.7
    bev_class_ids: tuple[int, ...] = (1, 3)
    agent_existence_weight: float = 1.0
    agent_position_weight: float = 1.0
    agent_size_weight: float = 0.5
    agent_heading_weight: float = 0.5
    bev_weight: float = 1.0


@dataclass(frozen=True)
class PlanningDistillationConfig:
    response_weight: float = 1.0
    mode_kl_weight: float = 1.0
    rollout_timesteps: tuple[int, ...] = (10, 0)
    initial_noise_timestep: int = 10


@dataclass(frozen=True)
class RolloutSchedule:
    """A DDIM subsequence whose queried states have an unambiguous time label."""

    query_timesteps: tuple[int, ...]
    initial_noise_timestep: int
    transition_stride: int
    scheduler_inference_steps: int


@dataclass(frozen=True)
class DriveOPDConfig:
    lambda_perception: float = 1.0
    lambda_planning: float = 1.0
    perception: PerceptionDistillationConfig = PerceptionDistillationConfig()
    planning: PlanningDistillationConfig = PlanningDistillationConfig()


@dataclass
class PlannerQueryAudit:
    """Runtime trace emitted by the real adapter query path."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def count(self, role: str) -> int:
        return sum(record["role"] == role for record in self.records)

    def assert_budget(self, *, student: int, teacher: int) -> None:
        observed_student = self.count("student")
        observed_teacher = self.count("teacher")
        if observed_student != student or observed_teacher != teacher:
            raise DriveOPDError(
                "real adapter query budget mismatch: "
                f"student expected={student} observed={observed_student}; "
                f"teacher expected={teacher} observed={observed_teacher}"
            )


def _zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def _bernoulli_kl(teacher_probability: torch.Tensor, student_logit: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(student_logit.dtype).eps
    teacher = teacher_probability.to(student_logit.dtype).clamp(epsilon, 1.0 - epsilon)
    student = student_logit.sigmoid().clamp(epsilon, 1.0 - epsilon)
    return teacher * (teacher.log() - student.log()) + (1.0 - teacher) * (
        (1.0 - teacher).log() - (1.0 - student).log()
    )


def matched_agent_distillation(
    student_states: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_states: torch.Tensor,
    teacher_logits: torch.Tensor,
    config: PerceptionDistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hungarian-match high-confidence teacher agents without locking new agents."""

    if student_states.ndim != 3 or student_states.shape[-1] < 5:
        raise DriveOPDError(f"expected student agent states [B,N,>=5], got {student_states.shape}")
    if teacher_states.shape != student_states.shape:
        raise DriveOPDError("student and teacher agent-state shapes must match")
    if student_logits.shape != student_states.shape[:2] or teacher_logits.shape != student_logits.shape:
        raise DriveOPDError("agent logit shapes do not match agent states")

    losses: list[torch.Tensor] = []
    matched_count = 0
    for batch_index in range(student_states.shape[0]):
        teacher_probability = teacher_logits[batch_index].detach().sigmoid()
        selected = torch.nonzero(
            teacher_probability >= config.agent_confidence, as_tuple=False
        ).flatten()
        if selected.numel() == 0:
            continue
        cost = torch.cdist(
            student_states[batch_index, :, :2].detach().float(),
            teacher_states[batch_index, selected, :2].detach().float(),
            p=1,
        )
        student_index_np, selected_index_np = linear_sum_assignment(cost.cpu().numpy())
        student_index = torch.as_tensor(student_index_np, device=student_states.device)
        teacher_index = selected[
            torch.as_tensor(selected_index_np, device=student_states.device)
        ]
        student_box = student_states[batch_index, student_index]
        teacher_box = teacher_states[batch_index, teacher_index].detach()
        existence = _bernoulli_kl(
            teacher_probability[teacher_index], student_logits[batch_index, student_index]
        ).mean()
        position = F.smooth_l1_loss(student_box[:, :2], teacher_box[:, :2])
        heading = (1.0 - torch.cos(student_box[:, 2] - teacher_box[:, 2])).mean()
        size = F.smooth_l1_loss(student_box[:, 3:5], teacher_box[:, 3:5])
        losses.append(
            config.agent_existence_weight * existence
            + config.agent_position_weight * position
            + config.agent_heading_weight * heading
            + config.agent_size_weight * size
        )
        matched_count += int(student_index.numel())
    if not losses:
        return _zero_like(student_states), {"matched_agents": 0.0}
    return torch.stack(losses).mean().clamp_min(0.0), {
        "matched_agents": float(matched_count)
    }


def masked_bev_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    config: PerceptionDistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL-distil confident road/centerline cells while leaving novel cells free."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 4:
        raise DriveOPDError("BEV logits must have matching [B,C,H,W] shapes")
    teacher_probability = teacher_logits.detach().softmax(dim=1)
    confidence, semantic_class = teacher_probability.max(dim=1)
    class_mask = torch.zeros_like(semantic_class, dtype=torch.bool)
    for class_id in config.bev_class_ids:
        class_mask |= semantic_class == int(class_id)
    mask = class_mask & (confidence >= config.bev_confidence)
    if not mask.any():
        return _zero_like(student_logits), {"bev_masked_cells": 0.0, "bev_mask_fraction": 0.0}
    pointwise_kl = F.kl_div(
        student_logits.log_softmax(dim=1),
        teacher_probability,
        reduction="none",
    ).sum(dim=1)
    loss = pointwise_kl[mask].mean().clamp_min(0.0)
    return loss, {
        "bev_masked_cells": float(mask.sum().item()),
        "bev_mask_fraction": float(mask.float().mean().item()),
    }


def perception_distillation_loss(
    student: DiffusionContext,
    teacher: DiffusionContext,
    config: PerceptionDistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    agent_loss, agent_metrics = matched_agent_distillation(
        student.agent_states,
        student.agent_logits,
        teacher.agent_states,
        teacher.agent_logits,
        config,
    )
    bev_loss, bev_metrics = masked_bev_distillation(
        student.bev_semantic_map, teacher.bev_semantic_map, config
    )
    loss = agent_loss + config.bev_weight * bev_loss
    metrics = {
        "perception_agent_loss": float(agent_loss.detach().item()),
        "perception_bev_loss": float(bev_loss.detach().item()),
        **agent_metrics,
        **bev_metrics,
    }
    return loss, metrics


def planning_distillation_loss(
    student_query: QueryFunction,
    teacher_query: QueryFunction,
    states: Sequence[torch.Tensor],
    timesteps: Sequence[int],
    config: PlanningDistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Distil denoising response and mode probabilities at matched states."""

    if not states or len(states) != len(timesteps):
        raise DriveOPDError("states and timesteps must be non-empty and aligned")
    if tuple(int(value) for value in timesteps) != config.rollout_timesteps:
        raise DriveOPDError("registered states must use the frozen rollout timesteps")
    response_losses: list[torch.Tensor] = []
    mode_losses: list[torch.Tensor] = []
    agreements: list[torch.Tensor] = []
    for state, timestep in zip(states, timesteps):
        timestep_tensor = torch.full(
            (state.shape[0],), int(timestep), dtype=torch.long, device=state.device
        )
        student_response, student_mode = student_query(state, timestep_tensor)
        with torch.no_grad():
            teacher_response, teacher_mode = teacher_query(state, timestep_tensor)
        response_losses.append(
            F.mse_loss(student_response, teacher_response.detach())
        )
        mode_losses.append(
            F.kl_div(
                student_mode.log_softmax(dim=-1),
                teacher_mode.detach().softmax(dim=-1),
                reduction="batchmean",
            ).clamp_min(0.0)
        )
        agreements.append(
            (student_mode.argmax(dim=-1) == teacher_mode.argmax(dim=-1)).float().mean()
        )
    response = torch.stack(response_losses).mean()
    mode = torch.stack(mode_losses).mean()
    loss = config.response_weight * response + config.mode_kl_weight * mode
    return loss, {
        "planning_response_mse": float(response.detach().item()),
        "planning_mode_forward_kl": float(mode.detach().item()),
        "planning_mode_top1_agreement": float(torch.stack(agreements).mean().item()),
        "planning_query_states": float(len(states)),
    }


def configure_rollout_schedule(
    scheduler: Any,
    config: PlanningDistillationConfig,
    device: torch.device,
) -> RolloutSchedule:
    """Configure a constant-stride DDIM subsequence and audit every time label.

    DiffusionDrive's historical two-step inference noises an anchor at time 8,
    conditions the first denoiser at time 10, and configures the scheduler for
    unit-size transitions.  That heuristic is not a controlled OPD/LwF state
    comparison.  The research loss instead registers one explicit subsequence:
    the initial forward-noise time equals the first queried time, and each
    scheduler transition lands exactly on the next queried time.
    """

    timesteps = tuple(int(value) for value in config.rollout_timesteps)
    if len(timesteps) < 2 or timesteps[-1] != 0:
        raise DriveOPDError("rollout timesteps must contain at least two times and end at zero")
    if any(left <= right for left, right in zip(timesteps, timesteps[1:])):
        raise DriveOPDError("rollout timesteps must be strictly decreasing")
    strides = tuple(left - right for left, right in zip(timesteps, timesteps[1:]))
    if len(set(strides)) != 1:
        raise DriveOPDError("rollout timesteps must form a constant-stride DDIM subsequence")
    stride = strides[0]
    if config.initial_noise_timestep != timesteps[0]:
        raise DriveOPDError(
            "initial noise timestep must equal the first denoiser query timestep"
        )
    scheduler_config = getattr(scheduler, "config", None)
    if isinstance(scheduler_config, Mapping):
        train_timesteps = scheduler_config.get("num_train_timesteps")
    else:
        train_timesteps = getattr(scheduler_config, "num_train_timesteps", None)
    if train_timesteps is None:
        raise DriveOPDError("diffusion scheduler does not declare num_train_timesteps")
    train_timesteps = int(train_timesteps)
    if timesteps[0] >= train_timesteps or train_timesteps % stride != 0:
        raise DriveOPDError("rollout stride is incompatible with the diffusion training grid")
    inference_steps = train_timesteps // stride
    scheduler.set_timesteps(inference_steps, device)
    registered = getattr(scheduler, "timesteps", None)
    if registered is not None:
        registered_values = [int(value) for value in registered]
        registered_set = set(registered_values)
        missing = sorted(set(timesteps) - registered_set)
        if missing:
            raise DriveOPDError(f"scheduler did not register rollout timesteps: {missing}")
        for left, right in zip(timesteps, timesteps[1:]):
            position = registered_values.index(left)
            if (
                position + 1 >= len(registered_values)
                or registered_values[position + 1] != right
            ):
                raise DriveOPDError(
                    f"scheduler does not transition directly from {left} to {right}"
                )
    return RolloutSchedule(
        query_timesteps=timesteps,
        initial_noise_timestep=config.initial_noise_timestep,
        transition_stride=stride,
        scheduler_inference_steps=inference_steps,
    )


def _rollout_schedule_metrics(schedule: RolloutSchedule) -> dict[str, float]:
    return {
        "planning_schedule_consistent": 1.0,
        "planning_initial_noise_timestep": float(schedule.initial_noise_timestep),
        "planning_transition_stride": float(schedule.transition_stride),
        "planning_scheduler_inference_steps": float(
            schedule.scheduler_inference_steps
        ),
        "planning_first_query_timestep": float(schedule.query_timesteps[0]),
        "planning_last_query_timestep": float(schedule.query_timesteps[-1]),
        "planning_rollout_semigradient": 1.0,
    }


@contextlib.contextmanager
def _temporary_eval(module: torch.nn.Module) -> Iterable[None]:
    states = {item: item.training for item in module.modules()}
    module.eval()
    try:
        yield
    finally:
        for item, training in states.items():
            item.training = training


def _unwrap_model(model_or_agent: torch.nn.Module) -> torch.nn.Module:
    return getattr(model_or_agent, "_transfuser_model", model_or_agent)


class DiffusionDriveOPDAdapter:
    """Audited access to DiffusionDrive perception context and denoiser response."""

    REQUIRED_ATTRIBUTES = (
        "_backbone",
        "_bev_downscale",
        "_status_encoding",
        "_keyval_embedding",
        "_query_embedding",
        "_tf_decoder",
        "_bev_semantic_head",
        "_trajectory_head",
        "_agent_head",
        "bev_proj",
        "_query_splits",
    )

    def __init__(self, student: torch.nn.Module, teacher: torch.nn.Module) -> None:
        self.student = _unwrap_model(student)
        self.teacher = _unwrap_model(teacher)
        for label, model in (("student", self.student), ("teacher", self.teacher)):
            missing = [name for name in self.REQUIRED_ATTRIBUTES if not hasattr(model, name)]
            if missing:
                raise DriveOPDError(f"{label} is not supported DiffusionDrive V2: {missing}")
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self._active_query_audit: PlannerQueryAudit | None = None

    @staticmethod
    def encode_context(model: torch.nn.Module, features: TensorMap) -> DiffusionContext:
        camera = features["camera_feature"]
        status = features["status_feature"]
        batch_size = status.shape[0]
        bev_up, bev_raw, _ = model._backbone(camera, None)
        spatial_shape = tuple(int(value) for value in bev_up.shape[2:])
        raw_shape = bev_raw.shape[2:]
        bev_tokens = model._bev_downscale(bev_raw).flatten(-2, -1).permute(0, 2, 1)
        status_encoding = model._status_encoding(status)
        keyval = torch.cat([bev_tokens, status_encoding[:, None]], dim=1)
        keyval = keyval + model._keyval_embedding.weight[None, ...]
        cross = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, raw_shape[0], raw_shape[1]
        )
        cross = F.interpolate(cross, size=spatial_shape, mode="bilinear", align_corners=False)
        cross = torch.cat([cross, bev_up], dim=1)
        cross = model.bev_proj(cross.flatten(-2, -1).permute(0, 2, 1))
        cross = cross.permute(0, 2, 1).contiguous().view(
            batch_size, -1, spatial_shape[0], spatial_shape[1]
        )
        query = model._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = model._tf_decoder(query, keyval)
        ego_query, agents_query = query_out.split(model._query_splits, dim=1)
        agent_output = model._agent_head(agents_query)
        return DiffusionContext(
            ego_query=ego_query,
            agents_query=agents_query,
            bev_feature=cross,
            bev_spatial_shape=spatial_shape,
            status_encoding=status_encoding[:, None],
            bev_semantic_map=model._bev_semantic_head(bev_up),
            agent_states=agent_output["agent_states"],
            agent_logits=agent_output["agent_labels"],
        )

    @staticmethod
    def official_predictions(
        model: torch.nn.Module,
        context: DiffusionContext,
        targets: TensorMap,
    ) -> dict[str, torch.Tensor]:
        trajectory = model._trajectory_head(
            context.ego_query,
            context.agents_query,
            context.bev_feature,
            context.bev_spatial_shape,
            context.status_encoding,
            targets=targets,
            global_img=None,
        )
        return {
            "bev_semantic_map": context.bev_semantic_map,
            "agent_states": context.agent_states,
            "agent_labels": context.agent_logits,
            **trajectory,
        }

    @staticmethod
    def _query_impl(
        planner: torch.nn.Module,
        context: DiffusionContext,
        normalized_state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = planner._trajectory_head
        state = normalized_state.clamp(-1.0, 1.0)
        points = head.denorm_odo(state)
        position = __import__(
            "navsim.agents.diffusiondrive.modules.blocks", fromlist=["gen_sineembed_for_position"]
        ).gen_sineembed_for_position(points, hidden_dim=64)
        feature = head.plan_anchor_encoder(position.flatten(-2)).view(
            state.shape[0], state.shape[1], -1
        )
        time_embedding = head.time_mlp(timesteps).view(state.shape[0], 1, -1)
        regressions, logits = head.diff_decoder(
            feature,
            points,
            context.bev_feature,
            context.bev_spatial_shape,
            context.agents_query,
            context.ego_query,
            time_embedding,
            context.status_encoding,
            None,
        )
        response = head.norm_odo(regressions[-1])
        return response, logits[-1]

    def query(
        self,
        planner: torch.nn.Module,
        context: DiffusionContext,
        normalized_state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if planner is self.student:
            role = "student"
        elif planner is self.teacher:
            role = "teacher"
        else:
            raise DriveOPDError("planner query does not belong to the registered student/teacher")
        audit = getattr(self, "_active_query_audit", None)
        if audit is not None:
            audit.records.append(
                {
                    "role": role,
                    # Never read CUDA values here: the audit must not introduce
                    # a device synchronization or host copy on the hot query path.
                    "timestep_shape": tuple(int(value) for value in timesteps.shape),
                    "timestep_dtype": str(timesteps.dtype),
                    "timestep_device": str(timesteps.device),
                    "state_shape": tuple(int(value) for value in normalized_state.shape),
                }
            )
        return self._query_impl(planner, context, normalized_state, timesteps)

    @contextlib.contextmanager
    def capture_query_audit(self) -> Iterable[PlannerQueryAudit]:
        """Capture calls made by the actual DiffusionDrive adapter query method."""

        if getattr(self, "_active_query_audit", None) is not None:
            raise DriveOPDError("nested planner query audits are not supported")
        audit = PlannerQueryAudit()
        self._active_query_audit = audit
        try:
            yield audit
        finally:
            self._active_query_audit = None

    def deterministic_contexts(self, features: TensorMap) -> tuple[DiffusionContext, DiffusionContext]:
        # Aux contexts use eval behavior for both networks.  Student gradients are
        # retained, while dropout/BN mode cannot create a false T0 mismatch.
        with _temporary_eval(self.student):
            student_context = self.encode_context(self.student, features)
        with torch.no_grad(), _temporary_eval(self.teacher):
            teacher_context = self.encode_context(self.teacher, features)
        return student_context, teacher_context

    @contextlib.contextmanager
    def deterministic_planner_queries(self) -> Iterable[None]:
        """Use matched eval behavior without disabling student gradients."""

        with _temporary_eval(self.student), _temporary_eval(self.teacher):
            yield

    def student_support_states(
        self,
        context: DiffusionContext,
        config: PlanningDistillationConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> list[torch.Tensor]:
        head = self.student._trajectory_head
        detached = context.detached()
        batch_size = context.ego_query.shape[0]
        anchor = head.plan_anchor.detach().unsqueeze(0).repeat(batch_size, 1, 1, 1)
        normalized_anchor = head.norm_odo(anchor)[..., :2]
        schedule = configure_rollout_schedule(
            head.diffusion_scheduler, config, normalized_anchor.device
        )
        noise = torch.randn(
            normalized_anchor.shape,
            dtype=normalized_anchor.dtype,
            device=normalized_anchor.device,
            generator=generator,
        )
        initial_timestep = torch.full(
            (batch_size,),
            schedule.initial_noise_timestep,
            dtype=torch.long,
            device=normalized_anchor.device,
        )
        state = head.diffusion_scheduler.add_noise(
            normalized_anchor, noise, initial_timestep
        )
        states: list[torch.Tensor] = []
        with torch.no_grad(), _temporary_eval(self.student):
            for timestep_index, timestep in enumerate(schedule.query_timesteps):
                states.append(state.detach())
                time = torch.full(
                    (batch_size,), timestep, dtype=torch.long, device=state.device
                )
                response, _ = self.query(self.student, detached, state, time)
                if timestep_index + 1 < len(schedule.query_timesteps):
                    state = head.diffusion_scheduler.step(
                        model_output=response[..., :2], timestep=timestep, sample=state
                    ).prev_sample
        return states

    def student_support_planning_loss(
        self,
        student_context: DiffusionContext,
        teacher_context: DiffusionContext,
        config: PlanningDistillationConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Query and advance student support once per registered timestep.

        The queried student response is reused (detached) for the DDIM state
        transition.  Consequently OPD and exogenous-support LwF both execute
        exactly one student and one teacher denoiser query per timestep.
        """

        head = self.student._trajectory_head
        batch_size = student_context.ego_query.shape[0]
        anchor = head.plan_anchor.detach().unsqueeze(0).repeat(batch_size, 1, 1, 1)
        normalized_anchor = head.norm_odo(anchor)[..., :2]
        schedule = configure_rollout_schedule(
            head.diffusion_scheduler, config, normalized_anchor.device
        )
        noise = torch.randn(
            normalized_anchor.shape,
            dtype=normalized_anchor.dtype,
            device=normalized_anchor.device,
            generator=generator,
        )
        initial_timestep = torch.full(
            (batch_size,),
            schedule.initial_noise_timestep,
            dtype=torch.long,
            device=normalized_anchor.device,
        )
        state = head.diffusion_scheduler.add_noise(
            normalized_anchor, noise, initial_timestep
        ).detach()
        response_losses: list[torch.Tensor] = []
        mode_losses: list[torch.Tensor] = []
        agreements: list[torch.Tensor] = []
        for timestep_index, timestep in enumerate(schedule.query_timesteps):
            time = torch.full(
                (batch_size,), timestep, dtype=torch.long, device=state.device
            )
            student_response, student_mode = self.query(
                self.student, student_context, state, time
            )
            with torch.no_grad():
                teacher_response, teacher_mode = self.query(
                    self.teacher, teacher_context, state, time
                )
            response_losses.append(
                F.mse_loss(student_response, teacher_response.detach())
            )
            mode_losses.append(
                F.kl_div(
                    student_mode.log_softmax(dim=-1),
                    teacher_mode.detach().softmax(dim=-1),
                    reduction="batchmean",
                ).clamp_min(0.0)
            )
            agreements.append(
                (student_mode.argmax(-1) == teacher_mode.argmax(-1)).float().mean()
            )
            if timestep_index + 1 < len(schedule.query_timesteps):
                state = head.diffusion_scheduler.step(
                    model_output=student_response[..., :2].detach(),
                    timestep=timestep,
                    sample=state,
                ).prev_sample.detach()
        response = torch.stack(response_losses).mean()
        mode = torch.stack(mode_losses).mean()
        loss = config.response_weight * response + config.mode_kl_weight * mode
        return loss, {
            "planning_response_mse": float(response.detach().item()),
            "planning_mode_forward_kl": float(mode.detach().item()),
            "planning_mode_top1_agreement": float(torch.stack(agreements).mean().item()),
            "planning_query_states": float(len(schedule.query_timesteps)),
            "student_denoiser_queries": float(len(schedule.query_timesteps)),
            "teacher_denoiser_queries": float(len(schedule.query_timesteps)),
            **_rollout_schedule_metrics(schedule),
        }

    def exogenous_states(
        self,
        targets: TensorMap,
        config: PlanningDistillationConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> list[torch.Tensor]:
        trajectory = targets["trajectory"]
        if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
            raise DriveOPDError("LwF requires targets['trajectory'] shaped [B,T,>=2]")
        head = self.student._trajectory_head
        # NAVSIM caches trajectories as float64 on some installations while the
        # DiffusionDrive head is float32.  LwF states must live on the same
        # numerical support as the planner, just like student-rollout OPD states.
        trajectory = trajectory.to(
            device=head.plan_anchor.device,
            dtype=head.plan_anchor.dtype,
        )
        modes = int(head.plan_anchor.shape[0])
        base = trajectory[..., :2].unsqueeze(1).repeat(1, modes, 1, 1)
        base = head.norm_odo(base)
        schedule = configure_rollout_schedule(
            head.diffusion_scheduler, config, base.device
        )
        noise = torch.randn(
            base.shape, dtype=base.dtype, device=base.device, generator=generator
        )
        states = []
        for timestep in schedule.query_timesteps:
            time = torch.full(
                (base.shape[0],), timestep, dtype=torch.long, device=base.device
            )
            states.append(head.diffusion_scheduler.add_noise(base, noise, time).detach())
        return states

    def distillation_loss(
        self,
        features: TensorMap,
        targets: TensorMap,
        config: DriveOPDConfig,
        *,
        support: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if support not in {"student", "exogenous"}:
            raise DriveOPDError("support must be 'student' (OPD) or 'exogenous' (LwF)")
        student_context, teacher_context = self.deterministic_contexts(features)
        if config.lambda_perception > 0.0:
            perception, perception_metrics = perception_distillation_loss(
                student_context, teacher_context, config.perception
            )
        else:
            perception = _zero_like(student_context.agent_states)
            perception_metrics = {
                "perception_agent_loss": 0.0,
                "perception_bev_loss": 0.0,
                "matched_agents": 0.0,
                "bev_masked_cells": 0.0,
                "bev_mask_fraction": 0.0,
            }
        if config.lambda_planning > 0.0:
            with _temporary_eval(self.student), _temporary_eval(self.teacher):
                if support == "student":
                    planning, planning_metrics = self.student_support_planning_loss(
                        student_context,
                        teacher_context,
                        config.planning,
                        generator=generator,
                    )
                else:
                    states = self.exogenous_states(
                        targets, config.planning, generator=generator
                    )
                    schedule = configure_rollout_schedule(
                        self.student._trajectory_head.diffusion_scheduler,
                        config.planning,
                        states[0].device,
                    )
                    student_query = lambda state, time: self.query(
                        self.student, student_context, state, time
                    )
                    teacher_query = lambda state, time: self.query(
                        self.teacher, teacher_context, state, time
                    )
                    planning, planning_metrics = planning_distillation_loss(
                        student_query,
                        teacher_query,
                        states,
                        config.planning.rollout_timesteps,
                        config.planning,
                    )
                    planning_metrics.update(
                        {
                            "student_denoiser_queries": float(len(states)),
                            "teacher_denoiser_queries": float(len(states)),
                            **_rollout_schedule_metrics(schedule),
                        }
                    )
        else:
            planning = _zero_like(student_context.ego_query)
            planning_metrics = {
                "planning_response_mse": 0.0,
                "planning_mode_forward_kl": 0.0,
                "planning_mode_top1_agreement": 0.0,
                "planning_query_states": 0.0,
                "student_denoiser_queries": 0.0,
                "teacher_denoiser_queries": 0.0,
            }
        total = config.lambda_perception * perception + config.lambda_planning * planning
        return total, {
            "distillation_total": float(total.detach().item()),
            "distillation_support_student": float(support == "student"),
            **perception_metrics,
            **planning_metrics,
        }

    def context_swap_responses(
        self,
        features: TensorMap,
        state: torch.Tensor,
        timestep: int,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return the four perception/planner swaps needed for causal audit."""

        student_context, teacher_context = self.deterministic_contexts(features)
        time = torch.full(
            (state.shape[0],), timestep, dtype=torch.long, device=state.device
        )
        with _temporary_eval(self.student), _temporary_eval(self.teacher):
            return {
                "teacher_perception_teacher_planner": self.query(
                    self.teacher, teacher_context, state, time
                ),
                "student_perception_teacher_planner": self.query(
                    self.teacher, student_context, state, time
                ),
                "teacher_perception_student_planner": self.query(
                    self.student, teacher_context, state, time
                ),
                "student_perception_student_planner": self.query(
                    self.student, student_context, state, time
                ),
            }


@torch.no_grad()
def update_ema_teacher_(
    teacher: torch.nn.Module, student: torch.nn.Module, momentum: float = 0.99
) -> None:
    """EMA-update parameters and floating buffers; copy discrete buffers."""

    if not 0.0 <= momentum < 1.0:
        raise DriveOPDError(f"EMA momentum must be in [0,1), got {momentum}")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if teacher_state.keys() != student_state.keys():
        raise DriveOPDError("EMA teacher/student state dictionaries differ")
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach().to(teacher_value.device)
        if teacher_value.is_floating_point():
            teacher_value.mul_(momentum).add_(student_value, alpha=1.0 - momentum)
        else:
            teacher_value.copy_(student_value)


def clone_frozen_teacher(student: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher

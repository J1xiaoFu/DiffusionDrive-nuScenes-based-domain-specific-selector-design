"""Native distillation losses for the nuScenes SparseDrive implementation.

The official SparseDrive forward remains the mainline loss.  This module only
consumes tensors already produced by that forward and one frozen-teacher
forward, then queries both planning denoisers at identical states.  LwF and
OPD differ only in whether those states are GT-anchor noising or the student's
detached rollout.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from selector_bench.continual.drive_opd import bernoulli_mode_forward_kl


def fixed_arm_loss_switches(arm: str) -> tuple[bool, bool, bool]:
    """Return perception, planning and mode-KL switches for fixed CL arms."""
    switches = {
        "lwf": (True, True, True),
        "fixed_opd": (True, True, True),
        "perception_only": (True, False, False),
        "planning_only": (False, True, False),
    }
    try:
        return switches[str(arm)]
    except KeyError as error:
        raise ValueError(f"unsupported fixed nuScenes Drive-CL arm: {arm}") from error


@contextlib.contextmanager
def temporary_eval(module: torch.nn.Module):
    modes = {item: bool(item.training) for item in module.modules()}
    module.eval()
    try:
        yield
    finally:
        for item, mode in modes.items():
            item.training = mode


def _query(
    head: torch.nn.Module,
    context: Mapping[str, Any],
    latent: torch.Tensor,
    timestep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from projects.mmdet3d_plugin.models.attention import (
        gen_sineembed_for_position,
    )

    batch_size = int(context["batch_size"])
    x_boxes = latent.clamp(-1.0, 1.0)
    noisy_points = head.denormalize_ego_fut_trajs(x_boxes)
    position = gen_sineembed_for_position(
        noisy_points, hidden_dim=128
    ).flatten(-2)
    feature = head.plan_pos_encoder(position).view(
        batch_size, head.ego_fut_mode, -1
    )
    timesteps = torch.full(
        (batch_size * head.ego_fut_mode,),
        int(timestep),
        dtype=torch.long,
        device=latent.device,
    )
    time_embedding = head.time_mlp(timesteps).view(
        batch_size, head.ego_fut_mode, -1
    )
    regression = noisy_points
    classification = None
    for index, operation in enumerate(head.diff_operation_order):
        layer = head.diff_layers[index]
        if layer is None:
            continue
        if operation == "traj_pooler":
            if regression.ndim != 3:
                regression = regression[
                    :, :, -head.ego_fut_mode :
                ].flatten(0, 2)
            feature = layer(
                feature,
                regression,
                context["metas"],
                context["feature_maps"],
                modal_num=head.ego_fut_mode,
            )
        elif operation == "self_attn":
            feature = layer(feature, feature, feature)
        elif operation == "modulation":
            feature = layer(feature, time_embedding, global_cond=None)
        elif operation == "agent_cross_gnn":
            feature = head.diff_graph_model(
                index,
                feature,
                context["instance_feature_selected"],
                context["instance_feature_selected"],
                query_pos=context["repeat_ego_anchor_embed"],
                key_pos=context["anchor_embed_selected"],
            )
        elif operation == "map_cross_gnn":
            feature = layer(
                feature,
                context["map_instance_feature_selected"],
                context["map_instance_feature_selected"],
                query_pos=context["repeat_ego_anchor_embed"],
                key_pos=context["map_anchor_embed_selected"],
            )
        elif operation == "anchor_cross_gnn":
            feature = layer(
                feature,
                key=context["cmd_plan_nav_query"],
                value=context["cmd_plan_nav_query"],
            )
        elif operation in ("norm", "ffn"):
            feature = layer(feature)
        elif operation == "diff_refine":
            regression, classification = layer(feature)
    if classification is None:
        raise RuntimeError("SparseDrive CL denoiser produced no mode logits")
    if regression.ndim != 3:
        regression = regression[:, :, -head.ego_fut_mode :].flatten(0, 2)
    model_output = head.normalize_ego_fut_trajs(regression)
    transition = head.diffusion_scheduler.step(
        model_output=model_output,
        timestep=int(timestep),
        sample=latent,
    ).prev_sample
    mode_logits = classification[..., -head.ego_fut_mode :].reshape(
        batch_size, head.ego_fut_mode
    )
    return transition, mode_logits


def planning_distillation(
    student_head: torch.nn.Module,
    teacher_head: torch.nn.Module,
    student_context: Mapping[str, Any],
    teacher_context: Mapping[str, Any],
    *,
    support: str,
    mode_weight: float,
    iteration: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distil path response and independent sigmoid-anchor scores."""

    if support not in {"student_rollout", "shared_exogenous"}:
        raise ValueError("unknown SparseDrive distillation support")
    batch_size = int(student_context["batch_size"])
    clean = student_head.normalize_ego_fut_trajs(
        student_context["tgt_cmd_plan_anchor"]
    )
    device = clean.device
    generator = torch.Generator(device=device)
    generator.manual_seed(170003 + int(iteration) * 1009 + int(device.index or 0))
    noise = torch.randn(
        clean.shape, dtype=clean.dtype, device=device, generator=generator
    )
    query_times = (10, 0)
    student_head.diffusion_scheduler.set_timesteps(100, device)
    teacher_head.diffusion_scheduler.set_timesteps(100, device)
    initial_time = torch.full(
        (batch_size,), 10, dtype=torch.long, device=device
    )
    latent = student_head.diffusion_scheduler.add_noise(
        original_samples=clean, noise=noise, timesteps=initial_time
    ).reshape(
        batch_size * student_head.ego_fut_mode,
        student_head.ego_fut_ts,
        2,
    ).detach()
    response_losses = []
    mode_losses = []
    agreements = []
    with temporary_eval(student_head), temporary_eval(teacher_head):
        for timestep in query_times:
            if support == "shared_exogenous":
                external_time = torch.full(
                    (batch_size,),
                    timestep,
                    dtype=torch.long,
                    device=device,
                )
                latent = student_head.diffusion_scheduler.add_noise(
                    original_samples=clean,
                    noise=noise,
                    timesteps=external_time,
                ).reshape(
                    batch_size * student_head.ego_fut_mode,
                    student_head.ego_fut_ts,
                    2,
                ).detach()
            student_transition, student_logits = _query(
                student_head, student_context, latent, timestep
            )
            with torch.no_grad():
                teacher_transition, teacher_logits = _query(
                    teacher_head, teacher_context, latent, timestep
                )
            response_losses.append(
                F.mse_loss(student_transition, teacher_transition.detach())
            )
            mode_losses.append(
                bernoulli_mode_forward_kl(teacher_logits, student_logits)
            )
            agreements.append(
                (student_logits.argmax(-1) == teacher_logits.argmax(-1))
                .float()
                .mean()
            )
            if support == "student_rollout":
                latent = student_transition.detach()
    response = torch.stack(response_losses).mean()
    mode = torch.stack(mode_losses).mean()
    total = response + float(mode_weight) * mode
    return total, {
        "drive_cl_planning_response_mse": response.detach(),
        "drive_cl_mode_bernoulli_forward_kl": mode.detach(),
        # Argmax is retained only as selected-anchor agreement, not as a
        # categorical-distribution metric.
        "drive_cl_mode_selected_anchor_agreement": (
            torch.stack(agreements).mean().detach()
        ),
        "drive_cl_student_support": response.new_tensor(
            float(support == "student_rollout")
        ),
        "drive_cl_student_denoiser_queries": response.new_tensor(2.0),
        "drive_cl_teacher_denoiser_queries": response.new_tensor(2.0),
    }


def _matched_structured_loss(
    student_boxes: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_boxes: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, int]:
    losses = []
    matched = 0
    for batch_index in range(student_boxes.shape[0]):
        teacher_score = teacher_logits[batch_index].detach().sigmoid().amax(-1)
        count = min(int(topk), int(teacher_score.numel()))
        teacher_index = teacher_score.topk(count).indices
        student_score = student_logits[batch_index].detach().sigmoid().amax(-1)
        student_index = student_score.topk(min(count * 2, student_score.numel())).indices
        teacher_xy = teacher_boxes[batch_index, teacher_index, :2].detach().float()
        student_xy = student_boxes[batch_index, student_index, :2].detach().float()
        rows, columns = linear_sum_assignment(
            torch.cdist(student_xy, teacher_xy, p=1).cpu().numpy()
        )
        selected_student = student_index[
            torch.as_tensor(rows, device=student_boxes.device)
        ]
        selected_teacher = teacher_index[
            torch.as_tensor(columns, device=student_boxes.device)
        ]
        student_box = student_boxes[batch_index, selected_student]
        teacher_box = teacher_boxes[batch_index, selected_teacher].detach()
        box_dims = min(int(student_box.shape[-1]), int(teacher_box.shape[-1]))
        geometry = F.smooth_l1_loss(
            student_box[..., :box_dims], teacher_box[..., :box_dims]
        )
        teacher_probability = teacher_logits[
            batch_index, selected_teacher
        ].detach().float().sigmoid()
        classification = F.binary_cross_entropy_with_logits(
            student_logits[batch_index, selected_student].float(),
            teacher_probability,
        )
        losses.append(geometry + classification)
        matched += len(rows)
    if not losses:
        return student_boxes.sum() * 0.0, 0
    return torch.stack(losses).mean(), matched


def perception_distillation(
    student_outputs: tuple[Any, Any, Any, Any],
    teacher_outputs: tuple[Any, Any, Any, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_det, student_map, _, _ = student_outputs
    teacher_det, teacher_map, _, _ = teacher_outputs
    det_loss, det_matches = _matched_structured_loss(
        student_det["prediction"][-1],
        student_det["classification"][-1],
        teacher_det["prediction"][-1],
        teacher_det["classification"][-1],
        topk=50,
    )
    map_loss, map_matches = _matched_structured_loss(
        student_map["prediction"][-1],
        student_map["classification"][-1],
        teacher_map["prediction"][-1],
        teacher_map["classification"][-1],
        topk=10,
    )
    total = det_loss + map_loss
    return total, {
        "drive_cl_perception_det": det_loss.detach(),
        "drive_cl_perception_map": map_loss.detach(),
        "drive_cl_perception_det_matches": total.new_tensor(float(det_matches)),
        "drive_cl_perception_map_matches": total.new_tensor(float(map_matches)),
    }


def compute_sparse_drive_distillation(
    student_head: torch.nn.Module,
    teacher_head: torch.nn.Module,
    student_outputs: tuple[Any, Any, Any, Any],
    teacher_outputs: tuple[Any, Any, Any, Any],
    student_context: Mapping[str, Any],
    teacher_context: Mapping[str, Any],
    *,
    arm: str,
    mode_policy: str,
    iteration: int,
    perception_active: bool,
    planning_active: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reference = student_outputs[0]["prediction"][-1]
    total = reference.sum() * 0.0
    diagnostics: dict[str, torch.Tensor] = {}
    if perception_active:
        perception, values = perception_distillation(
            student_outputs, teacher_outputs
        )
        total = total + perception
        diagnostics.update(values)
    if planning_active:
        support = "shared_exogenous" if arm == "lwf" else "student_rollout"
        planning, values = planning_distillation(
            student_head.motion_plan_head,
            teacher_head.motion_plan_head,
            student_context,
            teacher_context,
            support=support,
            mode_weight=0.0 if mode_policy == "off" else 1.0,
            iteration=iteration,
        )
        total = total + planning
        diagnostics.update(values)
    diagnostics["drive_cl_perception_active"] = total.new_tensor(
        float(perception_active)
    )
    diagnostics["drive_cl_planning_active"] = total.new_tensor(
        float(planning_active)
    )
    diagnostics["drive_cl_mode_active"] = total.new_tensor(
        float(planning_active and mode_policy != "off")
    )
    return total, diagnostics

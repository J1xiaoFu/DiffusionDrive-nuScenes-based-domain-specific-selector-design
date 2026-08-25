"""nuScenes/SparseDrive adapter for shared driving-functional losses.

This adapter is intentionally a small tensor boundary rather than another
model layer.  It mirrors the real DiffusionDrive training representation:

* ego planning regression is command-selected, six-mode, six-step ``xy``
  *increments* and is integrated with ``cumsum``;
* detection anchors are encoded ``x,y,z,log(x-size),log(y-size),log(h),
  sin(yaw),cos(yaw),vx,vy,...`` (see ``core/box3d.py``);
* agent motion and ego/agent ground truth are also future increments.

The shared loss convention is current ego-local ``x=right, y=forward`` with
mathematical heading measured counter-clockwise from +x.  Consequently the
stationary/current ego heading is pi/2.  nuScenes CAN-bus status is supplied
in vehicle ``forward,left`` coordinates and is converted to ``right,forward``.

Agent predictions or ground truth used by ego safety are detached here, in
addition to the defensive detach in the core.  Drivable-area support is
deliberately deferred: no raster/map surrogate is silently substituted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from selector_bench.continual.driving_functional_loss import DrivingFunctionalLossError


NUSCENES_HORIZON_STEPS = 6
NUSCENES_DT_SECONDS = 0.5
NUSCENES_COMMANDS = 3
NUSCENES_PLANNING_MODES = 6
NUSCENES_EGO_EXTENT_LW = (4.08, 1.73)

# SparseDrive names index 3 ``W`` and index 4 ``L``, but the nuScenes
# converter explicitly emits x-size,y-size = physical length,width and the
# official ego decoder assigns [4.08,1.73] in that same index order.  These
# semantic names avoid propagating the misleading symbol names.
ANCHOR_X = 0
ANCHOR_Y = 1
ANCHOR_LOG_LENGTH = 3
ANCHOR_LOG_WIDTH = 4
ANCHOR_SIN_YAW = 6
ANCHOR_COS_YAW = 7


@dataclass(frozen=True)
class NuScenesFunctionalTensors:
    """The exact keyword payload accepted by ``driving_functional_losses``."""

    ego_modes_xyh: torch.Tensor
    mode_logits: torch.Tensor
    ego_gt_xyh: torch.Tensor
    ego_v0_xy: torch.Tensor
    ego_a0_xy: torch.Tensor
    ego_extent_lw: torch.Tensor
    agent_boxes_xyhlw: torch.Tensor
    agent_velocity_xy: torch.Tensor
    agent_valid: torch.Tensor
    route_xy: torch.Tensor
    route_valid: torch.Tensor

    def as_core_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "ego_modes_xyh": self.ego_modes_xyh,
            "mode_logits": self.mode_logits,
            "ego_gt_xyh": self.ego_gt_xyh,
            "ego_v0_xy": self.ego_v0_xy,
            "ego_a0_xy": self.ego_a0_xy,
            "ego_extent_lw": self.ego_extent_lw,
            "agent_boxes_xyhlw": self.agent_boxes_xyhlw,
            "agent_velocity_xy": self.agent_velocity_xy,
            "agent_valid": self.agent_valid,
            "route_xy": self.route_xy,
            "route_valid": self.route_valid,
        }


@dataclass(frozen=True)
class NuScenesAgentGeometry:
    """Detached dynamic-obstacle geometry in shared loss convention."""

    boxes_xyhlw: torch.Tensor
    velocity_xy: torch.Tensor
    valid: torch.Tensor


def _require_float(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise DrivingFunctionalLossError(f"{name} must be a floating torch tensor")


def _validate_positive_extents(name: str, extent_lw: torch.Tensor) -> None:
    if extent_lw.shape[-1] != 2:
        raise DrivingFunctionalLossError(f"{name} must end in length,width")
    if not bool(torch.isfinite(extent_lw).all()) or not bool((extent_lw > 0).all()):
        raise DrivingFunctionalLossError(f"{name} must contain finite positive length,width")


def xy_to_xyh(
    future_xy: torch.Tensor,
    *,
    initial_xy: torch.Tensor | None = None,
    initial_heading_rad: torch.Tensor | float = math.pi / 2.0,
    stationary_epsilon_m: float = 1e-6,
) -> torch.Tensor:
    """Add differentiable finite-difference headings to absolute future xy.

    A stationary step inherits its preceding heading.  The branch is used
    only to avoid undefined ``atan2(0,0)``; moving trajectories retain the
    ordinary differentiable atan2 derivative.
    """

    _require_float("future_xy", future_xy)
    if future_xy.ndim < 3 or future_xy.shape[-1] != 2:
        raise DrivingFunctionalLossError("future_xy must end in [T,2]")
    if stationary_epsilon_m <= 0.0:
        raise DrivingFunctionalLossError("stationary_epsilon_m must be positive")

    leading = future_xy.shape[:-2]
    if initial_xy is None:
        previous_xy = torch.zeros(
            *leading, 2, dtype=future_xy.dtype, device=future_xy.device
        )
    else:
        if initial_xy.shape != (*leading, 2):
            raise DrivingFunctionalLossError(
                "initial_xy must match future_xy leading dimensions"
            )
        previous_xy = initial_xy.to(dtype=future_xy.dtype, device=future_xy.device)

    if torch.is_tensor(initial_heading_rad):
        if initial_heading_rad.shape != leading:
            raise DrivingFunctionalLossError(
                "initial heading must match future_xy leading dimensions"
            )
        previous_heading = initial_heading_rad.to(
            dtype=future_xy.dtype, device=future_xy.device
        )
    else:
        previous_heading = future_xy.new_full(leading, float(initial_heading_rad))

    headings: list[torch.Tensor] = []
    for step in range(future_xy.shape[-2]):
        delta = future_xy[..., step, :] - previous_xy
        moving_heading = torch.atan2(delta[..., 1], delta[..., 0])
        moving = torch.linalg.vector_norm(delta, dim=-1) > stationary_epsilon_m
        heading = torch.where(moving, moving_heading, previous_heading)
        headings.append(heading)
        previous_xy = future_xy[..., step, :]
        previous_heading = heading
    return torch.cat((future_xy, torch.stack(headings, dim=-1).unsqueeze(-1)), dim=-1)


def _command_indices(command: torch.Tensor, *, batch: int) -> torch.Tensor:
    command = command.detach()
    if command.ndim == 2 and command.shape == (batch, NUSCENES_COMMANDS):
        # This selects a frozen data command, not an ego planning mode.
        return command.argmax(dim=-1).long()
    if command.ndim == 1 and command.shape[0] == batch:
        indices = command.long()
        if not torch.equal(indices.to(command.dtype), command):
            raise DrivingFunctionalLossError("command indices must be integral")
        if bool(((indices < 0) | (indices >= NUSCENES_COMMANDS)).any()):
            raise DrivingFunctionalLossError("command indices must be in [0,2]")
        return indices
    raise DrivingFunctionalLossError("command must be [B,3] one-hot/score or [B] indices")


def _reshape_command_modes(
    value: torch.Tensor,
    *,
    name: str,
    trailing_shape: tuple[int, ...],
) -> tuple[torch.Tensor, bool]:
    """Return [B,3,6,...] or already-selected [B,6,...]."""

    if value.ndim >= 2 and value.shape[1] == 1:
        value = value.squeeze(1)
    selected_dimensions = 2 + len(trailing_shape)
    command_dimensions = 3 + len(trailing_shape)
    if value.ndim == selected_dimensions and value.shape[1] == NUSCENES_PLANNING_MODES:
        if tuple(value.shape[2:]) != trailing_shape:
            raise DrivingFunctionalLossError(f"{name} trailing shape must be {trailing_shape}")
        return value, True
    if value.ndim == selected_dimensions and value.shape[1] == (
        NUSCENES_COMMANDS * NUSCENES_PLANNING_MODES
    ):
        value = value.reshape(
            value.shape[0], NUSCENES_COMMANDS, NUSCENES_PLANNING_MODES, *trailing_shape
        )
    if value.ndim != command_dimensions or value.shape[1:3] != (
        NUSCENES_COMMANDS,
        NUSCENES_PLANNING_MODES,
    ):
        raise DrivingFunctionalLossError(
            f"{name} must be command-selected [B,6,...] or [B,3,6,...]/[B,18,...]"
        )
    if tuple(value.shape[3:]) != trailing_shape:
        raise DrivingFunctionalLossError(f"{name} trailing shape must be {trailing_shape}")
    return value, False


def select_nuscenes_planning_modes(
    planning_deltas_xy: torch.Tensor,
    planning_logits: torch.Tensor,
    command: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the data command and integrate six SparseDrive ego modes.

    Accepted real-training layouts include ``[B,1,18,6,2]`` and
    ``[B,3,6,6,2]`` (plus already-selected ``[B,6,6,2]``).  The returned
    trajectory is absolute ``[B,6,6,3]`` and remains connected to planner
    regression gradients.  The returned score is the original independent
    Bernoulli logit ``[B,6]``.
    """

    _require_float("planning_deltas_xy", planning_deltas_xy)
    _require_float("planning_logits", planning_logits)
    deltas, deltas_selected = _reshape_command_modes(
        planning_deltas_xy,
        name="planning_deltas_xy",
        trailing_shape=(NUSCENES_HORIZON_STEPS, 2),
    )
    logits, logits_selected = _reshape_command_modes(
        planning_logits,
        name="planning_logits",
        trailing_shape=(),
    )
    if deltas.shape[0] != logits.shape[0] or deltas_selected != logits_selected:
        raise DrivingFunctionalLossError("planning predictions and logits use different layouts")

    if deltas_selected:
        selected_deltas = deltas
        selected_logits = logits
    else:
        command_index = _command_indices(command, batch=deltas.shape[0]).to(deltas.device)
        batch_index = torch.arange(deltas.shape[0], device=deltas.device)
        selected_deltas = deltas[batch_index, command_index]
        selected_logits = logits[batch_index, command_index]

    absolute_xy = selected_deltas.cumsum(dim=-2)
    return xy_to_xyh(absolute_xy), selected_logits


def nuscenes_status_to_planning_state(
    ego_status: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CAN ``forward,left`` acceleration/velocity to ``right,forward``.

    The converter stores accel at indices 0:3 and velocity at 6:9.  This is
    consistent with SparseDrive assigning status index 6 to decoded box VY.
    """

    _require_float("ego_status", ego_status)
    if ego_status.ndim != 2 or ego_status.shape[1] < 8:
        raise DrivingFunctionalLossError("ego_status must be [B,>=8]")
    acceleration = torch.stack((-ego_status[:, 1], ego_status[:, 0]), dim=-1)
    velocity = torch.stack((-ego_status[:, 7], ego_status[:, 6]), dim=-1)
    return velocity.detach(), acceleration.detach()


def build_nuscenes_ego_gt_and_route(
    ego_gt_deltas_xy: torch.Tensor,
    ego_gt_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate six ego GT increments and use their ordered path as route."""

    _require_float("ego_gt_deltas_xy", ego_gt_deltas_xy)
    if ego_gt_deltas_xy.ndim != 3 or ego_gt_deltas_xy.shape[1:] != (
        NUSCENES_HORIZON_STEPS,
        2,
    ):
        raise DrivingFunctionalLossError("ego GT increments must be [B,6,2]")
    if ego_gt_valid.shape != ego_gt_deltas_xy.shape[:2]:
        raise DrivingFunctionalLossError("ego_gt_valid must be [B,6]")
    absolute_xy = ego_gt_deltas_xy.detach().cumsum(dim=-2)
    gt_xyh = xy_to_xyh(absolute_xy).detach()
    origin = absolute_xy.new_zeros((absolute_xy.shape[0], 1, 2))
    route_xy = torch.cat((origin, absolute_xy), dim=1).detach()
    origin_valid = torch.ones(
        (absolute_xy.shape[0], 1), dtype=torch.bool, device=absolute_xy.device
    )
    route_valid = torch.cat((origin_valid, ego_gt_valid.detach().bool()), dim=1)
    return gt_xyh, route_xy, route_valid


def _normalised_sigmoid_confidence(logits: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    confidence = logits.detach().sigmoid()
    return confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def build_nuscenes_predicted_agents(
    detection_anchors: torch.Tensor,
    motion_deltas_xy: torch.Tensor,
    motion_logits: torch.Tensor,
    agent_valid: torch.Tensor,
    *,
    dt_seconds: float = NUSCENES_DT_SECONDS,
) -> NuScenesAgentGeometry:
    """Build detached agent boxes from SparseDrive prediction tensors.

    Motion modes are reduced with detached, normalised sigmoid confidences;
    this avoids an argmax path while preserving the independent-score
    semantics.  It is a safety obstacle estimate, never a target for agent
    prediction gradients.
    """

    _require_float("detection_anchors", detection_anchors)
    _require_float("motion_deltas_xy", motion_deltas_xy)
    _require_float("motion_logits", motion_logits)
    if detection_anchors.ndim != 3 or detection_anchors.shape[-1] < 8:
        raise DrivingFunctionalLossError("detection anchors must be [B,A,>=8]")
    if motion_deltas_xy.ndim != 5 or motion_deltas_xy.shape[-2:] != (
        NUSCENES_HORIZON_STEPS,
        2,
    ):
        raise DrivingFunctionalLossError("motion deltas must be [B,A,K,6,2]")
    batch, agents, modes = motion_deltas_xy.shape[:3]
    if detection_anchors.shape[:2] != (batch, agents):
        raise DrivingFunctionalLossError("detection and motion agent axes differ")
    if motion_logits.shape != (batch, agents, modes):
        raise DrivingFunctionalLossError("motion logits must be [B,A,K]")
    if dt_seconds <= 0.0:
        raise DrivingFunctionalLossError("dt_seconds must be positive")

    anchors = detection_anchors.detach()
    motion = motion_deltas_xy.detach()
    weight = _normalised_sigmoid_confidence(motion_logits).unsqueeze(-1).unsqueeze(-1)
    mode_xy = motion.cumsum(dim=-2) + anchors[..., None, None, :2]
    future_xy = (weight * mode_xy).sum(dim=2)
    current_xy = anchors[..., :2]
    current_yaw = torch.atan2(
        anchors[..., ANCHOR_SIN_YAW], anchors[..., ANCHOR_COS_YAW]
    )
    future_xyh = xy_to_xyh(
        future_xy,
        initial_xy=current_xy,
        initial_heading_rad=current_yaw,
    )
    # Encoded x-size,y-size is already physical length,width.
    extent_lw = torch.stack(
        (
            anchors[..., ANCHOR_LOG_LENGTH].exp(),
            anchors[..., ANCHOR_LOG_WIDTH].exp(),
        ),
        dim=-1,
    )
    _validate_positive_extents("decoded agent extent", extent_lw)
    boxes = torch.cat(
        (
            future_xyh,
            extent_lw[:, :, None].expand(-1, -1, NUSCENES_HORIZON_STEPS, -1),
        ),
        dim=-1,
    ).permute(0, 2, 1, 3)

    velocity = torch.empty_like(future_xy)
    velocity[:, :, 0] = (future_xy[:, :, 0] - current_xy) / dt_seconds
    velocity[:, :, 1:] = (future_xy[:, :, 1:] - future_xy[:, :, :-1]) / dt_seconds
    velocity = velocity.permute(0, 2, 1, 3)
    if agent_valid.ndim == 2 and agent_valid.shape == (batch, agents):
        valid = agent_valid[:, None].expand(-1, NUSCENES_HORIZON_STEPS, -1)
    elif agent_valid.shape == (batch, NUSCENES_HORIZON_STEPS, agents):
        valid = agent_valid
    else:
        raise DrivingFunctionalLossError("agent_valid must be [B,A] or [B,6,A]")
    return NuScenesAgentGeometry(boxes.detach(), velocity.detach(), valid.detach().bool())


def build_nuscenes_gt_agents(
    current_boxes_xyzlwhr: torch.Tensor,
    future_deltas_xy: torch.Tensor,
    future_valid: torch.Tensor,
    *,
    dt_seconds: float = NUSCENES_DT_SECONDS,
) -> NuScenesAgentGeometry:
    """Build detached geometry from decoded LiDAR boxes and GT increments.

    ``current_boxes_xyzlwhr`` follows the actual decoded nuScenes box tensor
    order ``x,y,z,length,width,height,yaw,...``.  Future GT is cumulative
    displacement relative to each current box.
    """

    _require_float("current_boxes_xyzlwhr", current_boxes_xyzlwhr)
    _require_float("future_deltas_xy", future_deltas_xy)
    if current_boxes_xyzlwhr.ndim != 3 or current_boxes_xyzlwhr.shape[-1] < 7:
        raise DrivingFunctionalLossError("decoded current boxes must be [B,A,>=7]")
    if future_deltas_xy.ndim != 4 or future_deltas_xy.shape[-2:] != (
        NUSCENES_HORIZON_STEPS,
        2,
    ):
        raise DrivingFunctionalLossError("agent GT increments must be [B,A,6,2]")
    batch, agents = current_boxes_xyzlwhr.shape[:2]
    if future_deltas_xy.shape[:2] != (batch, agents):
        raise DrivingFunctionalLossError("current boxes and future GT axes differ")
    if future_valid.shape != (batch, agents, NUSCENES_HORIZON_STEPS):
        raise DrivingFunctionalLossError("future_valid must be [B,A,6]")
    if dt_seconds <= 0.0:
        raise DrivingFunctionalLossError("dt_seconds must be positive")

    current = current_boxes_xyzlwhr.detach()
    future_xy = future_deltas_xy.detach().cumsum(dim=-2) + current[..., None, :2]
    future_xyh = xy_to_xyh(
        future_xy,
        initial_xy=current[..., :2],
        initial_heading_rad=current[..., 6],
    )
    extent_lw = current[..., 3:5]
    _validate_positive_extents("decoded GT agent extent", extent_lw)
    boxes = torch.cat(
        (
            future_xyh,
            extent_lw[:, :, None].expand(-1, -1, NUSCENES_HORIZON_STEPS, -1),
        ),
        dim=-1,
    ).permute(0, 2, 1, 3)
    velocity = future_deltas_xy.detach() / dt_seconds
    return NuScenesAgentGeometry(
        boxes.detach(),
        velocity.permute(0, 2, 1, 3).detach(),
        future_valid.permute(0, 2, 1).detach().bool(),
    )


def pad_nuscenes_gt_agents(
    current_boxes: Sequence[torch.Tensor | object],
    future_deltas_xy: Sequence[torch.Tensor],
    future_valid: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad the variable-length list emitted by the real training collate."""

    if not current_boxes or not (
        len(current_boxes) == len(future_deltas_xy) == len(future_valid)
    ):
        raise DrivingFunctionalLossError("agent GT lists must have the same non-zero batch length")
    box_tensors: list[torch.Tensor] = []
    for boxes in current_boxes:
        tensor = boxes if torch.is_tensor(boxes) else getattr(boxes, "tensor", None)
        if tensor is None or tensor.ndim != 2 or tensor.shape[-1] < 7:
            raise DrivingFunctionalLossError("each current box item must expose tensor [A,>=7]")
        box_tensors.append(tensor)
    maximum = max(item.shape[0] for item in box_tensors)
    batch = len(box_tensors)
    prototype = box_tensors[0]
    box_width = max(item.shape[-1] for item in box_tensors)
    padded_boxes = prototype.new_zeros((batch, maximum, box_width))
    padded_future = prototype.new_zeros(
        (batch, maximum, NUSCENES_HORIZON_STEPS, 2)
    )
    padded_valid = torch.zeros(
        (batch, maximum, NUSCENES_HORIZON_STEPS),
        dtype=torch.bool,
        device=prototype.device,
    )
    for batch_index, (boxes, future, valid) in enumerate(
        zip(box_tensors, future_deltas_xy, future_valid)
    ):
        agents = boxes.shape[0]
        if future.shape != (agents, NUSCENES_HORIZON_STEPS, 2):
            raise DrivingFunctionalLossError("each future GT item must be [A,6,2]")
        if valid.shape != (agents, NUSCENES_HORIZON_STEPS):
            raise DrivingFunctionalLossError("each future mask item must be [A,6]")
        padded_boxes[batch_index, :agents, : boxes.shape[-1]] = boxes
        padded_future[batch_index, :agents] = future.to(prototype)
        padded_valid[batch_index, :agents] = valid.to(device=prototype.device).bool()
    return padded_boxes, padded_future, padded_valid


def build_nuscenes_functional_tensors(
    *,
    planning_deltas_xy: torch.Tensor,
    planning_logits: torch.Tensor,
    command: torch.Tensor,
    ego_gt_deltas_xy: torch.Tensor,
    ego_gt_valid: torch.Tensor,
    ego_status: torch.Tensor,
    agents: NuScenesAgentGeometry,
    ego_extent_lw: torch.Tensor | None = None,
) -> NuScenesFunctionalTensors:
    """Assemble one real-training adapter payload for the shared core."""

    ego_modes_xyh, mode_logits = select_nuscenes_planning_modes(
        planning_deltas_xy, planning_logits, command
    )
    ego_gt_xyh, route_xy, route_valid = build_nuscenes_ego_gt_and_route(
        ego_gt_deltas_xy, ego_gt_valid
    )
    velocity, acceleration = nuscenes_status_to_planning_state(ego_status)
    if ego_extent_lw is None:
        ego_extent_lw = ego_modes_xyh.new_tensor(NUSCENES_EGO_EXTENT_LW)
    _validate_positive_extents("ego_extent_lw", ego_extent_lw)
    if agents.boxes_xyhlw.shape[:2] != (
        ego_modes_xyh.shape[0],
        NUSCENES_HORIZON_STEPS,
    ):
        raise DrivingFunctionalLossError("agent geometry batch/horizon differs from ego")
    return NuScenesFunctionalTensors(
        ego_modes_xyh=ego_modes_xyh,
        mode_logits=mode_logits,
        ego_gt_xyh=ego_gt_xyh,
        ego_v0_xy=velocity,
        ego_a0_xy=acceleration,
        ego_extent_lw=ego_extent_lw,
        agent_boxes_xyhlw=agents.boxes_xyhlw,
        agent_velocity_xy=agents.velocity_xy,
        agent_valid=agents.valid,
        route_xy=route_xy,
        route_valid=route_valid,
    )


__all__ = [
    "NUSCENES_DT_SECONDS",
    "NUSCENES_EGO_EXTENT_LW",
    "NUSCENES_HORIZON_STEPS",
    "NUSCENES_PLANNING_MODES",
    "NuScenesAgentGeometry",
    "NuScenesFunctionalTensors",
    "build_nuscenes_ego_gt_and_route",
    "build_nuscenes_functional_tensors",
    "build_nuscenes_gt_agents",
    "build_nuscenes_predicted_agents",
    "nuscenes_status_to_planning_state",
    "pad_nuscenes_gt_agents",
    "select_nuscenes_planning_modes",
    "xy_to_xyh",
]

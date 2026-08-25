"""Differentiable, driving-native function losses shared across datasets.

The core deliberately consumes geometry in one convention: absolute future
poses in the current ego-local frame, with ``x``/``y`` in metres, heading in
radians, and box extents ordered as length/width.  Dataset-specific adapters
own coordinate conversion, delta integration, padding, and shape checks.

Mode scores are independent Bernoulli logits.  They are never passed through
a categorical softmax or categorical KL.  Detached sigmoid confidences are
normalised only to form a numerical reduction over already-computed per-mode
function costs; the all-mode mean is returned alongside that reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


class DrivingFunctionalLossError(ValueError):
    """Raised when a functional-loss tensor violates the frozen interface."""


@dataclass(frozen=True)
class DrivingFunctionalLossConfig:
    """Numerical constants for the six-step, 0.5-second loss surface.

    The temperatures smooth otherwise non-differentiable geometric extrema;
    they are not method weights.  This module intentionally does not combine
    the returned components into a single scientifically weighted objective.
    """

    horizon_steps: int = 6
    dt_seconds: float = 0.5
    sat_smooth_abs_epsilon: float = 1e-6
    sat_smoothmax_temperature_m: float = 0.05
    collision_margin_m: float = 0.0
    collision_hinge_temperature_m: float = 0.10
    ttc_safe_horizon_seconds: float = 3.0
    ttc_closing_temperature_mps: float = 0.05
    ttc_hinge_temperature_m: float = 0.10
    route_projection_temperature_m2: float = 0.25
    progress_positive_part_epsilon_m: float = 1e-6
    denominator_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.horizon_steps != 6:
            raise DrivingFunctionalLossError("the shared interface requires six future steps")
        if not math.isclose(self.dt_seconds, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise DrivingFunctionalLossError("the shared interface requires dt=0.5 seconds")
        positive = (
            self.sat_smooth_abs_epsilon,
            self.sat_smoothmax_temperature_m,
            self.collision_hinge_temperature_m,
            self.ttc_safe_horizon_seconds,
            self.ttc_closing_temperature_mps,
            self.ttc_hinge_temperature_m,
            self.route_projection_temperature_m2,
            self.progress_positive_part_epsilon_m,
            self.denominator_epsilon,
        )
        if any(value <= 0.0 for value in positive):
            raise DrivingFunctionalLossError(
                "all smoothing and denominator constants must be positive"
            )


@dataclass(frozen=True)
class DrivingFunctionalLossResult:
    """Per-mode function costs, two reductions, and geometric diagnostics."""

    per_mode: Mapping[str, torch.Tensor]
    confidence_weighted: Mapping[str, torch.Tensor]
    all_mode_mean: Mapping[str, torch.Tensor]
    mode_weights: torch.Tensor
    signed_clearance_m: torch.Tensor
    closing_speed_mps: torch.Tensor
    route_progress_m: torch.Tensor
    gt_route_progress_m: torch.Tensor


def _require_shape(name: str, tensor: torch.Tensor, dimensions: int, tail: tuple[int, ...]) -> None:
    if not torch.is_tensor(tensor):
        raise DrivingFunctionalLossError(f"{name} must be a torch tensor")
    if tensor.ndim != dimensions or tuple(tensor.shape[-len(tail) :]) != tail:
        raise DrivingFunctionalLossError(
            f"{name} must have {dimensions} dimensions ending in {tail}, got {tuple(tensor.shape)}"
        )


def _require_floating(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise DrivingFunctionalLossError(f"{name} must be floating point")


def _smooth_abs(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Subtracting epsilon makes abs(0) exactly zero while retaining a smooth
    # derivative everywhere.
    return torch.sqrt(value.square() + epsilon * epsilon) - epsilon


def _smooth_positive_part(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    # A smooth ReLU approximation.  Its epsilon/2 value at zero is deliberate
    # and negligible compared with metre-scale progress.
    return 0.5 * (value + torch.sqrt(value.square() + epsilon * epsilon))


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _normalised_sigmoid_weights(logits: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return detached, numerically normalised independent confidences."""

    probability = logits.detach().sigmoid()
    total = probability.sum(dim=-1, keepdim=True)
    normalised = probability / total.clamp_min(epsilon)
    uniform = torch.full_like(probability, 1.0 / probability.shape[-1])
    return torch.where(total > epsilon, normalised, uniform)


def _box_axes(boxes_xyhlw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    heading = boxes_xyhlw[..., 2]
    forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    lateral = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
    return forward, lateral


def smooth_obb_signed_clearance(
    ego_boxes_xyhlw: torch.Tensor,
    agent_boxes_xyhlw: torch.Tensor,
    *,
    smooth_abs_epsilon: float = 1e-6,
    smoothmax_temperature_m: float = 0.05,
) -> torch.Tensor:
    """Conservative smooth-SAT clearance for four oriented-box axes.

    Args:
        ego_boxes_xyhlw: ``[B, M, T, 5]`` boxes ordered x/y/heading/length/width.
        agent_boxes_xyhlw: constant ``[B, T, A, 5]`` boxes in the same frame.

    Returns:
        ``[B, M, T, A]`` signed clearance.  Positive values separate the
        boxes, negative values indicate overlap.  The log-mean-exp reduction
        is a conservative smooth approximation to the SAT maximum.
    """

    _require_shape("ego_boxes_xyhlw", ego_boxes_xyhlw, 4, (5,))
    _require_shape("agent_boxes_xyhlw", agent_boxes_xyhlw, 4, (5,))
    if ego_boxes_xyhlw.shape[0] != agent_boxes_xyhlw.shape[0]:
        raise DrivingFunctionalLossError("ego and agent box batches differ")
    if ego_boxes_xyhlw.shape[2] != agent_boxes_xyhlw.shape[1]:
        raise DrivingFunctionalLossError("ego and agent box horizons differ")
    if smooth_abs_epsilon <= 0.0 or smoothmax_temperature_m <= 0.0:
        raise DrivingFunctionalLossError("SAT smoothing constants must be positive")

    agent = agent_boxes_xyhlw.detach()
    ego = ego_boxes_xyhlw.unsqueeze(3)
    agent = agent.unsqueeze(1)
    ego_forward, ego_lateral = _box_axes(ego)
    agent_forward, agent_lateral = _box_axes(agent)
    batch, modes, steps, _ = ego_boxes_xyhlw.shape
    agents = agent_boxes_xyhlw.shape[2]
    axes = torch.stack(
        (
            ego_forward.expand(batch, modes, steps, agents, 2),
            ego_lateral.expand(batch, modes, steps, agents, 2),
            agent_forward.expand(batch, modes, steps, agents, 2),
            agent_lateral.expand(batch, modes, steps, agents, 2),
        ),
        dim=-2,
    )

    centre_delta = agent[..., :2] - ego[..., :2]
    centre_projection = (centre_delta.unsqueeze(-2) * axes).sum(dim=-1)
    centre_projection = _smooth_abs(centre_projection, smooth_abs_epsilon)

    def projected_radius(
        box: torch.Tensor, forward: torch.Tensor, lateral: torch.Tensor
    ) -> torch.Tensor:
        half_length = 0.5 * box[..., 3]
        half_width = 0.5 * box[..., 4]
        forward_projection = _smooth_abs(
            (forward.unsqueeze(-2) * axes).sum(dim=-1), smooth_abs_epsilon
        )
        lateral_projection = _smooth_abs(
            (lateral.unsqueeze(-2) * axes).sum(dim=-1), smooth_abs_epsilon
        )
        return half_length.unsqueeze(-1) * forward_projection + half_width.unsqueeze(
            -1
        ) * lateral_projection

    ego_radius = projected_radius(ego, ego_forward, ego_lateral)
    agent_radius = projected_radius(agent, agent_forward, agent_lateral)
    axis_clearance = centre_projection - ego_radius - agent_radius
    temperature = axis_clearance.new_tensor(smoothmax_temperature_m)
    return temperature * (
        torch.logsumexp(axis_clearance / temperature, dim=-1)
        - math.log(axis_clearance.shape[-1])
    )


def _masked_mean_per_mode(
    value: torch.Tensor, mask: torch.Tensor, *, modes: int, epsilon: float
) -> torch.Tensor:
    expanded = mask.detach().to(dtype=value.dtype).unsqueeze(1).expand(
        -1, modes, *mask.shape[1:]
    )
    numerator = (value * expanded).flatten(start_dim=2).sum(dim=-1)
    denominator = expanded.flatten(start_dim=2).sum(dim=-1)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(epsilon),
        torch.zeros_like(numerator),
    )


def directed_route_soft_progress(
    points_xy: torch.Tensor,
    route_xy: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    temperature_m2: float = 0.25,
    denominator_epsilon: float = 1e-9,
) -> torch.Tensor:
    """Soft-project points onto an oriented route and return arc progress.

    ``points_xy`` may be ``[B,M,T,2]`` or ``[B,T,2]``.  Route geometry and
    validity are treated as constants.  Direction is given by route order.
    """

    if points_xy.ndim not in (3, 4) or points_xy.shape[-1] != 2:
        raise DrivingFunctionalLossError("points_xy must be [B,T,2] or [B,M,T,2]")
    _require_shape("route_xy", route_xy, 3, (2,))
    if route_valid.ndim != 2 or route_valid.shape != route_xy.shape[:2]:
        raise DrivingFunctionalLossError("route_valid must match route_xy [B,R]")
    if points_xy.shape[0] != route_xy.shape[0] or route_xy.shape[1] < 2:
        raise DrivingFunctionalLossError("route batch differs or has fewer than two points")
    if temperature_m2 <= 0.0 or denominator_epsilon <= 0.0:
        raise DrivingFunctionalLossError("route smoothing constants must be positive")

    squeeze_mode = points_xy.ndim == 3
    points = points_xy.unsqueeze(1) if squeeze_mode else points_xy
    route = route_xy.detach()
    valid = route_valid.detach().bool()
    start = route[:, :-1]
    delta = route[:, 1:] - start
    length = torch.linalg.vector_norm(delta, dim=-1)
    segment_valid = valid[:, :-1] & valid[:, 1:] & (length > denominator_epsilon)
    if not bool(segment_valid.any(dim=-1).all()):
        raise DrivingFunctionalLossError("each route must contain at least one valid segment")

    unit = delta / length.clamp_min(denominator_epsilon).unsqueeze(-1)
    point_delta = points.unsqueeze(-2) - start[:, None, None]
    scalar = (point_delta * unit[:, None, None]).sum(dim=-1)
    scalar = scalar.clamp(min=0.0, max=None)
    scalar = torch.minimum(scalar, length[:, None, None])
    projection = start[:, None, None] + scalar.unsqueeze(-1) * unit[:, None, None]
    distance_sq = (points.unsqueeze(-2) - projection).square().sum(dim=-1)
    score = -distance_sq / temperature_m2
    score = score.masked_fill(~segment_valid[:, None, None], -torch.inf)
    weight = torch.softmax(score, dim=-1)

    valid_length = torch.where(segment_valid, length, torch.zeros_like(length))
    prefix = torch.cumsum(valid_length, dim=-1) - valid_length
    along_route = prefix[:, None, None] + scalar
    progress = (weight * along_route).sum(dim=-1)
    return progress.squeeze(1) if squeeze_mode else progress


def driving_functional_losses(
    *,
    ego_modes_xyh: torch.Tensor,
    mode_logits: torch.Tensor,
    ego_gt_xyh: torch.Tensor,
    ego_v0_xy: torch.Tensor,
    ego_a0_xy: torch.Tensor,
    ego_extent_lw: torch.Tensor,
    agent_boxes_xyhlw: torch.Tensor,
    agent_velocity_xy: torch.Tensor,
    agent_valid: torch.Tensor,
    route_xy: torch.Tensor,
    route_valid: torch.Tensor,
    config: DrivingFunctionalLossConfig = DrivingFunctionalLossConfig(),
) -> DrivingFunctionalLossResult:
    """Compute differentiable safety, comfort, and route-progress costs.

    Only ``ego_modes_xyh`` receives gradients.  Mode confidence, ego/agent GT,
    initial state, obstacle geometry, velocities, and route geometry are
    detached inside the core.  This prevents a safety loss from moving agent
    predictions or altering a frozen perception system.
    """

    _require_shape("ego_modes_xyh", ego_modes_xyh, 4, (config.horizon_steps, 3))
    _require_shape("ego_gt_xyh", ego_gt_xyh, 3, (config.horizon_steps, 3))
    _require_shape("ego_v0_xy", ego_v0_xy, 2, (2,))
    _require_shape("ego_a0_xy", ego_a0_xy, 2, (2,))
    _require_shape("agent_boxes_xyhlw", agent_boxes_xyhlw, 4, (5,))
    if mode_logits.ndim != 2 or mode_logits.shape != ego_modes_xyh.shape[:2]:
        raise DrivingFunctionalLossError("mode_logits must match ego modes [B,M]")
    batch, modes, steps, _ = ego_modes_xyh.shape
    if ego_gt_xyh.shape[0] != batch or ego_v0_xy.shape[0] != batch or ego_a0_xy.shape[0] != batch:
        raise DrivingFunctionalLossError("ego tensor batches differ")
    if agent_boxes_xyhlw.shape[:2] != (batch, steps):
        raise DrivingFunctionalLossError("agent boxes must be [B,T,A,5]")
    if agent_valid.shape != agent_boxes_xyhlw.shape[:3]:
        raise DrivingFunctionalLossError("agent_valid must match [B,T,A]")
    if agent_velocity_xy.ndim == 3:
        if agent_velocity_xy.shape != (batch, agent_boxes_xyhlw.shape[2], 2):
            raise DrivingFunctionalLossError("static agent velocity must be [B,A,2]")
        agent_velocity_xy = agent_velocity_xy[:, None].expand(-1, steps, -1, -1)
    elif agent_velocity_xy.ndim == 4:
        if agent_velocity_xy.shape != (*agent_boxes_xyhlw.shape[:3], 2):
            raise DrivingFunctionalLossError("agent velocity must be [B,T,A,2]")
    else:
        raise DrivingFunctionalLossError("agent velocity must be [B,A,2] or [B,T,A,2]")
    if ego_extent_lw.ndim == 1:
        if ego_extent_lw.shape != (2,):
            raise DrivingFunctionalLossError("shared ego extent must be [2]")
        ego_extent_lw = ego_extent_lw[None].expand(batch, -1)
    elif ego_extent_lw.shape != (batch, 2):
        raise DrivingFunctionalLossError("ego extent must be [2] or [B,2]")
    for name, tensor in (
        ("ego_modes_xyh", ego_modes_xyh),
        ("mode_logits", mode_logits),
        ("ego_gt_xyh", ego_gt_xyh),
        ("ego_v0_xy", ego_v0_xy),
        ("ego_a0_xy", ego_a0_xy),
        ("ego_extent_lw", ego_extent_lw),
        ("agent_boxes_xyhlw", agent_boxes_xyhlw),
        ("agent_velocity_xy", agent_velocity_xy),
        ("route_xy", route_xy),
    ):
        _require_floating(name, tensor)

    gt = ego_gt_xyh.detach()
    initial_velocity = ego_v0_xy.detach()
    initial_acceleration = ego_a0_xy.detach()
    extent = ego_extent_lw.detach().to(dtype=ego_modes_xyh.dtype, device=ego_modes_xyh.device)
    agents = agent_boxes_xyhlw.detach().to(
        dtype=ego_modes_xyh.dtype, device=ego_modes_xyh.device
    )
    agent_velocity = agent_velocity_xy.detach().to(
        dtype=ego_modes_xyh.dtype, device=ego_modes_xyh.device
    )
    valid = agent_valid.detach().to(device=ego_modes_xyh.device).bool()
    if not bool(torch.isfinite(extent).all()) or not bool((extent > 0).all()):
        raise DrivingFunctionalLossError("ego extent must be finite positive length,width")
    if agents.shape[2] and (
        not bool(torch.isfinite(agents).all())
        or not bool((agents[..., 3:5] > 0).all())
    ):
        raise DrivingFunctionalLossError(
            "agent boxes must be finite with positive length,width"
        )

    ego_extent = extent[:, None, None].expand(-1, modes, steps, -1)
    ego_boxes = torch.cat((ego_modes_xyh, ego_extent), dim=-1)
    if agents.shape[2] == 0:
        clearance = ego_modes_xyh.new_empty((batch, modes, steps, 0))
        closing_speed = clearance.clone()
        collision_per_mode = ego_modes_xyh.new_zeros((batch, modes))
        ttc_per_mode = ego_modes_xyh.new_zeros((batch, modes))
    else:
        clearance = smooth_obb_signed_clearance(
            ego_boxes,
            agents,
            smooth_abs_epsilon=config.sat_smooth_abs_epsilon,
            smoothmax_temperature_m=config.sat_smoothmax_temperature_m,
        )
        collision_pair = F.softplus(
            (config.collision_margin_m - clearance)
            / config.collision_hinge_temperature_m
        ) * config.collision_hinge_temperature_m
        collision_per_mode = _masked_mean_per_mode(
            collision_pair,
            valid,
            modes=modes,
            epsilon=config.denominator_epsilon,
        )

        # Estimate clearance one frame before the first prediction.  The
        # current centres are reconstructed using the supplied velocities;
        # headings and extents are held at their first-future values.
        ego_previous = ego_boxes[:, :, :1].clone()
        ego_previous[..., :2] = (
            ego_boxes[:, :, :1, :2]
            - initial_velocity[:, None, None] * config.dt_seconds
        )
        agent_previous = agents[:, :1].clone()
        agent_previous[..., :2] = (
            agents[:, :1, :, :2]
            - agent_velocity[:, :1] * config.dt_seconds
        )
        previous_clearance = smooth_obb_signed_clearance(
            ego_previous,
            agent_previous,
            smooth_abs_epsilon=config.sat_smooth_abs_epsilon,
            smoothmax_temperature_m=config.sat_smoothmax_temperature_m,
        )
        augmented = torch.cat((previous_clearance, clearance), dim=2)
        closing_speed = (augmented[:, :, :-1] - augmented[:, :, 1:]) / config.dt_seconds
        positive_closing = F.softplus(
            closing_speed / config.ttc_closing_temperature_mps
        ) * config.ttc_closing_temperature_mps
        safe_horizon_violation = (
            config.ttc_safe_horizon_seconds * positive_closing - clearance
        )
        ttc_pair = F.softplus(
            safe_horizon_violation / config.ttc_hinge_temperature_m
        ) * config.ttc_hinge_temperature_m
        ttc_per_mode = _masked_mean_per_mode(
            ttc_pair,
            valid,
            modes=modes,
            epsilon=config.denominator_epsilon,
        )

    position = ego_modes_xyh[..., :2]
    segment_velocity = (position[:, :, 1:] - position[:, :, :-1]) / config.dt_seconds
    velocity = torch.cat(
        (initial_velocity[:, None, None].expand(-1, modes, -1, -1), segment_velocity),
        dim=2,
    )
    derived_acceleration = (velocity[:, :, 1:] - velocity[:, :, :-1]) / config.dt_seconds
    acceleration = torch.cat(
        (
            initial_acceleration[:, None, None].expand(-1, modes, -1, -1),
            derived_acceleration,
        ),
        dim=2,
    )
    jerk = (acceleration[:, :, 1:] - acceleration[:, :, :-1]) / config.dt_seconds
    heading_delta = _wrap_angle(ego_modes_xyh[:, :, 1:, 2] - ego_modes_xyh[:, :, :-1, 2])
    yaw_rate = heading_delta / config.dt_seconds
    segment_length = torch.linalg.vector_norm(
        position[:, :, 1:] - position[:, :, :-1], dim=-1
    )
    curvature = heading_delta / segment_length.clamp_min(config.denominator_epsilon)

    acceleration_per_mode = acceleration.square().sum(dim=-1).mean(dim=-1)
    jerk_per_mode = jerk.square().sum(dim=-1).mean(dim=-1)
    yaw_rate_per_mode = yaw_rate.square().mean(dim=-1)
    curvature_per_mode = curvature.square().mean(dim=-1)

    progress = directed_route_soft_progress(
        position,
        route_xy.to(dtype=position.dtype, device=position.device),
        route_valid.to(device=position.device),
        temperature_m2=config.route_projection_temperature_m2,
        denominator_epsilon=config.denominator_epsilon,
    )
    gt_progress = directed_route_soft_progress(
        gt[..., :2].to(dtype=position.dtype, device=position.device),
        route_xy.to(dtype=position.dtype, device=position.device),
        route_valid.to(device=position.device),
        temperature_m2=config.route_projection_temperature_m2,
        denominator_epsilon=config.denominator_epsilon,
    ).detach()
    progress_shortfall = gt_progress[:, None] - progress
    progress_per_mode = _smooth_positive_part(
        progress_shortfall, config.progress_positive_part_epsilon_m
    ).mean(dim=-1)

    per_mode = {
        "collision": collision_per_mode,
        "ttc": ttc_per_mode,
        "acceleration": acceleration_per_mode,
        "jerk": jerk_per_mode,
        "yaw_rate": yaw_rate_per_mode,
        "curvature": curvature_per_mode,
        "progress": progress_per_mode,
    }
    mode_weights = _normalised_sigmoid_weights(mode_logits, config.denominator_epsilon)
    confidence_weighted = {
        name: (value * mode_weights).sum(dim=-1).mean() for name, value in per_mode.items()
    }
    all_mode_mean = {name: value.mean() for name, value in per_mode.items()}
    return DrivingFunctionalLossResult(
        per_mode=per_mode,
        confidence_weighted=confidence_weighted,
        all_mode_mean=all_mode_mean,
        mode_weights=mode_weights,
        signed_clearance_m=clearance,
        closing_speed_mps=closing_speed,
        route_progress_m=progress,
        gt_route_progress_m=gt_progress,
    )


__all__ = [
    "DrivingFunctionalLossConfig",
    "DrivingFunctionalLossError",
    "DrivingFunctionalLossResult",
    "directed_route_soft_progress",
    "driving_functional_losses",
    "smooth_obb_signed_clearance",
]

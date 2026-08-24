"""Auditable continual-learning baselines adapted to end-to-end driving."""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from selector_bench.continual.drive_opd import (
    DriveOPDError,
    QueryFunction,
    bernoulli_mode_forward_kl,
    bernoulli_mode_reverse_kl,
)


EWC_ESTIMATOR = "per_example_official_loss_gradient_second_moment_eval_mode"


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@dataclass(frozen=True)
class EWCState:
    anchor: dict[str, torch.Tensor]
    fisher: dict[str, torch.Tensor]
    sample_count: int
    estimator: str = EWC_ESTIMATOR

    @classmethod
    def from_model_and_fisher(
        cls,
        model: torch.nn.Module,
        fisher: Mapping[str, torch.Tensor],
        sample_count: int,
    ) -> "EWCState":
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if names != set(fisher):
            missing = sorted(names - set(fisher))
            extra = sorted(set(fisher) - names)
            raise DriveOPDError(f"EWC Fisher mismatch; missing={missing[:3]} extra={extra[:3]}")
        return cls(
            anchor={
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            fisher={name: value.detach().cpu().clone() for name, value in fisher.items()},
            sample_count=int(sample_count),
            estimator=EWC_ESTIMATOR,
        )

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        for name, parameter in model.named_parameters():
            if name not in self.fisher:
                continue
            anchor = self.anchor[name].to(parameter.device, parameter.dtype)
            fisher = self.fisher[name].to(parameter.device, parameter.dtype)
            terms.append((fisher * (parameter - anchor).square()).sum())
        if not terms:
            raise DriveOPDError("EWC penalty found no registered trainable parameters")
        return 0.5 * torch.stack(terms).sum()


class DiagonalFisherAccumulator:
    """Accumulate a per-example diagonal loss-gradient second moment.

    DiffusionDrive's official objective is a composite surrogate rather than a
    normalized log-likelihood, so this is an explicit EWC importance proxy and
    not a claim that the statistic is the model's true Fisher information.
    Batch-gradient squaring is forbidden because cross-example gradient terms
    make it a biased replacement for the per-example second moment.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.values = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.sample_count = 0

    @torch.no_grad()
    def add(self, model: torch.nn.Module, batch_size: int) -> None:
        if batch_size != 1:
            raise DriveOPDError(
                "EWC importance requires one-example backward calls; "
                "squared aggregate batch gradients are not accepted"
            )
        for name, parameter in model.named_parameters():
            if name in self.values:
                # A conditional multimodal head can legitimately be inactive for
                # a batch.  Its empirical Fisher contribution is zero, not an
                # estimator failure.
                if parameter.grad is not None:
                    self.values[name].add_(parameter.grad.detach().cpu().square())
        self.sample_count += 1

    def finalize(self, model: torch.nn.Module) -> EWCState:
        if self.sample_count <= 0:
            raise DriveOPDError("cannot finalize an empty Fisher accumulator")
        fisher = {name: value / self.sample_count for name, value in self.values.items()}
        return EWCState.from_model_and_fisher(model, fisher, self.sample_count)


@contextlib.contextmanager
def preserve_module_buffers(model: torch.nn.Module) -> Iterable[None]:
    """Restore every module buffer after an auxiliary train-mode forward.

    A-GEM needs gradients of the replay loss but the replay sample is not part
    of the deployed model's forward stream.  Keeping train mode preserves the
    same response function as the current-batch loss; restoring buffers prevents
    the auxiliary pass from silently applying a second BatchNorm/statistics
    update.
    """

    snapshot = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    try:
        yield
    finally:
        current = dict(model.named_buffers())
        if current.keys() != snapshot.keys():
            raise DriveOPDError("module buffers changed identity during protected forward")
        with torch.no_grad():
            for name, value in snapshot.items():
                current[name].copy_(value)


def capture_gradient(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().flatten()
        for parameter in parameters
    ]
    if not pieces:
        raise DriveOPDError("cannot capture an empty parameter list")
    return torch.cat(pieces)


@torch.no_grad()
def set_gradient_(parameters: Sequence[torch.nn.Parameter], gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        piece = gradient[offset : offset + size].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = piece.clone()
        else:
            parameter.grad.copy_(piece)
        offset += size
    if offset != gradient.numel():
        raise DriveOPDError("flat gradient has the wrong length")


@torch.no_grad()
def project_agem_gradient_(
    parameters: Sequence[torch.nn.Parameter],
    reference_gradient: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Project the current gradient only when it increases replay-buffer loss."""

    current = capture_gradient(parameters)
    reference = reference_gradient.to(current.device, current.dtype)
    if current.shape != reference.shape:
        raise DriveOPDError("A-GEM current/reference gradient shapes differ")
    dot = torch.dot(current, reference)
    norm_sq = torch.dot(reference, reference)
    projected = dot < 0 and norm_sq > epsilon
    if projected:
        current = current - dot / norm_sq * reference
        set_gradient_(parameters, current)
    return {
        "agem_projected": float(bool(projected)),
        "agem_gradient_dot": float(dot.item()),
        "agem_reference_norm": float(norm_sq.sqrt().item()),
    }


def derpp_drive_loss(
    current_supervised_loss: torch.Tensor,
    replay_supervised_loss: torch.Tensor,
    replay_dark_response_loss: torch.Tensor,
    *,
    replay_weight: float = 1.0,
    dark_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine current data, old-log replay, and stored structured responses."""

    total = (
        current_supervised_loss
        + replay_weight * replay_supervised_loss
        + dark_weight * replay_dark_response_loss
    )
    return total, {
        "derpp_current_supervised": float(current_supervised_loss.detach().item()),
        "derpp_replay_supervised": float(replay_supervised_loss.detach().item()),
        "derpp_dark_response": float(replay_dark_response_loss.detach().item()),
    }


class StagewiseLoRALinear(torch.nn.Module):
    """Task-ID-free additive LoRA: inference sums all learned stage adapters."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise DriveOPDError("LoRA rank and alpha must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.a = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        self.start_stage()

    def start_stage(self) -> None:
        for parameter in [*self.a, *self.b]:
            parameter.requires_grad_(False)
        a = torch.nn.Parameter(self.base.weight.new_empty(self.rank, self.base.in_features))
        b = torch.nn.Parameter(self.base.weight.new_zeros(self.base.out_features, self.rank))
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.a.append(a)
        self.b.append(b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        scale = self.alpha / self.rank
        for a, b in zip(self.a, self.b):
            result = result + F.linear(F.linear(inputs, a), b) * scale
        return result

    def orthogonality_penalty(self) -> torch.Tensor:
        if len(self.a) <= 1:
            return self.a[-1].sum() * 0.0
        current_a, current_b = self.a[-1], self.b[-1]
        terms = []
        for previous_a, previous_b in zip(self.a[:-1], self.b[:-1]):
            terms.append((current_a @ previous_a.detach().T).square().mean())
            terms.append((current_b.T @ previous_b.detach()).square().mean())
        return torch.stack(terms).mean()


def _resolve_parent(model: torch.nn.Module, qualified_name: str) -> tuple[torch.nn.Module, str]:
    components = qualified_name.split(".")
    parent = model
    for component in components[:-1]:
        parent = parent[int(component)] if component.isdigit() else getattr(parent, component)
    return parent, components[-1]


def inject_stagewise_lora(
    model: torch.nn.Module,
    *,
    include_name_fragments: Sequence[str],
    rank: int = 8,
    alpha: float = 16.0,
) -> list[str]:
    """Replace selected Linear layers and return the exact modified names."""

    selected = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(fragment in name for fragment in include_name_fragments)
    ]
    if not selected:
        raise DriveOPDError(f"no Linear layer matched {include_name_fragments}")
    for name in selected:
        parent, child_name = _resolve_parent(model, name)
        original = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        replacement = StagewiseLoRALinear(original, rank=rank, alpha=alpha)
        if child_name.isdigit():
            parent[int(child_name)] = replacement
        else:
            setattr(parent, child_name, replacement)
    return sorted(selected)


def start_new_lora_stage(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, StagewiseLoRALinear):
            module.start_stage()
            count += 1
    if count == 0:
        raise DriveOPDError("model has no StagewiseLoRALinear modules")
    return count


def lora_orthogonality_penalty(model: torch.nn.Module) -> torch.Tensor:
    terms = [
        module.orthogonality_penalty()
        for module in model.modules()
        if isinstance(module, StagewiseLoRALinear)
    ]
    if not terms:
        raise DriveOPDError("model has no StagewiseLoRALinear modules")
    return torch.stack(terms).mean()


def _forward_mode_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    return bernoulli_mode_forward_kl(teacher_logits, student_logits)


def aler_adversarial_latent_search(
    initial_state: torch.Tensor,
    student_query: QueryFunction,
    teacher_query: QueryFunction,
    timestep: int,
    *,
    search_steps: int = 1,
    step_size: float = 0.05,
    radius: float = 0.1,
    mode_weight: float = 1.0,
    gradient_floor: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Search a bounded latent for maximal fixed-teacher/student disagreement."""

    if search_steps <= 0 or step_size <= 0 or radius <= 0 or gradient_floor < 0:
        raise DriveOPDError("ALER search steps, step size, and radius must be positive")
    origin = initial_state.detach()
    latent = origin.clone()
    last_objective = 0.0
    for _ in range(search_steps):
        latent.requires_grad_(True)
        time = torch.full(
            (latent.shape[0],), timestep, dtype=torch.long, device=latent.device
        )
        student_response, student_mode = student_query(latent, time)
        with torch.no_grad():
            teacher_response, teacher_mode = teacher_query(latent, time)
        objective = F.mse_loss(student_response, teacher_response) + mode_weight * _forward_mode_kl(
            teacher_mode, student_mode
        )
        gradient = torch.autograd.grad(objective, latent, only_inputs=True)[0]
        flat = gradient.flatten(1)
        gradient_norm = flat.norm(dim=1)
        active = gradient_norm > gradient_floor
        normalized = gradient / gradient_norm.clamp_min(1e-12).view(
            -1, *([1] * (gradient.ndim - 1))
        )
        normalized = normalized * active.view(
            -1, *([1] * (gradient.ndim - 1))
        )
        proposal = latent.detach() + step_size * normalized
        delta = proposal - origin
        flat_delta = delta.flatten(1)
        factor = (radius / flat_delta.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
        latent = origin + delta * factor.view(-1, *([1] * (delta.ndim - 1)))
        last_objective = float(objective.detach().item())
    distance = (latent - origin).flatten(1).norm(dim=1)
    return latent.detach(), {
        "aler_search_objective": last_objective,
        "aler_search_l2_mean": float(distance.mean().item()),
        "aler_teacher_queries": float(search_steps),
        "aler_active_search_fraction": float(active.float().mean().item()),
    }


def aler_repair_loss(
    state: torch.Tensor,
    student_query: QueryFunction,
    teacher_query: QueryFunction,
    timestep: int,
    *,
    mode_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ALER repair: response MSE plus reverse Bernoulli KL per anchor."""

    time = torch.full((state.shape[0],), timestep, dtype=torch.long, device=state.device)
    student_response, student_logits = student_query(state, time)
    with torch.no_grad():
        teacher_response, teacher_logits = teacher_query(state, time)
    response = F.mse_loss(student_response, teacher_response)
    reverse_kl = bernoulli_mode_reverse_kl(teacher_logits, student_logits)
    return response + mode_weight * reverse_kl, {
        "aler_repair_response_mse": float(response.detach().item()),
        "aler_repair_mode_bernoulli_reverse_kl": float(reverse_kl.detach().item()),
        "aler_teacher_queries": 1.0,
    }


def nearest_manifold_distance(
    searched_latent: torch.Tensor, real_latents: torch.Tensor
) -> torch.Tensor:
    """Distance each searched latent to the nearest observed driving latent."""

    searched = searched_latent.flatten(1).float()
    reference = real_latents.flatten(1).float()
    if searched.shape[1] != reference.shape[1] or reference.shape[0] == 0:
        raise DriveOPDError("searched and real latent manifolds are incompatible")
    return torch.cdist(searched, reference).min(dim=1).values


def talr_scene_weights(
    teacher_trajectories: torch.Tensor,
    teacher_mode_logits: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    minimum_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Map token confidence to task-ID-free scene weights for trajectory modes."""

    if teacher_trajectories.ndim != 4 or ground_truth.ndim != 3:
        raise DriveOPDError("TALR expects trajectories [B,M,T,D] and GT [B,T,D]")
    if teacher_trajectories.shape[0] != ground_truth.shape[0]:
        raise DriveOPDError("TALR batch sizes differ")
    distances = (
        teacher_trajectories[..., :2] - ground_truth[:, None, :, :2]
    ).square().mean(dim=(-1, -2))
    nearest = distances.argmin(dim=1)
    # DiffusionDrive trains every anchor with sigmoid focal loss.  Confidence
    # in the nearest trajectory is therefore that anchor's independent
    # Bernoulli probability, not a softmax share across mutually exclusive
    # modes.
    confidence = (
        teacher_mode_logits.detach()
        .sigmoid()
        .gather(1, nearest[:, None])
        .squeeze(1)
    )
    weights = confidence.clamp_min(minimum_weight)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.detach(), {
        "talr_teacher_selected_anchor_sigmoid_confidence_mean": float(
            confidence.mean().item()
        ),
        "talr_scene_weight_min": float(weights.min().item()),
        "talr_scene_weight_max": float(weights.max().item()),
    }


def deterministic_replay_choice(
    current_tokens: Sequence[str],
    replay_tokens: Sequence[str],
    *,
    step: int,
    seed: int,
    replay_fraction: float,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """Choose a step-matched current/replay batch without changing step count."""

    if not current_tokens or not replay_tokens:
        raise DriveOPDError("step-matched replay needs both current and replay tokens")
    if not 0.0 < replay_fraction < 1.0 or batch_size <= 1:
        raise DriveOPDError("invalid replay fraction or batch size")
    replay_count = min(batch_size - 1, max(1, round(batch_size * replay_fraction)))
    current_count = batch_size - replay_count

    def choose(pool: Sequence[str], count: int, salt: str) -> list[str]:
        ranked = sorted(
            pool,
            key=lambda token: hashlib.sha256(
                f"{seed}\0{step}\0{salt}\0{token}".encode()
            ).digest(),
        )
        if count <= len(ranked):
            return ranked[:count]
        return [ranked[index % len(ranked)] for index in range(count)]

    return choose(current_tokens, current_count, "current"), choose(
        replay_tokens, replay_count, "replay"
    )

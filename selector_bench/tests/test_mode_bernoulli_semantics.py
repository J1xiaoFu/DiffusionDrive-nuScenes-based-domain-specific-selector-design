from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from selector_bench.continual import baselines
from selector_bench.continual.drive_opd import (
    DiffusionDriveOPDAdapter,
    PlanningDistillationConfig,
    bernoulli_mode_forward_kl,
    bernoulli_mode_reverse_kl,
    planning_distillation_loss,
)
from selector_bench.continual import nuscenes_sparsedrive_opd


def _load_context_swap_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / (
        "43_audit_diffusiondrive_context_swap.py"
    )
    spec = importlib.util.spec_from_file_location("context_swap_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_logit_translation_is_not_categorically_invariant() -> None:
    teacher = torch.full((2, 6), -5.0)
    student = torch.full((2, 6), 5.0, requires_grad=True)
    loss = bernoulli_mode_forward_kl(teacher, student)
    assert float(loss) > 4.0
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_identical_logits_are_exact_zero_and_teacher_is_detached() -> None:
    teacher = torch.tensor([[-8.0, -1.0, 0.0, 3.0, 9.0]], requires_grad=True)
    student = teacher.detach().clone().requires_grad_(True)
    forward = bernoulli_mode_forward_kl(teacher, student)
    reverse = bernoulli_mode_reverse_kl(teacher, student)
    assert float(forward) == 0.0
    assert float(reverse) == 0.0
    (forward + reverse).backward()
    assert teacher.grad is None
    assert student.grad is not None
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad))


def test_extreme_logits_are_finite_for_forward_and_reverse_kl() -> None:
    teacher = torch.tensor([[-1000.0, 1000.0, -80.0, 80.0]], requires_grad=True)
    student = torch.tensor([[1000.0, -1000.0, 80.0, -80.0]], requires_grad=True)
    forward = bernoulli_mode_forward_kl(teacher, student)
    reverse = bernoulli_mode_reverse_kl(teacher, student)
    assert torch.isfinite(forward)
    assert torch.isfinite(reverse)
    (forward + reverse).backward()
    assert teacher.grad is None
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_drive_opd_response_mse_is_unchanged_by_mode_semantics() -> None:
    scale = torch.nn.Parameter(torch.tensor(2.0))

    def student(state: torch.Tensor, time: torch.Tensor):
        del time
        return state * scale, torch.full((state.shape[0], 6), 5.0)

    def teacher(state: torch.Tensor, time: torch.Tensor):
        del time
        return state, torch.full((state.shape[0], 6), -5.0)

    states = [torch.ones(2, 3), torch.full((2, 3), 2.0)]
    loss, metrics = planning_distillation_loss(
        student,
        teacher,
        states,
        [10, 0],
        PlanningDistillationConfig(response_weight=1.0, mode_kl_weight=0.0),
    )
    expected = torch.stack(
        [torch.nn.functional.mse_loss(state * scale, state) for state in states]
    ).mean()
    torch.testing.assert_close(loss, expected)
    assert metrics["planning_response_mse"] == float(expected.detach())


def test_aler_forward_reverse_and_talr_use_sigmoid_anchor_semantics() -> None:
    teacher = torch.full((2, 3), -5.0)
    student = torch.full((2, 3), 5.0)
    assert float(baselines._forward_mode_kl(teacher, student)) > 4.0
    assert float(bernoulli_mode_reverse_kl(teacher, student)) > 4.0

    trajectories = torch.zeros(2, 2, 3, 2)
    trajectories[:, 1] = 10.0
    ground_truth = torch.zeros(2, 3, 2)
    logits = torch.tensor([[-5.0, 10.0], [-1.0, 10.0]])
    shifted = logits + 10.0
    original_weights, original_metrics = baselines.talr_scene_weights(
        trajectories, logits, ground_truth, minimum_weight=1e-9
    )
    shifted_weights, _ = baselines.talr_scene_weights(
        trajectories, shifted, ground_truth, minimum_weight=1e-9
    )
    assert not torch.allclose(original_weights, shifted_weights)
    assert "talr_teacher_selected_anchor_sigmoid_confidence_mean" in original_metrics


def test_context_swap_uses_bernoulli_kl_and_selected_anchor_agreement() -> None:
    module = _load_context_swap_module()
    response = torch.zeros(2, 3)
    reference = (response, torch.full((2, 6), -5.0))
    candidate = (response.clone(), torch.full((2, 6), 5.0))
    metrics = module.response_metrics(candidate, reference)
    assert torch.all(metrics["mode_bernoulli_forward_kl"] > 4.0)
    assert torch.equal(
        metrics["mode_selected_anchor_changed"], torch.zeros(2)
    )
    identical = module.response_metrics(reference, reference)
    assert torch.equal(
        identical["mode_bernoulli_forward_kl"], torch.zeros(2)
    )


class _NoOpScheduler:
    def set_timesteps(self, steps: int, device: torch.device) -> None:
        self.steps = (steps, device)

    @staticmethod
    def add_noise(
        *, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        del noise, timesteps
        return original_samples


class _SyntheticSparseDriveHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ego_fut_mode = 2
        self.ego_fut_ts = 3
        self.diffusion_scheduler = _NoOpScheduler()

    @staticmethod
    def normalize_ego_fut_trajs(value: torch.Tensor) -> torch.Tensor:
        return value


def test_nuscenes_real_adapter_uses_bernoulli_kl_without_changing_response() -> None:
    student = _SyntheticSparseDriveHead()
    teacher = _SyntheticSparseDriveHead()
    student_context = {
        "batch_size": 1,
        "tgt_cmd_plan_anchor": torch.zeros(1, 2, 3, 2),
    }
    teacher_context = {"batch_size": 1}
    original_query = nuscenes_sparsedrive_opd._query

    def query(head, context, latent, timestep):
        del context, timestep
        if head is student:
            return latent + 2.0, torch.full((1, 2), 5.0)
        return latent, torch.full((1, 2), -5.0)

    nuscenes_sparsedrive_opd._query = query
    try:
        total, metrics = nuscenes_sparsedrive_opd.planning_distillation(
            student,
            teacher,
            student_context,
            teacher_context,
            support="shared_exogenous",
            mode_weight=1.0,
            iteration=0,
        )
    finally:
        nuscenes_sparsedrive_opd._query = original_query

    torch.testing.assert_close(
        metrics["drive_cl_planning_response_mse"], torch.tensor(4.0)
    )
    assert float(metrics["drive_cl_mode_bernoulli_forward_kl"]) > 4.0
    assert float(metrics["drive_cl_mode_selected_anchor_agreement"]) == 1.0
    assert float(total) > 8.0


def test_all_targeted_mode_paths_have_no_categorical_softmax_kl() -> None:
    targeted = (
        inspect.getsource(planning_distillation_loss),
        inspect.getsource(DiffusionDriveOPDAdapter.student_support_planning_loss),
        inspect.getsource(nuscenes_sparsedrive_opd.planning_distillation),
        inspect.getsource(baselines._forward_mode_kl),
        inspect.getsource(baselines.aler_repair_loss),
        inspect.getsource(baselines.talr_scene_weights),
        inspect.getsource(_load_context_swap_module().response_metrics),
    )
    for source in targeted:
        assert ".log_softmax(" not in source
        assert ".softmax(" not in source


def test_official_planning_score_contract_is_sigmoid_focal() -> None:
    repository = Path(__file__).resolve().parents[2]
    config = (
        repository
        / "DiffusionDrive/projects/configs/diffusiondrive_configs/"
        / "diffusiondrive_small_stage2.py"
    ).read_text()
    decoder = (
        repository
        / "DiffusionDrive/projects/mmdet3d_plugin/models/motion/decoder.py"
    ).read_text()
    planning_loss = config[config.index("plan_loss_cls=dict(") :]
    assert "type='FocalLoss'" in planning_loss[:300]
    assert "use_sigmoid=True" in planning_loss[:300]
    assert '"planning_score": cls.sigmoid().cpu()' in decoder

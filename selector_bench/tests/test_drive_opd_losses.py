from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from selector_bench.continual.drive_opd import (
    DiffusionDriveOPDAdapter,
    PerceptionDistillationConfig,
    PlanningDistillationConfig,
    masked_bev_distillation,
    matched_agent_distillation,
    planning_distillation_loss,
    update_ema_teacher_,
)


class _IdentityScheduler:
    @staticmethod
    def add_noise(base, noise, timestep):
        del timestep
        return base + noise


class DriveOPDLossTest(unittest.TestCase):
    def test_agent_matching_is_permutation_invariant_and_differentiable(self) -> None:
        teacher_states = torch.tensor(
            [[[1.0, 2.0, 0.1, 4.0, 2.0], [9.0, 3.0, -0.2, 3.0, 1.0]]]
        )
        teacher_logits = torch.tensor([[5.0, 4.0]])
        student_states = teacher_states.flip(1).clone().requires_grad_(True)
        student_logits = teacher_logits.flip(1).clone().requires_grad_(True)
        loss, metrics = matched_agent_distillation(
            student_states,
            student_logits,
            teacher_states,
            teacher_logits,
            PerceptionDistillationConfig(agent_confidence=0.7),
        )
        self.assertLess(float(loss), 1e-6)
        self.assertEqual(metrics["matched_agents"], 2.0)
        loss.backward()
        self.assertIsNotNone(student_states.grad)

    def test_bev_mask_only_preserves_selected_confident_classes(self) -> None:
        teacher = torch.full((1, 4, 2, 2), -8.0)
        teacher[:, 1, 0, 0] = 8.0
        teacher[:, 2, 0, 1] = 8.0
        teacher[:, 3, 1, 0] = 8.0
        student = teacher.clone().requires_grad_(True)
        loss, metrics = masked_bev_distillation(
            student,
            teacher,
            PerceptionDistillationConfig(bev_class_ids=(1, 3), bev_confidence=0.9),
        )
        self.assertLess(abs(float(loss)), 1e-6)
        self.assertEqual(metrics["bev_masked_cells"], 2.0)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_lwf_and_opd_loss_share_query_implementation(self) -> None:
        scale = torch.nn.Parameter(torch.tensor(0.8))

        def student(state: torch.Tensor, time: torch.Tensor):
            response = state * scale
            logits = torch.stack([scale.expand(state.shape[0]), -scale.expand(state.shape[0])], -1)
            return response, logits

        def teacher(state: torch.Tensor, time: torch.Tensor):
            return state, torch.tensor([[1.0, -1.0]]).repeat(state.shape[0], 1)

        states = [torch.ones(2, 3), torch.full((2, 3), 2.0)]
        loss, metrics = planning_distillation_loss(
            student,
            teacher,
            states,
            [10, 0],
            PlanningDistillationConfig(),
        )
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(metrics["planning_query_states"], 2.0)
        loss.backward()
        self.assertIsNotNone(scale.grad)

    def test_lwf_exogenous_states_match_planner_dtype(self) -> None:
        adapter = DiffusionDriveOPDAdapter.__new__(DiffusionDriveOPDAdapter)
        head = SimpleNamespace(
            plan_anchor=torch.zeros(2, 4, 2, dtype=torch.float32),
            norm_odo=lambda value: value,
            diffusion_scheduler=_IdentityScheduler(),
        )
        adapter.student = SimpleNamespace(_trajectory_head=head)
        targets = {"trajectory": torch.zeros(3, 4, 3, dtype=torch.float64)}
        states = adapter.exogenous_states(
            targets,
            PlanningDistillationConfig(rollout_timesteps=(10, 0)),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(len(states), 2)
        self.assertTrue(all(state.dtype == torch.float32 for state in states))

    def test_ema_update_has_declared_momentum(self) -> None:
        student = torch.nn.Linear(2, 1, bias=False)
        teacher = torch.nn.Linear(2, 1, bias=False)
        student.weight.data.fill_(1.0)
        teacher.weight.data.zero_()
        update_ema_teacher_(teacher, student, momentum=0.75)
        torch.testing.assert_close(teacher.weight, torch.full_like(teacher.weight, 0.25))


if __name__ == "__main__":
    unittest.main()

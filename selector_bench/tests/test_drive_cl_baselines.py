from __future__ import annotations

import unittest

import torch

from selector_bench.continual.baselines import (
    DiagonalFisherAccumulator,
    aler_adversarial_latent_search,
    aler_repair_loss,
    capture_gradient,
    deterministic_replay_choice,
    inject_stagewise_lora,
    lora_orthogonality_penalty,
    nearest_manifold_distance,
    project_agem_gradient_,
    start_new_lora_stage,
    talr_scene_weights,
)


class DriveCLBaselinesTest(unittest.TestCase):
    def test_ewc_fisher_and_penalty(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        accumulator = DiagonalFisherAccumulator(model)
        model(torch.ones(3, 2)).sum().backward()
        accumulator.add(model, batch_size=3)
        state = accumulator.finalize(model)
        self.assertEqual(state.sample_count, 3)
        model.weight.data.add_(0.2)
        self.assertGreater(float(state.penalty(model)), 0.0)

    def test_ewc_treats_inactive_parameters_as_zero_importance(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Linear(1, 1))
        accumulator = DiagonalFisherAccumulator(model)
        model[0](torch.ones(2, 2)).sum().backward()
        accumulator.add(model, batch_size=2)
        state = accumulator.finalize(model)
        self.assertEqual(float(state.fisher["1.weight"].sum()), 0.0)

    def test_agem_projection_removes_negative_dot(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([-1.0, 0.0])
        reference = torch.tensor([1.0, 0.0])
        metrics = project_agem_gradient_([parameter], reference)
        self.assertEqual(metrics["agem_projected"], 1.0)
        self.assertGreaterEqual(float(torch.dot(capture_gradient([parameter]), reference)), -1e-7)

    def test_stagewise_lora_is_task_id_free(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        names = inject_stagewise_lora(model, include_name_fragments=("0", "2"), rank=2, alpha=2)
        self.assertEqual(names, ["0", "2"])
        before = model(torch.ones(1, 3))
        self.assertEqual(start_new_lora_stage(model), 2)
        after = model(torch.ones(1, 3))
        torch.testing.assert_close(before, after)
        penalty = lora_orthogonality_penalty(model)
        self.assertTrue(torch.isfinite(penalty))

    def test_aler_search_and_manifold_audit(self) -> None:
        scale = torch.nn.Parameter(torch.tensor(0.5))

        def student(state, time):
            return state * scale, torch.stack([state[:, 0], -state[:, 0]], dim=-1)

        def teacher(state, time):
            return state, torch.zeros(state.shape[0], 2)

        initial = torch.ones(2, 3)
        searched, search_metrics = aler_adversarial_latent_search(
            initial, student, teacher, 10, search_steps=2, radius=0.2
        )
        self.assertLessEqual(float((searched - initial).flatten(1).norm(dim=1).max()), 0.20001)
        repair, _ = aler_repair_loss(searched, student, teacher, 10)
        repair.backward()
        self.assertIsNotNone(scale.grad)
        distances = nearest_manifold_distance(searched, initial[:1])
        self.assertEqual(tuple(distances.shape), (2,))
        self.assertEqual(search_metrics["aler_teacher_queries"], 2.0)

    def test_talr_and_replay_are_deterministic(self) -> None:
        trajectories = torch.zeros(2, 3, 4, 2)
        trajectories[:, 1] = 1.0
        ground_truth = torch.ones(2, 4, 2)
        logits = torch.tensor([[0.0, 2.0, -1.0], [0.0, 1.0, -1.0]])
        weights, _ = talr_scene_weights(trajectories, logits, ground_truth)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        first = deterministic_replay_choice(
            ["n1", "n2", "n3"], ["o1", "o2"], step=3, seed=7, replay_fraction=0.5, batch_size=4
        )
        second = deterministic_replay_choice(
            ["n1", "n2", "n3"], ["o1", "o2"], step=3, seed=7, replay_fraction=0.5, batch_size=4
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

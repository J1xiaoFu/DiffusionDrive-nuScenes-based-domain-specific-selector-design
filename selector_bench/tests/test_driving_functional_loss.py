from __future__ import annotations

import math
import unittest

import torch

from selector_bench.continual.driving_functional_loss import (
    directed_route_soft_progress,
    driving_functional_losses,
    smooth_obb_signed_clearance,
)


DTYPE = torch.double


def _fixture(*, agents: int = 1) -> dict[str, torch.Tensor]:
    x = torch.arange(1, 7, dtype=DTYPE)
    mode_0 = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    mode_1 = torch.stack((0.8 * x, 0.15 * x, 0.05 * x), dim=-1)
    ego = torch.stack((mode_0, mode_1), dim=0).unsqueeze(0).requires_grad_()
    gt = mode_0.unsqueeze(0)
    if agents:
        boxes = torch.zeros((1, 6, agents, 5), dtype=DTYPE)
        boxes[..., 0] = 20.0
        boxes[..., 3] = 4.0
        boxes[..., 4] = 2.0
        velocity = torch.zeros((1, 6, agents, 2), dtype=DTYPE)
        valid = torch.ones((1, 6, agents), dtype=torch.bool)
    else:
        boxes = torch.empty((1, 6, 0, 5), dtype=DTYPE)
        velocity = torch.empty((1, 6, 0, 2), dtype=DTYPE)
        valid = torch.empty((1, 6, 0), dtype=torch.bool)
    return {
        "ego_modes_xyh": ego,
        "mode_logits": torch.tensor([[2.0, -1.0]], dtype=DTYPE),
        "ego_gt_xyh": gt,
        "ego_v0_xy": torch.tensor([[2.0, 0.0]], dtype=DTYPE),
        "ego_a0_xy": torch.zeros((1, 2), dtype=DTYPE),
        "ego_extent_lw": torch.tensor([4.0, 2.0], dtype=DTYPE),
        "agent_boxes_xyhlw": boxes,
        "agent_velocity_xy": velocity,
        "agent_valid": valid,
        "route_xy": torch.tensor([[[0.0, 0.0], [10.0, 0.0]]], dtype=DTYPE),
        "route_valid": torch.ones((1, 2), dtype=torch.bool),
    }


def _rigid_xy(xy: torch.Tensor, angle: float, translation: tuple[float, float]) -> torch.Tensor:
    rotation = xy.new_tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return xy @ rotation.T + xy.new_tensor(translation)


class DrivingFunctionalLossTest(unittest.TestCase):
    def test_gradcheck_and_finite_components(self) -> None:
        fixture = _fixture()

        def scalar(ego: torch.Tensor) -> torch.Tensor:
            result = driving_functional_losses(**{**fixture, "ego_modes_xyh": ego})
            return torch.stack(tuple(result.all_mode_mean.values())).sum()

        self.assertTrue(torch.autograd.gradcheck(scalar, (fixture["ego_modes_xyh"],)))
        result = driving_functional_losses(**fixture)
        for value in (*result.per_mode.values(), *result.confidence_weighted.values()):
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_rigid_translation_rotation_invariance(self) -> None:
        fixture = _fixture()
        reference = driving_functional_losses(**fixture)
        angle = 0.73
        translation = (14.0, -8.0)
        transformed = {key: value.clone() for key, value in fixture.items()}
        transformed_ego = fixture["ego_modes_xyh"].detach().clone()
        transformed_ego[..., :2] = _rigid_xy(transformed_ego[..., :2], angle, translation)
        transformed_ego[..., 2] += angle
        transformed["ego_modes_xyh"] = transformed_ego.requires_grad_()
        transformed_gt = fixture["ego_gt_xyh"].clone()
        transformed_gt[..., :2] = _rigid_xy(transformed_gt[..., :2], angle, translation)
        transformed_gt[..., 2] += angle
        transformed["ego_gt_xyh"] = transformed_gt
        transformed_agents = fixture["agent_boxes_xyhlw"].clone()
        transformed_agents[..., :2] = _rigid_xy(
            transformed_agents[..., :2], angle, translation
        )
        transformed_agents[..., 2] += angle
        transformed["agent_boxes_xyhlw"] = transformed_agents
        transformed["route_xy"] = _rigid_xy(fixture["route_xy"], angle, translation)
        transformed["ego_v0_xy"] = _rigid_xy(fixture["ego_v0_xy"], angle, (0.0, 0.0))
        transformed["ego_a0_xy"] = _rigid_xy(fixture["ego_a0_xy"], angle, (0.0, 0.0))
        transformed["agent_velocity_xy"] = _rigid_xy(
            fixture["agent_velocity_xy"], angle, (0.0, 0.0)
        )
        actual = driving_functional_losses(**transformed)
        for name in reference.per_mode:
            torch.testing.assert_close(actual.per_mode[name], reference.per_mode[name])

    def test_no_agents_is_exact_zero_safety_risk(self) -> None:
        result = driving_functional_losses(**_fixture(agents=0))
        torch.testing.assert_close(result.per_mode["collision"], torch.zeros((1, 2), dtype=DTYPE))
        torch.testing.assert_close(result.per_mode["ttc"], torch.zeros((1, 2), dtype=DTYPE))
        self.assertEqual(tuple(result.signed_clearance_m.shape), (1, 2, 6, 0))

    def test_approaching_obstacle_increases_collision_and_ttc(self) -> None:
        far = _fixture()
        close = _fixture()
        close["agent_boxes_xyhlw"][..., 0] = 5.0
        far_result = driving_functional_losses(**far)
        close_result = driving_functional_losses(**close)
        self.assertGreater(
            close_result.all_mode_mean["collision"].item(),
            far_result.all_mode_mean["collision"].item(),
        )
        self.assertGreater(
            close_result.all_mode_mean["ttc"].item(),
            far_result.all_mode_mean["ttc"].item(),
        )

    def test_stopped_trajectory_has_progress_shortfall(self) -> None:
        fixture = _fixture(agents=0)
        moving = fixture["ego_gt_xyh"].unsqueeze(1)
        stopped = torch.zeros_like(moving)
        fixture["ego_modes_xyh"] = torch.cat((moving, stopped), dim=1).requires_grad_()
        fixture["mode_logits"] = torch.zeros((1, 2), dtype=DTYPE)
        result = driving_functional_losses(**fixture)
        self.assertLess(result.per_mode["progress"][0, 0], 1e-5)
        self.assertGreater(result.per_mode["progress"][0, 1], 2.0)

    def test_mode_reduction_is_detached_sigmoid_not_softmax(self) -> None:
        fixture = _fixture(agents=0)
        logits = torch.tensor([[-5.0, 1.0]], dtype=DTYPE, requires_grad=True)
        fixture["mode_logits"] = logits
        result = driving_functional_losses(**fixture)
        expected = logits.detach().sigmoid()
        expected = expected / expected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(result.mode_weights, expected)
        torch.testing.assert_close(
            result.all_mode_mean["progress"], result.per_mode["progress"].mean()
        )
        shifted = driving_functional_losses(
            **{**fixture, "mode_logits": (logits.detach() + 5.0)}
        )
        self.assertFalse(torch.allclose(result.mode_weights, shifted.mode_weights))
        result.confidence_weighted["progress"].backward()
        self.assertIsNone(logits.grad)

    def test_gradients_stop_at_all_constant_boundaries(self) -> None:
        fixture = _fixture()
        for key in (
            "mode_logits",
            "ego_gt_xyh",
            "ego_v0_xy",
            "ego_a0_xy",
            "ego_extent_lw",
            "agent_boxes_xyhlw",
            "agent_velocity_xy",
            "route_xy",
        ):
            fixture[key] = fixture[key].clone().requires_grad_()
        result = driving_functional_losses(**fixture)
        torch.stack(tuple(result.confidence_weighted.values())).sum().backward()
        self.assertIsNotNone(fixture["ego_modes_xyh"].grad)
        self.assertTrue(bool(torch.isfinite(fixture["ego_modes_xyh"].grad).all()))
        for key in (
            "mode_logits",
            "ego_gt_xyh",
            "ego_v0_xy",
            "ego_a0_xy",
            "ego_extent_lw",
            "agent_boxes_xyhlw",
            "agent_velocity_xy",
            "route_xy",
        ):
            self.assertIsNone(fixture[key].grad, key)

    def test_extreme_logits_and_geometry_remain_finite(self) -> None:
        fixture = _fixture()
        fixture["mode_logits"] = torch.tensor([[-1000.0, 1000.0]], dtype=DTYPE)
        fixture["ego_modes_xyh"] = (
            fixture["ego_modes_xyh"].detach().clone() * 100.0
        ).requires_grad_()
        result = driving_functional_losses(**fixture)
        total = torch.stack(tuple(result.confidence_weighted.values())).sum()
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()
        self.assertTrue(bool(torch.isfinite(fixture["ego_modes_xyh"].grad).all()))

    def test_smooth_sat_has_ego_gradient_and_detaches_agents(self) -> None:
        ego = torch.tensor([[[[0.0, 0.0, 0.2, 4.0, 2.0]]]], dtype=DTYPE, requires_grad=True)
        agent = torch.tensor([[[[3.0, 0.2, -0.1, 4.0, 2.0]]]], dtype=DTYPE, requires_grad=True)
        clearance = smooth_obb_signed_clearance(ego, agent)
        clearance.sum().backward()
        self.assertTrue(bool(torch.isfinite(ego.grad).all()))
        self.assertIsNone(agent.grad)

    def test_directed_route_progress_is_not_forward_displacement(self) -> None:
        route = torch.tensor([[[0.0, 0.0], [0.0, 5.0], [-5.0, 5.0]]], dtype=DTYPE)
        valid = torch.ones((1, 3), dtype=torch.bool)
        points = torch.tensor([[[0.0, 4.0], [-3.0, 5.0]]], dtype=DTYPE)
        progress = directed_route_soft_progress(points, route, valid)
        self.assertGreater(progress[0, 1], progress[0, 0])
        # The later point has a smaller x coordinate: directed arc progress,
        # not any fixed-axis displacement, determines the ordering.
        self.assertLess(points[0, 1, 0], points[0, 0, 0])


if __name__ == "__main__":
    unittest.main()

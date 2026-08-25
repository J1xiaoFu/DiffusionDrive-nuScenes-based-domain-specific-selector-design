from __future__ import annotations

import math
import unittest

import torch

from selector_bench.continual.driving_functional_loss import (
    DrivingFunctionalLossError,
    driving_functional_losses,
)
from selector_bench.integrations.diffusiondrive.nuscenes_functional_loss import (
    NuScenesAgentGeometry,
    build_nuscenes_functional_tensors,
    build_nuscenes_gt_agents,
    build_nuscenes_predicted_agents,
    nuscenes_status_to_planning_state,
    pad_nuscenes_gt_agents,
    select_nuscenes_planning_modes,
)


DTYPE = torch.double


class _Boxes:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class NuScenesFunctionalLossAdapterTest(unittest.TestCase):
    def test_real_command_shape_selects_then_cumsums_six_modes(self) -> None:
        deltas = torch.zeros((2, 1, 18, 6, 2), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            deltas[:, :, :, :, 1] = 1.0
            deltas[0, 0, 6:12, :, 0] = 0.25
        logits = torch.arange(36, dtype=DTYPE).reshape(2, 1, 18)
        command = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
        xyh, selected_logits = select_nuscenes_planning_modes(deltas, logits, command)
        self.assertEqual(tuple(xyh.shape), (2, 6, 6, 3))
        self.assertEqual(tuple(selected_logits.shape), (2, 6))
        torch.testing.assert_close(xyh[0, :, -1, 0], torch.full((6,), 1.5, dtype=DTYPE))
        torch.testing.assert_close(xyh[:, :, -1, 1], torch.full((2, 6), 6.0, dtype=DTYPE))
        torch.testing.assert_close(selected_logits[0], logits[0, 0, 6:12])
        xyh[..., :2].sum().backward()
        self.assertIsNotNone(deltas.grad)
        self.assertEqual(torch.count_nonzero(deltas.grad[0, 0, :6]), 0)
        self.assertGreater(torch.count_nonzero(deltas.grad[0, 0, 6:12]), 0)

    def test_predicted_agents_decode_length_width_and_detach(self) -> None:
        anchors = torch.zeros((1, 2, 11), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            anchors[0, :, 0] = torch.tensor([4.0, 8.0], dtype=DTYPE)
            anchors[0, :, 3] = math.log(4.5)  # encoded x-size / physical length
            anchors[0, :, 4] = math.log(2.0)  # encoded y-size / physical width
            anchors[0, :, 6] = 0.0
            anchors[0, :, 7] = 1.0
        motion = torch.zeros((1, 2, 3, 6, 2), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            motion[..., 0] = 0.5
            motion[:, :, 1, :, 1] = 0.3
        logits = torch.tensor([[[2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]], dtype=DTYPE)
        geometry = build_nuscenes_predicted_agents(
            anchors, motion, logits, torch.tensor([[True, False]])
        )
        self.assertEqual(tuple(geometry.boxes_xyhlw.shape), (1, 6, 2, 5))
        torch.testing.assert_close(
            geometry.boxes_xyhlw[0, :, :, 3], torch.full((6, 2), 4.5, dtype=DTYPE)
        )
        torch.testing.assert_close(
            geometry.boxes_xyhlw[0, :, :, 4], torch.full((6, 2), 2.0, dtype=DTYPE)
        )
        self.assertFalse(geometry.boxes_xyhlw.requires_grad)
        self.assertFalse(geometry.velocity_xy.requires_grad)
        self.assertTrue(bool(geometry.valid[0, :, 0].all()))
        self.assertFalse(bool(geometry.valid[0, :, 1].any()))

    def test_gt_agents_decode_width_length_and_future_delta(self) -> None:
        current = torch.zeros((1, 1, 9), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            current[..., 0] = 3.0
            current[..., 3] = 4.2  # decoded x-size / physical length
            current[..., 4] = 1.8  # decoded y-size / physical width
            current[..., 6] = 0.1
        future = torch.zeros((1, 1, 6, 2), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            future[..., 0] = 0.5
        geometry = build_nuscenes_gt_agents(
            current, future, torch.ones((1, 1, 6), dtype=torch.bool)
        )
        torch.testing.assert_close(
            geometry.boxes_xyhlw[0, -1, 0],
            torch.tensor([6.0, 0.0, 0.0, 4.2, 1.8], dtype=DTYPE),
        )
        self.assertFalse(geometry.boxes_xyhlw.requires_grad)

    def test_status_maps_forward_left_to_right_forward(self) -> None:
        status = torch.zeros((1, 10), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            status[0, 0] = 2.0
            status[0, 1] = -0.5
            status[0, 6] = 8.0
            status[0, 7] = 1.25
        velocity, acceleration = nuscenes_status_to_planning_state(status)
        torch.testing.assert_close(velocity, torch.tensor([[-1.25, 8.0]], dtype=DTYPE))
        torch.testing.assert_close(acceleration, torch.tensor([[0.5, 2.0]], dtype=DTYPE))
        self.assertFalse(velocity.requires_grad)
        self.assertFalse(acceleration.requires_grad)

    def test_variable_training_lists_pad_without_reordering(self) -> None:
        first = _Boxes(
            torch.tensor(
                [[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.0]], dtype=DTYPE
            )
        )
        second = _Boxes(
            torch.tensor(
                [
                    [3.0, 4.0, 0.0, 2.1, 4.1, 1.5, 0.1],
                    [5.0, 6.0, 0.0, 2.2, 4.2, 1.5, 0.2],
                ],
                dtype=DTYPE,
            )
        )
        future = [torch.zeros((1, 6, 2), dtype=DTYPE), torch.zeros((2, 6, 2), dtype=DTYPE)]
        valid = [torch.ones((1, 6), dtype=torch.bool), torch.ones((2, 6), dtype=torch.bool)]
        boxes, future_padded, valid_padded = pad_nuscenes_gt_agents(
            [first, second], future, valid
        )
        self.assertEqual(tuple(boxes.shape), (2, 2, 7))
        self.assertEqual(tuple(future_padded.shape), (2, 2, 6, 2))
        self.assertFalse(bool(valid_padded[0, 1].any()))
        torch.testing.assert_close(boxes[1, 1, :2], torch.tensor([5.0, 6.0], dtype=DTYPE))

    def test_real_adapter_core_gradient_reaches_planner_not_agents_or_status(self) -> None:
        plan = torch.zeros((1, 1, 18, 6, 2), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            plan[..., 1] = 0.8
        planning_logits = torch.zeros((1, 1, 18), dtype=DTYPE, requires_grad=True)
        agent_boxes = torch.zeros((1, 6, 1, 5), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            agent_boxes[..., 1] = 7.0
            agent_boxes[..., 2] = math.pi / 2.0
            agent_boxes[..., 3] = 4.0
            agent_boxes[..., 4] = 2.0
        agent_velocity = torch.zeros((1, 6, 1, 2), dtype=DTYPE, requires_grad=True)
        agents = NuScenesAgentGeometry(
            agent_boxes,
            agent_velocity,
            torch.ones((1, 6, 1), dtype=torch.bool),
        )
        ego_gt = torch.zeros((1, 6, 2), dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            ego_gt[..., 1] = 1.0
        status = torch.zeros((1, 10), dtype=DTYPE, requires_grad=True)
        payload = build_nuscenes_functional_tensors(
            planning_deltas_xy=plan,
            planning_logits=planning_logits,
            command=torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
            ego_gt_deltas_xy=ego_gt,
            ego_gt_valid=torch.ones((1, 6), dtype=torch.bool),
            ego_status=status,
            agents=agents,
        )
        result = driving_functional_losses(**payload.as_core_kwargs())
        total = torch.stack(tuple(result.confidence_weighted.values())).sum()
        total.backward()
        self.assertIsNotNone(plan.grad)
        self.assertTrue(bool(torch.isfinite(plan.grad).all()))
        self.assertIsNone(planning_logits.grad)
        self.assertIsNone(agent_boxes.grad)
        self.assertIsNone(agent_velocity.grad)
        self.assertIsNone(ego_gt.grad)
        self.assertIsNone(status.grad)

    def test_shape_and_extent_contracts_fail_closed(self) -> None:
        with self.assertRaises(DrivingFunctionalLossError):
            select_nuscenes_planning_modes(
                torch.zeros((1, 5, 6, 2), dtype=DTYPE),
                torch.zeros((1, 5), dtype=DTYPE),
                torch.zeros((1,), dtype=torch.long),
            )
        current = torch.zeros((1, 1, 7), dtype=DTYPE)
        with self.assertRaises(DrivingFunctionalLossError):
            build_nuscenes_gt_agents(
                current,
                torch.zeros((1, 1, 6, 2), dtype=DTYPE),
                torch.ones((1, 1, 6), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()

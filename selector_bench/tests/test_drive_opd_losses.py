from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from selector_bench.continual.baselines import (
    aler_adversarial_latent_search,
    aler_repair_loss,
)
from selector_bench.continual.drive_opd import (
    DriveOPDConfig,
    DriveOPDError,
    DiffusionDriveOPDAdapter,
    PerceptionDistillationConfig,
    PlanningDistillationConfig,
    configure_rollout_schedule,
    masked_bev_distillation,
    matched_agent_distillation,
    planning_distillation_loss,
    update_ema_teacher_,
)


class _IdentityScheduler:
    config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, inference_steps, device):
        self.inference_steps = inference_steps
        stride = self.config.num_train_timesteps // inference_steps
        self.timesteps = torch.arange(
            self.config.num_train_timesteps - stride,
            -1,
            -stride,
            device=device,
        )

    @staticmethod
    def add_noise(base, noise, timestep):
        del timestep
        return base + noise

    @staticmethod
    def step(model_output, timestep, sample):
        del timestep
        return SimpleNamespace(prev_sample=sample - model_output)


class _NonAdjacentScheduler(_IdentityScheduler):
    def set_timesteps(self, inference_steps, device):
        self.inference_steps = inference_steps
        self.timesteps = torch.tensor([20, 10, 5, 0], device=device)


class _FrozenStateScheduler(_IdentityScheduler):
    def __init__(self):
        self.step_calls: list[int] = []

    def step(self, model_output, timestep, sample):
        del model_output
        self.step_calls.append(int(timestep))
        return SimpleNamespace(prev_sample=sample)


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

    def test_real_adapter_query_audit_counts_and_rejects_budget_mismatch(self) -> None:
        adapter = DiffusionDriveOPDAdapter.__new__(DiffusionDriveOPDAdapter)
        adapter.student = torch.nn.Module()
        adapter.teacher = torch.nn.Module()
        adapter._active_query_audit = None
        adapter._query_impl = lambda planner, context, state, time: (
            state,
            torch.zeros(state.shape[0], 2),
        )
        state = torch.zeros(2, 3)
        time = torch.full((2,), 10, dtype=torch.long)
        with adapter.capture_query_audit() as audit:
            adapter.query(adapter.student, None, state, time)
            adapter.query(adapter.teacher, None, state, time)
        self.assertEqual(audit.count("student"), 1)
        self.assertEqual(audit.count("teacher"), 1)
        audit.assert_budget(student=1, teacher=1)
        with self.assertRaisesRegex(DriveOPDError, "query budget mismatch"):
            audit.assert_budget(student=2, teacher=1)

    def test_aler_search_and_repair_use_measured_adapter_queries(self) -> None:
        adapter = DiffusionDriveOPDAdapter.__new__(DiffusionDriveOPDAdapter)
        adapter.student = torch.nn.Module()
        adapter.teacher = torch.nn.Module()
        adapter._active_query_audit = None
        scale = torch.nn.Parameter(torch.tensor(0.8))

        def query_impl(planner, context, state, time):
            del context, time
            if planner is adapter.student:
                logits = torch.stack(
                    [scale.expand(state.shape[0]), -scale.expand(state.shape[0])], -1
                )
                return state * scale, logits
            return state, torch.tensor([[1.0, -1.0]]).repeat(state.shape[0], 1)

        adapter._query_impl = query_impl
        state = torch.ones(2, 3)
        student_query = lambda value, time: adapter.query(
            adapter.student, None, value, time
        )
        teacher_query = lambda value, time: adapter.query(
            adapter.teacher, None, value, time
        )
        with adapter.capture_query_audit() as audit:
            searched, search_metrics = aler_adversarial_latent_search(
                state,
                student_query,
                teacher_query,
                10,
                search_steps=1,
            )
            loss, repair_metrics = aler_repair_loss(
                searched, student_query, teacher_query, 10
            )
        declared = int(
            search_metrics["aler_teacher_queries"]
            + repair_metrics["aler_teacher_queries"]
        )
        audit.assert_budget(student=declared, teacher=declared)
        self.assertEqual(declared, 2)
        loss.backward()
        self.assertIsNotNone(scale.grad)

    def test_full_distillation_loss_and_gradients_are_identical_with_query_audit(self) -> None:
        def build(scale: torch.nn.Parameter):
            scheduler = _FrozenStateScheduler()
            head = SimpleNamespace(
                plan_anchor=torch.zeros(2, 3, 2),
                norm_odo=lambda value: value,
                diffusion_scheduler=scheduler,
            )
            student = torch.nn.Module()
            student._trajectory_head = head
            teacher = torch.nn.Module()
            adapter = DiffusionDriveOPDAdapter.__new__(DiffusionDriveOPDAdapter)
            adapter.student = student
            adapter.teacher = teacher
            adapter._active_query_audit = None
            context = SimpleNamespace(
                ego_query=torch.zeros(2, 1),
                agent_states=torch.zeros(2, 1, 5),
            )
            adapter.deterministic_contexts = lambda features: (context, context)

            def query_impl(planner, query_context, state, time):
                del query_context, time
                if planner is student:
                    logits = torch.stack(
                        [scale.expand(state.shape[0]), -scale.expand(state.shape[0])],
                        dim=-1,
                    )
                    return state * scale, logits
                return state, torch.tensor([[1.0, -1.0]]).repeat(state.shape[0], 1)

            adapter._query_impl = query_impl
            return adapter

        plain_scale = torch.nn.Parameter(torch.tensor(0.8))
        audited_scale = torch.nn.Parameter(torch.tensor(0.8))
        plain_optimizer = torch.optim.AdamW([plain_scale], lr=1e-3)
        audited_optimizer = torch.optim.AdamW([audited_scale], lr=1e-3)
        plain = build(plain_scale)
        audited = build(audited_scale)
        config = DriveOPDConfig(lambda_perception=0.0, lambda_planning=1.0)
        targets = {"trajectory": torch.zeros(2, 3, 3)}
        plain_loss, plain_metrics = plain.distillation_loss(
            {},
            targets,
            config,
            support="student",
            generator=torch.Generator().manual_seed(17),
        )
        with audited.capture_query_audit() as audit:
            audited_loss, audited_metrics = audited.distillation_loss(
                {},
                targets,
                config,
                support="student",
                generator=torch.Generator().manual_seed(17),
            )
        audit.assert_budget(student=2, teacher=2)
        plain_loss.backward()
        audited_loss.backward()
        torch.testing.assert_close(plain_loss, audited_loss)
        torch.testing.assert_close(plain_scale.grad, audited_scale.grad)
        self.assertEqual(plain_metrics, audited_metrics)
        plain_optimizer.step()
        audited_optimizer.step()
        torch.testing.assert_close(plain_scale, audited_scale)
        plain_state = plain_optimizer.state[plain_scale]
        audited_state = audited_optimizer.state[audited_scale]
        self.assertEqual(set(plain_state), set(audited_state))
        for key in plain_state:
            if isinstance(plain_state[key], torch.Tensor):
                torch.testing.assert_close(plain_state[key], audited_state[key])
            else:
                self.assertEqual(plain_state[key], audited_state[key])
        self.assertEqual([record["role"] for record in audit.records], [
            "student",
            "teacher",
            "student",
            "teacher",
        ])
        self.assertTrue(all("timesteps" not in record for record in audit.records))

    def test_identical_registered_states_give_identical_losses_and_gradients(self) -> None:
        first_scale = torch.nn.Parameter(torch.tensor(0.8))
        second_scale = torch.nn.Parameter(torch.tensor(0.8))

        def query(scale):
            def implementation(state: torch.Tensor, time: torch.Tensor):
                del time
                logits = torch.stack(
                    [scale.expand(state.shape[0]), -scale.expand(state.shape[0])], -1
                )
                return state * scale, logits

            return implementation

        def teacher(state: torch.Tensor, time: torch.Tensor):
            del time
            return state, torch.tensor([[1.0, -1.0]]).repeat(state.shape[0], 1)

        registered = [torch.ones(2, 3), torch.full((2, 3), 2.0)]
        first_loss, _ = planning_distillation_loss(
            query(first_scale),
            teacher,
            registered,
            [10, 0],
            PlanningDistillationConfig(),
        )
        second_loss, _ = planning_distillation_loss(
            query(second_scale),
            teacher,
            [state.clone() for state in registered],
            [10, 0],
            PlanningDistillationConfig(),
        )
        first_loss.backward()
        second_loss.backward()
        torch.testing.assert_close(first_loss, second_loss)
        torch.testing.assert_close(first_scale.grad, second_scale.grad)

    def test_lwf_and_opd_match_when_query_state_distributions_are_identical(self) -> None:
        def build(scale: torch.nn.Parameter) -> DiffusionDriveOPDAdapter:
            head = SimpleNamespace(
                plan_anchor=torch.zeros(2, 3, 3),
                norm_odo=lambda value: value,
                diffusion_scheduler=_FrozenStateScheduler(),
            )
            student = torch.nn.Module()
            student._trajectory_head = head
            teacher = torch.nn.Module()
            adapter = DiffusionDriveOPDAdapter.__new__(DiffusionDriveOPDAdapter)
            adapter.student = student
            adapter.teacher = teacher
            adapter._active_query_audit = None
            context = SimpleNamespace(
                ego_query=torch.zeros(2, 1),
                agent_states=torch.zeros(2, 1, 5),
            )
            adapter.deterministic_contexts = lambda features: (context, context)

            def query_impl(planner, query_context, state, time):
                del query_context, time
                logits = torch.stack(
                    [scale.expand(state.shape[0]), -scale.expand(state.shape[0])], -1
                )
                if planner is student:
                    return state * scale, logits
                return state, torch.zeros_like(logits)

            adapter._query_impl = query_impl
            return adapter

        opd_scale = torch.nn.Parameter(torch.tensor(0.8))
        lwf_scale = torch.nn.Parameter(torch.tensor(0.8))
        opd = build(opd_scale)
        lwf = build(lwf_scale)
        config = DriveOPDConfig(lambda_perception=0.0, lambda_planning=1.0)
        targets = {"trajectory": torch.zeros(2, 3, 3)}
        opd_loss, opd_metrics = opd.distillation_loss(
            {},
            targets,
            config,
            support="student",
            generator=torch.Generator().manual_seed(31),
        )
        lwf_loss, lwf_metrics = lwf.distillation_loss(
            {},
            targets,
            config,
            support="exogenous",
            generator=torch.Generator().manual_seed(31),
        )
        opd_loss.backward()
        lwf_loss.backward()
        torch.testing.assert_close(opd_loss, lwf_loss)
        torch.testing.assert_close(opd_scale.grad, lwf_scale.grad)
        self.assertEqual(
            opd_metrics["student_denoiser_queries"],
            lwf_metrics["student_denoiser_queries"],
        )
        self.assertEqual(
            opd_metrics["teacher_denoiser_queries"],
            lwf_metrics["teacher_denoiser_queries"],
        )

    def test_real_support_branches_match_when_registered_states_are_forced_equal(self) -> None:
        def build(scale: torch.nn.Parameter):
            scheduler = _FrozenStateScheduler()
            head = SimpleNamespace(
                plan_anchor=torch.zeros(2, 3, 2),
                norm_odo=lambda value: value,
                diffusion_scheduler=scheduler,
            )
            student = torch.nn.Module()
            student._trajectory_head = head
            teacher = torch.nn.Module()
            adapter = DiffusionDriveOPDAdapter.__new__(DiffusionDriveOPDAdapter)
            adapter.student = student
            adapter.teacher = teacher
            context = SimpleNamespace(
                ego_query=torch.zeros(2, 1),
                agent_states=torch.zeros(2, 1, 5),
            )
            adapter.deterministic_contexts = lambda features: (context, context)
            calls = []

            def query(model, query_context, state, time):
                del query_context
                label = "student" if model is student else "teacher"
                calls.append((label, tuple(int(v) for v in time), state.detach().clone()))
                if model is student:
                    response = state * scale
                    logits = torch.stack(
                        [scale.expand(state.shape[0]), -scale.expand(state.shape[0])],
                        dim=-1,
                    )
                    return response, logits
                return state, torch.tensor([[1.0, -1.0]]).repeat(state.shape[0], 1)

            adapter.query = query
            return adapter, scheduler, calls

        student_scale = torch.nn.Parameter(torch.tensor(0.8))
        exogenous_scale = torch.nn.Parameter(torch.tensor(0.8))
        student_adapter, student_scheduler, student_calls = build(student_scale)
        exogenous_adapter, exogenous_scheduler, exogenous_calls = build(exogenous_scale)
        config = DriveOPDConfig(lambda_perception=0.0, lambda_planning=1.0)
        targets = {"trajectory": torch.zeros(2, 3, 3)}
        student_loss, student_metrics = student_adapter.distillation_loss(
            {},
            targets,
            config,
            support="student",
            generator=torch.Generator().manual_seed(9),
        )
        exogenous_loss, exogenous_metrics = exogenous_adapter.distillation_loss(
            {},
            targets,
            config,
            support="exogenous",
            generator=torch.Generator().manual_seed(9),
        )
        student_loss.backward()
        exogenous_loss.backward()

        torch.testing.assert_close(student_loss, exogenous_loss)
        torch.testing.assert_close(student_scale.grad, exogenous_scale.grad)
        self.assertEqual(student_metrics["student_denoiser_queries"], 2.0)
        self.assertEqual(student_metrics["teacher_denoiser_queries"], 2.0)
        self.assertEqual(exogenous_metrics["student_denoiser_queries"], 2.0)
        self.assertEqual(exogenous_metrics["teacher_denoiser_queries"], 2.0)
        self.assertEqual(student_scheduler.step_calls, [10])
        self.assertEqual(exogenous_scheduler.step_calls, [])
        self.assertEqual(len(student_calls), 4)
        self.assertEqual(len(exogenous_calls), 4)
        self.assertEqual(
            [(item[0], item[1]) for item in student_calls],
            [
                ("student", (10, 10)),
                ("teacher", (10, 10)),
                ("student", (0, 0)),
                ("teacher", (0, 0)),
            ],
        )
        self.assertEqual(
            [(item[0], item[1]) for item in exogenous_calls],
            [
                ("student", (10, 10)),
                ("teacher", (10, 10)),
                ("student", (0, 0)),
                ("teacher", (0, 0)),
            ],
        )
        self.assertEqual([item[:2] for item in student_calls], [item[:2] for item in exogenous_calls])
        for student_call, exogenous_call in zip(student_calls, exogenous_calls):
            torch.testing.assert_close(student_call[2], exogenous_call[2])

    def test_rollout_schedule_is_constant_stride_and_registered(self) -> None:
        scheduler = _IdentityScheduler()
        schedule = configure_rollout_schedule(
            scheduler,
            PlanningDistillationConfig(),
            torch.device("cpu"),
        )
        self.assertEqual(schedule.query_timesteps, (10, 0))
        self.assertEqual(schedule.initial_noise_timestep, 10)
        self.assertEqual(schedule.transition_stride, 10)
        self.assertEqual(schedule.scheduler_inference_steps, 100)
        self.assertEqual(scheduler.inference_steps, 100)

    def test_rollout_schedule_rejects_historical_time_mismatch(self) -> None:
        scheduler = _IdentityScheduler()
        with self.assertRaises(DriveOPDError):
            configure_rollout_schedule(
                scheduler,
                PlanningDistillationConfig(initial_noise_timestep=8),
                torch.device("cpu"),
            )
        with self.assertRaises(DriveOPDError):
            configure_rollout_schedule(
                scheduler,
                PlanningDistillationConfig(
                    rollout_timesteps=(20, 5, 0), initial_noise_timestep=20
                ),
                torch.device("cpu"),
            )
        with self.assertRaisesRegex(DriveOPDError, "does not transition directly"):
            configure_rollout_schedule(
                _NonAdjacentScheduler(),
                PlanningDistillationConfig(),
                torch.device("cpu"),
            )

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

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from selector_bench.continual.run_budget import (
    BudgetMeasurementError,
    RunBudgetCounter,
    TRANSIENT_STORAGE_SEMANTICS,
    TransientStorageCounter,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def counter() -> RunBudgetCounter:
    return RunBudgetCounter(
        current_identities=frozenset({"c1", "c2", "c3", "c4"}),
        old_identities=frozenset({"o1", "o2", "o3"}),
    )


class DriveCLRunBudgetTest(unittest.TestCase):
    @staticmethod
    def runner_module():
        script = REPOSITORY_ROOT / "selector_bench/scripts/41_train_drive_cl_diffusiondrive.py"
        spec = importlib.util.spec_from_file_location("drive_cl_runner_budget_fixture", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_token_dataset_returns_stable_cache_identity(self) -> None:
        module = self.runner_module()

        class Dataset:
            tokens = ["c1", "c2"]

            def __getitem__(self, index: int):
                return {"feature": index}, {"target": index}

        wrapped = module.TokenDataset(Dataset(), ["c2", "c1"])
        self.assertEqual(wrapped[0], ("c2", {"feature": 1}, {"target": 1}))

    def test_short_agem_run_targets_only_replay_identities_actually_scheduled(self) -> None:
        module = self.runner_module()
        buffer_tokens = tuple(f"o{index}" for index in range(10))
        plan = module.planned_agem_replay_batches(
            buffer_tokens,
            replay_batch_size=2,
            updates_per_epoch=2,
            start_epoch=1,
            end_epoch=1,
            seed=17,
            stage_index=2,
        )
        delivered = [token for batch in plan[1] for token in batch]
        self.assertEqual(len(delivered), 4)
        self.assertEqual(len(set(delivered)), 4)
        self.assertLess(len(set(delivered)), len(buffer_tokens))

        class Dataset:
            tokens = list(buffer_tokens)

            def __getitem__(self, index: int):
                return {"feature": index}, {"target": index}

        wrapped = module.TokenDataset(Dataset(), buffer_tokens)
        generator = module.torch.Generator()
        generator.manual_seed(module.stable_seed("agem-order", 17, 2, 1))
        loader = module.DataLoader(
            wrapped,
            batch_size=2,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        iterator = iter(loader)
        actual = tuple(tuple(next(iterator)[0]) for _ in range(2))
        self.assertEqual(actual, plan[1])

        measured = RunBudgetCounter(
            current_identities=frozenset({"c1"}),
            old_identities=frozenset(buffer_tokens),
        )
        for batch in plan[1]:
            measured.observe_batch(batch)
            measured.observe_optimizer_step(
                forward_calls=2,
                backward_calls=2,
                student_queries=0,
                teacher_queries=0,
            )
        snapshot = measured.snapshot(completed_epochs=1)
        self.assertEqual(snapshot["old_unique_identities"], len(set(delivered)))
        self.assertEqual(snapshot["old_presentations"], len(delivered))

    def test_complete_and_partial_budgets_are_measured(self) -> None:
        measured = counter()
        for _ in range(2):
            measured.observe_batch(["c1", "c2"])
            measured.observe_batch(["c3", "c4"])
            measured.observe_optimizer_step(
                forward_calls=1,
                backward_calls=1,
                student_queries=3,
                teacher_queries=3,
            )
            measured.observe_optimizer_step(
                forward_calls=1,
                backward_calls=1,
                student_queries=3,
                teacher_queries=3,
            )
        self.assertEqual(
            measured.snapshot(completed_epochs=2),
            {
                "epochs": 2,
                "optimizer_updates": 4,
                "current_unique_identities": 4,
                "old_unique_identities": 0,
                "current_presentations": 8,
                "old_presentations": 0,
                "forward_calls": 4,
                "backward_calls": 4,
                "student_queries": 12,
                "teacher_queries": 12,
            },
        )
        partial = counter()
        partial.observe_batch(["c1", "c2"])
        partial.observe_optimizer_step(
            forward_calls=1, backward_calls=1, student_queries=0, teacher_queries=0
        )
        self.assertEqual(partial.snapshot(completed_epochs=0)["current_presentations"], 2)
        self.assertNotEqual(partial.snapshot(completed_epochs=0), measured.snapshot(completed_epochs=2))

    def test_step_matched_and_full_replay_measure_each_presentation(self) -> None:
        step_matched = counter()
        step_matched.observe_batch(["c1", "c2", "o1", "o2"])
        step_matched.observe_optimizer_step(
            forward_calls=1, backward_calls=1, student_queries=0, teacher_queries=0
        )
        self.assertEqual(step_matched.snapshot(completed_epochs=1)["current_presentations"], 2)
        self.assertEqual(step_matched.snapshot(completed_epochs=1)["old_presentations"], 2)

        full = counter()
        full.observe_batch(["c1", "c2", "c3", "c4"])
        full.observe_batch(["o1", "o2", "o3"])
        full.observe_optimizer_step(
            forward_calls=1, backward_calls=1, student_queries=0, teacher_queries=0
        )
        full.observe_optimizer_step(
            forward_calls=1, backward_calls=1, student_queries=0, teacher_queries=0
        )
        snapshot = full.snapshot(completed_epochs=1)
        self.assertEqual(snapshot["current_unique_identities"], 4)
        self.assertEqual(snapshot["old_unique_identities"], 3)

    def test_agem_counts_main_and_replay_batches_actually_consumed(self) -> None:
        measured = counter()
        for main, replay in ((["c1", "c2"], ["o1"]), (["c3", "c4"], ["o2", "o3"])):
            measured.observe_batch(main)
            measured.observe_batch(replay)
            measured.observe_optimizer_step(
                forward_calls=2,
                backward_calls=2,
                student_queries=0,
                teacher_queries=0,
            )
        snapshot = measured.snapshot(completed_epochs=1)
        self.assertEqual(snapshot["current_presentations"], 4)
        self.assertEqual(snapshot["old_presentations"], 3)
        self.assertEqual(snapshot["forward_calls"], 4)
        self.assertEqual(snapshot["backward_calls"], 4)

    def test_loader_delivery_mismatch_fails_closed(self) -> None:
        measured = counter()
        with self.assertRaises(BudgetMeasurementError):
            measured.observe_batch(["unregistered-token"])
        overlapping = RunBudgetCounter(
            current_identities=frozenset({"same"}),
            old_identities=frozenset({"same"}),
        )
        with self.assertRaises(BudgetMeasurementError):
            overlapping.observe_batch(["same"])

        delivered = counter()
        delivered.observe_batch(["c1", "c1", "c2"])
        delivered.observe_optimizer_step(
            forward_calls=1, backward_calls=1, student_queries=0, teacher_queries=0
        )
        frozen_target = {
            "epochs": 1,
            "optimizer_updates": 1,
            "current_unique_identities": 3,
            "old_unique_identities": 0,
            "current_presentations": 3,
            "old_presentations": 0,
            "forward_calls": 1,
            "backward_calls": 1,
            "student_queries": 0,
            "teacher_queries": 0,
        }
        self.assertNotEqual(delivered.snapshot(completed_epochs=1), frozen_target)

    def test_transient_storage_is_nonzero_measured_and_semantically_declared(self) -> None:
        self.assertEqual(
            TRANSIENT_STORAGE_SEMANTICS,
            "maximum observed byte size of an atomic temporary artifact before final result "
            "receipt serialization",
        )
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "checkpoint.tmp"
            artifact.write_bytes(b"x" * 137)
            measured = TransientStorageCounter()
            measured.observe_temporary(artifact)
            self.assertEqual(measured.peak_atomic_temporary_bytes, 137)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import build_optimizer, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from torch.utils.data import DataLoader, Dataset

from selector_bench.training.loss_duel_feedback import loss_summary_to_metric
from selector_bench.utils.io import read_json


class TokenSubsetDataset(Dataset):
    """A dynamic token subset view over an already-built DiffusionDrive dataset."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)
        if hasattr(dataset, "flag"):
            self.flag = np.asarray(dataset.flag)[self.indices]
        else:
            self.flag = np.zeros(len(self.indices), dtype=np.uint8)
        self.CLASSES = getattr(dataset, "CLASSES", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]


def summarize_loss_values(
    losses: list[float] | np.ndarray,
    loss_key: str = "loss",
    tail_window: int = 3,
) -> dict[str, Any]:
    if tail_window < 1:
        raise ValueError("tail_window must be positive.")
    values = np.asarray(losses, dtype=np.float64)
    if values.size < 1:
        raise ValueError("losses must not be empty.")
    window = min(int(tail_window), int(values.size))
    first_mean = float(np.mean(values[:window]))
    tail_mean = float(np.mean(values[-window:]))
    best = float(np.min(values))
    delta = first_mean - tail_mean
    relative_delta = delta / max(abs(first_mean), 1e-6)
    return {
        "log_path": None,
        "loss_key": loss_key,
        "num_points": int(values.size),
        "iters": [int(idx + 1) for idx in range(int(values.size))],
        "losses": values.astype(float).tolist(),
        "first_mean": first_mean,
        "tail_mean": tail_mean,
        "best": best,
        "delta": float(delta),
        "relative_delta": float(relative_delta),
    }


def selection_tokens(selection_path: str | Path) -> list[str]:
    artifact = read_json(selection_path)
    return [str(token) for token in artifact["selection"]["selected_tokens"]]


class InProcessDiffusionDriveLossWorker:
    """Persistent train-loss worker for selector/random micro-duels.

    The DiffusionDrive model and full train dataset are built once. Before every
    selector/random arm, the planner is restored to the same base checkpoint
    state. Planner updates therefore never carry across duels; only the selector
    checkpoint is updated by the outer feedback loop.
    """

    def __init__(
        self,
        diffusiondrive_root: str | Path,
        base_config: str | Path,
        source_info: str | Path,
        planner_init_checkpoint: str | Path,
        planner_lr: float,
        max_iters: int,
        work_dir: str | Path,
        seed: int = 0,
        samples_per_gpu: int = 1,
        workers_per_gpu: int = 0,
        gpu_id: int = 0,
        base_state_device: str = "cuda",
        grad_clip_max_norm: float | None = None,
        loss_key: str = "loss",
        tail_window: int = 3,
        utility_mode: str = "drop",
    ) -> None:
        if planner_lr <= 0:
            raise ValueError("planner_lr must be positive.")
        if max_iters < 1:
            raise ValueError("max_iters must be positive.")
        if samples_per_gpu != 1:
            raise ValueError("Only samples_per_gpu=1 is currently supported.")
        if workers_per_gpu < 0:
            raise ValueError("workers_per_gpu must be non-negative.")
        if base_state_device not in {"cuda", "cpu"}:
            raise ValueError("base_state_device must be 'cuda' or 'cpu'.")

        self.diffusiondrive_root = Path(diffusiondrive_root).resolve()
        self.base_config = Path(base_config).resolve()
        self.source_info = Path(source_info).resolve()
        self.planner_init_checkpoint = Path(planner_init_checkpoint).resolve()
        self.planner_lr = float(planner_lr)
        self.max_iters = int(max_iters)
        self.work_dir = Path(work_dir).resolve()
        self.seed = int(seed)
        self.samples_per_gpu = int(samples_per_gpu)
        self.workers_per_gpu = int(workers_per_gpu)
        self.gpu_id = int(gpu_id)
        self.base_state_device = base_state_device
        self.grad_clip_max_norm = grad_clip_max_norm
        self.loss_key = str(loss_key)
        self.tail_window = int(tail_window)
        self.utility_mode = str(utility_mode)

        self._prev_cwd = Path.cwd()
        os.chdir(self.diffusiondrive_root)
        self._ensure_diffusiondrive_on_path()
        self.cfg = self._build_cfg()
        self.dataset = build_dataset(self.cfg.data.train)
        self.token_to_idx = {
            str(info["token"]): idx
            for idx, info in enumerate(getattr(self.dataset, "data_infos", []))
        }
        if not self.token_to_idx:
            raise ValueError("Built dataset does not expose data_infos tokens.")
        self.model = self._build_model()
        self.base_state = self._snapshot_base_state()

    def close(self) -> None:
        os.chdir(self._prev_cwd)

    def run_pair(
        self,
        selector_selection_path: str | Path,
        random_selection_path: str | Path,
        round_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        selector_tokens = selection_tokens(selector_selection_path)
        random_tokens = selection_tokens(random_selection_path)
        arm_seed = self.seed + int(round_idx)
        selector = self.run_arm(selector_tokens, arm_name="selector", seed=arm_seed)
        random = self.run_arm(random_tokens, arm_name="random_same_budget", seed=arm_seed)
        return selector, random

    def run_arm(self, tokens: list[str], arm_name: str, seed: int) -> dict[str, Any]:
        if not tokens:
            raise ValueError("Cannot run an empty token arm.")
        missing = [token for token in tokens if token not in self.token_to_idx]
        if missing:
            raise ValueError(f"{len(missing)} selected tokens missing from train dataset: {missing[:5]}")

        set_random_seed(int(seed), deterministic=False)
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            torch.cuda.reset_peak_memory_stats()

        self._reset_model_to_base()
        optimizer = build_optimizer(self.model, self.cfg.optimizer)
        self.model.train()

        indices = [self.token_to_idx[token] for token in tokens]
        subset = TokenSubsetDataset(self.dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers_per_gpu,
            collate_fn=partial(collate, samples_per_gpu=self.samples_per_gpu),
            pin_memory=False,
        )
        data_iter = iter(loader)
        losses: list[float] = []
        log_rows: list[dict[str, float]] = []
        start = time.monotonic()
        for iter_idx in range(self.max_iters):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                data = next(data_iter)
            optimizer.zero_grad()
            outputs = self.model.train_step(data, optimizer)
            loss = outputs["loss"]
            loss.backward()
            grad_norm = None
            if self.grad_clip_max_norm is not None and self.grad_clip_max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.grad_clip_max_norm),
                    norm_type=2,
                )
            optimizer.step()
            log_vars = {
                str(key): float(value)
                for key, value in outputs.get("log_vars", {}).items()
                if isinstance(value, (int, float))
            }
            if self.loss_key not in log_vars:
                log_vars[self.loss_key] = float(loss.detach().item())
            if grad_norm is not None:
                log_vars["grad_norm"] = float(grad_norm)
            log_vars["iter"] = float(iter_idx + 1)
            losses.append(float(log_vars[self.loss_key]))
            log_rows.append(log_vars)

        runtime = time.monotonic() - start
        summary = summarize_loss_values(losses, loss_key=self.loss_key, tail_window=self.tail_window)
        metric_value = loss_summary_to_metric(summary, mode=self.utility_mode)
        return {
            "l2": float(metric_value),
            "obj_box_col": 0.0,
            "obj_col": 0.0,
            "loss_key": self.loss_key,
            "loss_utility_mode": self.utility_mode,
            "loss_first_mean": float(summary["first_mean"]),
            "loss_tail_mean": float(summary["tail_mean"]),
            "loss_best": float(summary["best"]),
            "loss_delta": float(summary["delta"]),
            "loss_relative_delta": float(summary["relative_delta"]),
            "loss_num_points": int(summary["num_points"]),
            "loss_iters": summary["iters"],
            "loss_values": summary["losses"],
            "loss_log_rows": log_rows,
            "train_runtime_s": float(runtime),
            "runtime_s": float(runtime),
            "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if torch.cuda.is_available()
            else 0.0,
            "inprocess_worker": True,
            "planner_reset_policy": "reset_to_base_checkpoint_before_each_arm",
            "arm_name": arm_name,
            "selected_count": len(tokens),
            "max_iters": self.max_iters,
        }

    def _ensure_diffusiondrive_on_path(self) -> None:
        root = str(self.diffusiondrive_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _build_cfg(self) -> Config:
        import importlib

        cfg = Config.fromfile(str(self.base_config))
        if cfg.get("custom_imports", None):
            from mmcv.utils import import_modules_from_strings

            import_modules_from_strings(**cfg["custom_imports"])
        if hasattr(cfg, "plugin") and cfg.plugin:
            plugin_dir = getattr(cfg, "plugin_dir", "projects/mmdet3d_plugin/")
            module_path = os.path.dirname(plugin_dir).replace("/", ".").strip(".")
            if module_path:
                importlib.import_module(module_path)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        cfg.work_dir = str(self.work_dir)
        cfg.gpu_ids = [self.gpu_id]
        cfg.seed = self.seed
        cfg.data.samples_per_gpu = self.samples_per_gpu
        cfg.data.workers_per_gpu = self.workers_per_gpu
        cfg.data.train.work_dir = str(self.work_dir)
        cfg.data.train.ann_file = _diffusiondrive_ref(self.source_info, self.diffusiondrive_root)
        cfg.data.train.sequences_split_num = 1
        cfg.optimizer["lr"] = self.planner_lr
        cfg.load_from = str(self.planner_init_checkpoint)
        cfg.resume_from = None
        if self.grad_clip_max_norm is None:
            grad_clip = cfg.get("optimizer_config", {}).get("grad_clip", None)
            if grad_clip and "max_norm" in grad_clip:
                self.grad_clip_max_norm = float(grad_clip["max_norm"])
        return cfg

    def _build_model(self) -> MMDataParallel:
        model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )
        model.init_weights()
        model.CLASSES = getattr(self.dataset, "CLASSES", None)
        load_checkpoint(model, str(self.planner_init_checkpoint), map_location="cpu")
        return MMDataParallel(model.cuda(self.gpu_id), device_ids=[self.gpu_id])

    def _snapshot_base_state(self) -> dict[str, torch.Tensor]:
        state = self.model.module.state_dict()
        if self.base_state_device == "cuda":
            return {key: value.detach().clone() for key, value in state.items()}
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def _reset_model_to_base(self) -> None:
        self.model.module.load_state_dict(self.base_state, strict=True)
        self.model.zero_grad(set_to_none=True)


def _diffusiondrive_ref(path: str | Path, diffusiondrive_root: str | Path) -> str:
    target = Path(path).resolve()
    root = Path(diffusiondrive_root).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)

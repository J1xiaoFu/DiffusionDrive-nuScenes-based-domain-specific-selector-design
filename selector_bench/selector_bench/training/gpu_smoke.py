from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from selector_bench.core.artifact import make_artifact_metadata
from selector_bench.core.feature_store import FeatureStore
from selector_bench.utils.io import write_json


def run_gpu_smoke_training(
    feature_dir: str | Path,
    output_path: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    run_id: str = "gpu_smoke",
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GPU smoke training.") from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    store = FeatureStore.from_feature_dir(feature_dir, cache=True)
    tokens = store.manifest.tokens(split="train")
    x_np, y_np = _build_xy(store, tokens)

    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    dev = torch.device(device)
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, y.shape[1]),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    start = time.time()
    losses: list[float] = []
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
        device_name = torch.cuda.get_device_name(dev)
    else:
        peak_memory_mb = 0.0
        device_name = "cpu"
    elapsed_s = time.time() - start

    metrics = {
        "device": str(dev),
        "device_name": device_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train": len(tokens),
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "elapsed_s": elapsed_s,
        "peak_memory_mb": peak_memory_mb,
    }
    artifact = {
        "metadata": make_artifact_metadata(
            artifact_type="gpu_smoke_report",
            run_id=run_id,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "device": device,
            },
            input_paths=[Path(feature_dir) / "manifest.jsonl"],
            metrics=metrics,
            summary={
                "final_loss": metrics["final_loss"],
                "peak_memory_mb": peak_memory_mb,
                "elapsed_s": elapsed_s,
            },
        ),
        "loss_history": losses,
    }
    write_json(output_path, artifact)
    return artifact


def _build_xy(store: FeatureStore, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for token in tokens:
        record = store.manifest.get(token)
        features = store.get(token)
        traj = features.traj_raw
        speed_col = traj[:, 2] if traj.ndim == 2 and traj.shape[1] > 2 else traj.reshape(-1)
        scalar = np.asarray(
            [
                record.teacher_loss,
                record.teacher_uncertainty,
                record.agent_count / 32.0,
                record.ego_speed / 30.0,
                record.domain_id / 10.0,
                np.mean(np.abs(speed_col)),
            ],
            dtype=np.float32,
        )
        row = np.concatenate(
            [
                features.scene_hidden.astype(np.float32),
                features.interaction_hidden.astype(np.float32),
                scalar,
            ]
        )
        rows.append(row)
        targets.append(
            np.asarray(
                [
                    traj[-1, 0] - traj[0, 0],
                    traj[-1, 1] - traj[0, 1],
                    speed_col[-1],
                ],
                dtype=np.float32,
            )
        )
    x = np.stack(rows)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    y = np.stack(targets)
    return x, y

#!/usr/bin/env python3
"""Compute a per-example diagonal importance proxy for Drive-CL EWC.

The old-log buffer and source checkpoint are hashed into the artifact.  The
model remains in evaluation mode so Fisher estimation cannot mutate source
BatchNorm buffers.  This is a preprocessing cost and is reported separately
from stage optimizer steps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("drive_cl_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, choices=(2, 3), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rap-root", type=Path, default=Path("/home/rguo/RAP-main/RAP-main"))
    parser.add_argument(
        "--cache", type=Path, default=Path("/home/rguo/rap_workspace/exp/training_cache")
    )
    parser.add_argument("--buffer-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument("--max-batches", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--schedule-epochs", type=int, default=180)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()
    for path in (args.protocol_manifest, args.source_checkpoint, args.rap_root, args.cache):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if min(args.buffer_size, args.batch_size, args.max_batches) <= 0 or args.workers < 0:
        parser.error("invalid buffer/batch/worker setting")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bind exactly one idle GPU through /home/rguo/bin/gpu-run")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.rap_root.resolve()))
    from navsim.planning.training.dataset import CacheOnlyDataset
    from selector_bench.continual.baselines import (
        EWC_ESTIMATOR,
        DiagonalFisherAccumulator,
    )

    runner = load_runner(Path(__file__).with_name("41_train_drive_cl_diffusiondrive.py"))
    manifest = json.loads(args.protocol_manifest.read_text())
    old_tokens = [
        token
        for stage in range(1, args.stage_index)
        for token in runner.stage_tokens(manifest, stage, "train")
    ]
    selected = sorted(
        old_tokens,
        key=lambda token: hashlib.sha256(
            f"{args.seed}\0ewc-buffer\0{token}".encode()
        ).digest(),
    )[: args.buffer_size]
    if not selected:
        raise RuntimeError("Fisher buffer is empty")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    agent = runner.build_agent(args.rap_root, args.learning_rate).to(device)
    optimizer, scheduler = runner.build_optimizer_scheduler(agent, args)
    runner.restore_training_state(args.source_checkpoint, agent, optimizer, scheduler)
    dataset = CacheOnlyDataset(
        args.cache,
        agent.get_feature_builders(),
        agent.get_target_builders(),
        use_cache_list=True,
    )
    selected_dataset = runner.TokenDataset(dataset, selected)
    order = torch.Generator().manual_seed(
        runner.stable_seed("ewc-fisher", args.seed, args.stage_index)
    )
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=order,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    agent.eval()
    accumulator = DiagonalFisherAccumulator(agent)
    for batch_index, (features, targets) in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        seed = runner.stable_seed("ewc-loss", args.seed, args.stage_index, batch_index)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        features = runner.move_to_device(features, device)
        targets = runner.move_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = agent.forward(features, targets)
        loss = agent.compute_loss(features, targets, predictions)["loss"]
        loss.backward()
        batch_size = int(next(iter(features.values())).shape[0])
        accumulator.add(agent, batch_size)
        print(
            json.dumps(
                {
                    "event": "fisher_batch",
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "loss": float(loss.detach().item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    state = accumulator.finalize(agent)
    payload = {
        "schema": "selector_bench.drive_cl_ewc_importance.v2",
        "protocol_content_sha256": manifest.get("content_sha256"),
        "stage_index": args.stage_index,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": runner.sha256_file(args.source_checkpoint),
        "sample_count": state.sample_count,
        "buffer_token_pool": selected,
        "fisher": state.fisher,
        "estimator": EWC_ESTIMATOR,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(args.output.parent).f_bavail * os.statvfs(args.output.parent).f_frsize
    if free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free before Fisher write")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "event": "result",
                "output": str(args.output.resolve()),
                "sha256": runner.sha256_file(args.output),
                "sample_count": state.sample_count,
                "parameters": len(state.fisher),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


REPLACEMENTS = {
    "version = 'trainval'": "version = 'mini'",
    "total_batch_size = 48": "total_batch_size = 1",
    "num_gpus = 8": "num_gpus = 1",
    "num_epochs = 10": "num_epochs = 1",
    "checkpoint_epoch_interval = 10": "checkpoint_epoch_interval = 1",
    "use_deformable_func = True": "use_deformable_func = False",
}


SMOKE_APPEND = """

# ================== selector_bench single-GPU smoke overrides ========================
# Generated for code-level and environment validation on one RTX 4090D.
# Keep this file separate from the official config.
data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 2
runner = dict(type="IterBasedRunner", max_iters=200)
evaluation = dict(interval=200, eval_mode=eval_mode)
checkpoint_config = dict(interval=200)
log_config["interval"] = 10
load_from = None
resume_from = None
model["img_backbone"]["pretrained"] = None
"""


def make_smoke_config(
    source_config: str | Path,
    output_config: str | Path,
    max_iters: int = 200,
    workers_per_gpu: int = 2,
) -> Path:
    source = Path(source_config)
    output = Path(output_config)
    text = source.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    append = SMOKE_APPEND.replace("max_iters=200", f"max_iters={max_iters}")
    append = append.replace("interval=200", f"interval={max_iters}")
    append = append.replace('log_config["interval"] = 10', 'log_config["interval"] = 10')
    append = append.replace('data["workers_per_gpu"] = 2', f'data["workers_per_gpu"] = {workers_per_gpu}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + append + "\n", encoding="utf-8")
    return output

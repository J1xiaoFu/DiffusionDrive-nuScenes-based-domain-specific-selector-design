# Agent Status

Last updated: 2026-07-08

This file is the handoff state for future agents. It intentionally summarizes
older modification logs; historical details remain in `selector_bench/reports/`,
run work directories, and generated JSON reports.

## Long-Term Goal

Implement and validate a two-stage domain/sample selector for DiffusionDrive on
nuScenes under one RTX 4090D:

- Stage 1: cold-start selector from real DiffusionDrive teacher signals.
- Stage 2: feedback optimization from same-budget selector-vs-random
  DiffusionDrive comparisons.
- Use small/cheap feedback often, but decide quality with real DiffusionDrive
  validation against multiple random seeds.
- Scale from controlled Trainval partials to larger/full Trainval only after the
  partial loop has stable multi-seed evidence and a defensible update rule.

## Current Short-Term Goal

The project is no longer asking whether Stage 2 contains signal. The latest
controlled experiments show that selector-vs-random feedback is learnable:
with high-frequency 8-sample loss-duels and `selector_lr=0.01`, the aggregate
selector selection beat 4/4 random aggregate baselines at r40.

The current goal is to improve how the reinforcement signal is used. Focus on
reward density, candidate retirement/replay, selector update strength,
loss-proxy reliability, and periodic real validation. Do not resume the older
strategy of simply incrementing one-shot v5/v6/v7 selector versions.

## Hardware And Data

- Workspace: `/root/autodl-tmp/domain_selector`
- DiffusionDrive repo: `/root/autodl-tmp/domain_selector/DiffusionDrive`
- Selector scaffold: `/root/autodl-tmp/domain_selector/selector_bench`
- GPU: one NVIDIA GeForce RTX 4090D, 24 GB VRAM
- Full raw nuScenes source:
  `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`
- Controlled partial data root:
  `/root/autodl-tmp/domain_selector/DiffusionDrive/data/nuscenes_trainval_partial_20260706`
- Train info:
  `/root/autodl-tmp/domain_selector/DiffusionDrive/data/infos/trainval_partial_20260706/nuscenes_partial_1600_400_infos_train.pkl`
- Val info:
  `/root/autodl-tmp/domain_selector/DiffusionDrive/data/infos/trainval_partial_20260706/nuscenes_partial_1600_400_infos_val.pkl`
- Train split: 40 scenes / 1611 samples
- Val split: 10 scenes / 401 samples
- Mini data is integration/debugging only; it is not selector-quality evidence.

## Validation Boundary

- Synthetic anchors in `DiffusionDrive/data/kmeans` are smoke-only.
- Mock/smoke feature caches are not selector-quality evidence.
- Current meaningful evidence is the real Trainval partial, still with 100-iter
  planning-only DiffusionDrive fine-tuning/eval.
- Exact-token DiffusionDrive subset configs should set `sequences_split_num=1`
  to avoid empty sequence-split assertions on sparse scene fragments.
- Use standalone eval via `selector_bench/scripts/18_eval_diffusiondrive_checkpoint.py`;
  do not rely on train-time EvalHook.
- Treat train-loss proxy rewards as cheap high-frequency hints. Real aggregate
  validation remains the deciding metric.

## Current Architecture

Important directories:

| Path | Role |
| --- | --- |
| `DiffusionDrive/` | Upstream DiffusionDrive repo plus local data/config patches. |
| `selector_bench/scripts/` | CLI entry points for data prep, selection, feedback, DiffusionDrive train/eval, and reports. |
| `selector_bench/selector_bench/` | Python package: feature store, selector training, high-frequency Stage 2 loop, DiffusionDrive wrappers, utilities. |
| `selector_bench/artifacts/` | Generated configs, feature caches, selector checkpoints, selections, stage targets, episode buffers, and reports. |
| `selector_bench/work_dirs/` | Heavy DiffusionDrive outputs: checkpoints, logs, raw eval outputs, status JSONs. |

Current Stage 2 loop:

1. Start from the stage1 base checkpoint
   `/root/autodl-tmp/domain_selector/DiffusionDrive/ckpts/sparsedrive_stage1.pth`.
2. Keep a resident DiffusionDrive loss worker in process.
3. For each micro-round, selector picks 8 samples, random picks a disjoint 8
   samples, and both arms run short 8-iteration train-loss feedback.
4. Reset planner weights back to the base checkpoint before each arm. Do not
   preserve planner updates across duels.
5. Update the selector immediately from the accumulated selector-vs-random
   feedback episodes.
6. Remove selected selector tokens from later candidate pools during the current
   cycle.
7. After a planned cutoff, aggregate selected selector tokens and run real
   100-iter DiffusionDrive fine-tuning/eval against one or more random
   aggregate baselines.

Recent code hardening:

- `selector_bench/selector_bench/utils/io.py`: JSON writes are atomic
  temp-file replaces.
- `selector_bench/selector_bench/training/stage2_high_frequency.py`: per-run
  lock and resume behavior that reuses existing episode records instead of
  duplicate appends.
- `selector_bench/selector_bench/training/stage2_neural.py`: duplicate
  `episode_id` rows are ignored when loading feedback episodes.

## Key Scripts

| Purpose | Script |
| --- | --- |
| Prepare partial nuScenes data | `selector_bench/scripts/26_prepare_nuscenes_partial.py` |
| Create partial DiffusionDrive infos | `selector_bench/scripts/27_create_diffusiondrive_partial_infos.py` |
| Generate real anchors | `selector_bench/scripts/17_make_diffusiondrive_real_mini_anchors.py` |
| Extract teacher/cache features | `selector_bench/scripts/02_extract_diffusiondrive_features.py` |
| Train neural selector | `selector_bench/scripts/21_train_neural_selector.py` |
| Random/selector baseline selection | `selector_bench/scripts/06_run_baseline_selection.py` |
| Export DiffusionDrive subset info | `selector_bench/scripts/19_export_diffusiondrive_info_subset.py` |
| Train DiffusionDrive checkpoint | `selector_bench/scripts/20_train_diffusiondrive_checkpoint.py` |
| Standalone eval | `selector_bench/scripts/18_eval_diffusiondrive_checkpoint.py` |
| Record feedback episode | `selector_bench/scripts/22_record_stage2_episode.py` |
| Train Stage 2 feedback selector | `selector_bench/scripts/25_train_neural_selector_stage2.py` |
| High-frequency in-process loss-duel loop | `selector_bench/scripts/37_run_stage2_loss_duel_inprocess.py` |
| Aggregate real validation from micro-rounds | `selector_bench/scripts/38_run_stage2_aggregate_selected_real_eval.py` |
| Random aggregate variance check | `selector_bench/scripts/39_run_stage2_aggregate_random_variance.py` |

## Latest Evidence

All rows below use the stage1 base checkpoint, planner lr `3e-5`, and 100-iter
aggregate real validation unless noted. Lower L2 is better.

| Experiment | Selector L2 | Random L2 | Result |
| --- | ---: | ---: | --- |
| Original stage1 base, no fine-tuning | see `stage1_init_partial100_selector3_20260707_report.json` | n/a | reference only |
| Full-data 1000+ sample fine-tuning sweeps | see `stage1_init_partial100_selector3_20260707_report.json` | n/a | learning-rate context |
| High-frequency r10 aggregate, `selector_lr=0.01` | 6.820722 | 6.616966 | selector worse |
| High-frequency r20 aggregate, `selector_lr=0.01` | 7.276919 | 7.003900 | selector worse |
| High-frequency r30 aggregate, `selector_lr=0.01` | 6.606814 | 6.777496 | selector better |
| High-frequency r40 aggregate, `selector_lr=0.01` | 6.309916 | 6.875033 | selector better |
| r40 random variance seeds 7200/7201/7202 | 6.309916 | mean 6.654263 | selector better 3/3 |
| r40 combined random seeds 7140/7200/7201/7202 | 6.309916 | mean 6.709456 | selector better 4/4 |
| r40 `selector_lr=0.003` ablation | 7.280338 | 6.605227 | selector worse |

Interpretation:

- Stage 2 random-comparison feedback contains learnable signal. The r40
  `selector_lr=0.01` aggregate result is stable across multiple random seeds.
- The signal is not automatically useful under every RL update. Lowering
  selector lr to `0.003` reduced micro-round performance and failed in real
  aggregate validation.
- Loss-duel rewards are noisy and can disagree with real validation; keep using
  real aggregate eval as the final arbiter.

## Important Artifacts

| Artifact | Path / Notes |
| --- | --- |
| Stage1 base checkpoint now used for experiments | `/root/autodl-tmp/domain_selector/DiffusionDrive/ckpts/sparsedrive_stage1.pth` |
| Stage1/full-finetune reference report | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage1_init_selector_experiments/stage1_init_partial100_selector3_20260707_report.json` |
| Successful loss-duel run | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_loss_duel_inprocess/stage2_loss_duel_inprocess_stage1_lr3e5_r40x8_train_test_20260707_report.json` |
| Successful loss-duel episodes | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_loss_duel_inprocess/stage2_loss_duel_inprocess_stage1_lr3e5_r40x8_train_test_20260707_episodes.jsonl` |
| Successful r10/r20/r30 aggregate report | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_aggregate_selected_real_eval/stage2_loss_duel_aggregate_progression_real_eval_lr3e5_100iter_20260707_report.json` |
| Successful r40 aggregate report | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_aggregate_selected_real_eval/stage2_loss_duel_aggregate320_real_eval_lr3e5_100iter_20260707_report.json` |
| r40 random variance report | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_aggregate_random_variance/stage2_loss_duel_r40_random_variance_lr3e5_100iter_20260708_report.json` |
| Failed `selector_lr=0.003` loss-duel report | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_loss_duel_inprocess/stage2_loss_duel_inprocess_stage1_lr3e5_r40x8_selectorlr3e3_clean_20260708_report.json` |
| Failed `selector_lr=0.003` aggregate report | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_aggregate_selected_real_eval/stage2_loss_duel_aggregate320_selectorlr3e3_clean_real_eval_lr3e5_100iter_20260708_report.json` |

Older v0-v7/rho100 one-shot Stage 2 artifacts remain useful historical
context, especially for negative-feedback and collision-aware reward analysis,
but they are not the current primary path.

## Timing Notes

- The in-process worker is being used by `37_run_stage2_loss_duel_inprocess.py`;
  it avoids repeated full DiffusionDrive initialization.
- A micro-round still trains two arms: selector 8 samples plus random 8 samples.
- In the `selector_lr=0.003` clean run, each arm averaged about 8.8-9.0s and a
  two-arm round averaged about 17.7s.
- Real validation remains the main wall-clock bottleneck because it runs
  100-iter fine-tuning and standalone validation-set evaluation.

## Recommended Next Work

1. Keep the successful `selector_lr=0.01` high-frequency setup as the current
   baseline. Do not adopt the `0.003` ablation.
2. Run update-rule ablations near the successful setting, not just lower lr: try
   reward clipping/normalization, entropy or score regularization, and smaller
   per-update epochs if selector overreacts.
3. Formalize candidate retirement/replay: selected selector tokens leave the
   current candidate pool; consider controlled re-entry after coverage reaches a
   threshold.
4. Add periodic aggregate eval checkpoints for promising runs, e.g. r10/r20/r30/r40,
   and keep all intermediate metrics.
5. Continue multi-seed random aggregate checks for any claimed improvement.
6. Do not scale to larger/full Trainval until the controlled partial loop has a
   clearer Stage 2 update rule and repeatable multi-seed gains.

## Verification Status

Recent checks and runs:

- `py_compile` passed for the updated high-frequency loop, feedback loader, IO
  helper, and script 37.
- The clean `selector_lr=0.003` 40x8 loss-duel run completed with 40 JSONL rows
  and no duplicate episode rows.
- The `selector_lr=0.003` aggregate real eval completed and confirmed the
  ablation is worse than random.
- The successful `selector_lr=0.01` r40 aggregate has been checked against four
  random aggregates total and remains better than all four.

## Prompt For Next Agent

Continue `/root/autodl-tmp/domain_selector`. Read `README.md` and
`AGENT_STATUS.md` first. The current primary direction is high-frequency Stage 2
loss-duel reinforcement learning, not the older v5/v6/v7 one-shot candidate
loop. The best current run is `selector_lr=0.01`, 40 rounds x 8 selector samples,
stage1 base checkpoint `/root/autodl-tmp/domain_selector/DiffusionDrive/ckpts/sparsedrive_stage1.pth`,
planner lr `3e-5`, and r40 aggregate real eval. It achieved selector L2
`6.309916`, beating the paired random aggregate L2 `6.875033` and three
additional random seeds with mean random L2 `6.654263`; combined, selector beat
4/4 random aggregates. The `selector_lr=0.003` ablation lost (`selector L2
7.280338`, random L2 `6.605227`) and should not replace the default. Next work
should explore better RL signal usage: reward shaping, update stability,
selected-token retirement/replay, and periodic aggregate eval. Do not add an
external dataset or scale to larger/full nuScenes Trainval until the controlled
partial loop has a clearer update rule and repeatable multi-seed gains.

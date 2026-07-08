# Domain Selector for DiffusionDrive

This workspace contains a lightweight selector-benchmark scaffold around the
DiffusionDrive nuScenes codebase. The project goal is to build and validate a
two-stage sample/domain selector on one RTX 4090D, using real DiffusionDrive
training/evaluation feedback rather than proxy-only signals.

Current evidence has moved the project from "is there a Stage 2 signal?" to
"how should we use that signal?" On the controlled Trainval partial, aggregate
Stage 2 selector selections beat multiple same-budget random baselines after
high-frequency selector/random loss-duel updates. This is meaningful evidence
that the selector can learn from random-comparison feedback. The open research
problem is now the reinforcement-learning/update design: reward density,
candidate retirement/replay, selector learning rate, proxy-loss reliability,
and when to spend expensive real validation runs.

Start every handoff by reading `AGENT_STATUS.md`; it is the compact current
state. This README explains the project architecture and normal workflow.

## Repository Layout

| Path | Role |
| --- | --- |
| `AGENT_STATUS.md` | Current handoff state, active goals, latest results, and next-agent prompt. |
| `DiffusionDrive/` | Upstream DiffusionDrive repository plus local environment/data/config patches needed to run on this machine. |
| `selector_bench/` | Selector scaffold: feature stores, selector training, subset export, DiffusionDrive wrappers, tests, reports, and generated artifacts. |
| `selector_bench/scripts/` | CLI entry points for data prep, feature extraction, selector training, DiffusionDrive train/eval, and feedback recording. |
| `selector_bench/selector_bench/` | Python package with feature store, manifest, selector training, domain clustering, and nuScenes partial-data helpers. |
| `selector_bench/artifacts/` | Generated configs, feature caches, selector checkpoints, selections, stage targets, subset metadata, and episode buffers. |
| `selector_bench/work_dirs/` | DiffusionDrive training/eval work directories, checkpoints, logs, and status JSONs. |
| `selector_bench/reports/` | Experiment tables and machine-readable preflight/report outputs. |
| `DiffusionDrive/data/` | Materialized nuScenes mini/partial data, info PKLs, anchors, and subset info PKLs. |
| `meeting_materials/weekly_stage2_feasibility_20260709/` | Lightweight tracked summary of the current evidence: tables, figures, slide outline, and copied JSON reports. |

## GitHub Tracking Scope

This repository is a source and experiment-summary repository, not a runtime
snapshot of the AutoDL workspace.

Tracked content should stay limited to:

- root handoff docs: `README.md` and `AGENT_STATUS.md`;
- selector source, scripts, configs, tests, env descriptors, and lightweight
  reports under `selector_bench/`;
- DiffusionDrive source/config/docs/assets plus the local code patches required
  by this project;
- small experiment summaries under `meeting_materials/weekly_stage2_feasibility_20260709/`.

The following are intentionally ignored and should not be pushed to GitHub:

- local environments such as `.conda/`;
- nuScenes materializations under `DiffusionDrive/data/`;
- model checkpoints under `DiffusionDrive/ckpt/`, `DiffusionDrive/ckpts/`, and
  top-level `*.pth` files;
- heavy DiffusionDrive outputs under `selector_bench/work_dirs/`;
- generated selector caches/checkpoints under `selector_bench/artifacts/`;
- Python caches, TensorBoard logs, compiled CUDA extensions, and compressed
  bundles.

Important generated artifacts are summarized in `AGENT_STATUS.md` and copied
into `meeting_materials/weekly_stage2_feasibility_20260709/raw_reports/` when
they are needed for handoff or presentation. Regenerate heavy artifacts from
the scripts rather than versioning them.

## Conceptual Workflow

The selector experiment is a closed loop:

1. Prepare a real nuScenes data slice for DiffusionDrive.
2. Generate real DiffusionDrive anchors for that slice.
3. Train or load a small DiffusionDrive teacher checkpoint.
4. Extract per-sample teacher features/loss/uncertainty into a feature cache.
5. Build Stage 1 cold-start targets and train a neural selector.
6. Sample a selector subset and a same-budget random subset with matched domain
   counts.
7. Export both selections to DiffusionDrive train-info PKLs.
8. Train DiffusionDrive checkpoints on each subset with the same budget.
9. Evaluate both checkpoints through standalone `tools/test.py`.
10. Record a Stage 2 feedback episode with selector/random metrics.
11. Fine-tune the selector from accumulated episodes and sample the next
    candidate.
12. For high-frequency Stage 2 experiments, use the in-process loss-duel worker:
    repeatedly select a small selector subset, compare it against a disjoint
    random subset, update the selector immediately, and later aggregate the
    selected tokens for a real DiffusionDrive validation run.
13. Repeat until the selector-vs-random evidence is stable enough to justify a
    larger data scale.

## Important Validation Rules

- Synthetic anchors under `DiffusionDrive/data/kmeans` are smoke-only.
- Mock feature caches are smoke-only.
- Mini experiments are useful for integration and debugging but are not robust
  selector-quality evidence.
- Current real evidence comes from the controlled Trainval partial, but it is
  still 100-iter, planning-only evidence.
- Exact-token subset configs should use `sequences_split_num=1`, because sparse
  scene fragments can trigger empty sequence-split assertions.
- Prefer standalone eval with
  `selector_bench/scripts/18_eval_diffusiondrive_checkpoint.py`; it avoids the
  train-time EvalHook/TextLogger path that previously caused logger errors.

## Current Data Scale

Current controlled Trainval partial:

- Source: `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval`
- Data root: `/root/autodl-tmp/domain_selector/DiffusionDrive/data/nuscenes_trainval_partial_20260706`
- Train info: `DiffusionDrive/data/infos/trainval_partial_20260706/nuscenes_partial_1600_400_infos_train.pkl`
- Val info: `DiffusionDrive/data/infos/trainval_partial_20260706/nuscenes_partial_1600_400_infos_val.pkl`
- Train: 40 scenes / 1611 samples
- Val: 10 scenes / 401 samples
- Blob extraction: 41670 required files found and extracted
- Real anchors: `DiffusionDrive/data/kmeans_trainval_partial_20260706/`

Do not introduce an external dataset yet. Scale within nuScenes Trainval first.

## Core Scripts

| Task | Script |
| --- | --- |
| Environment/preflight checks | `selector_bench/scripts/00_check_env.py`, `11_preflight_limited_hardware.py` |
| Prepare Trainval partial data | `selector_bench/scripts/26_prepare_nuscenes_partial.py` |
| Build partial DiffusionDrive infos | `selector_bench/scripts/27_create_diffusiondrive_partial_infos.py` |
| Create real anchors | `selector_bench/scripts/17_make_diffusiondrive_real_mini_anchors.py` |
| Extract teacher features | `selector_bench/scripts/02_extract_diffusiondrive_features.py` |
| Fit cluster/domain labels | `selector_bench/scripts/23_fit_stage1_domains.py` |
| Build Stage 1 target scores | `selector_bench/scripts/24_build_stage1_domain_targets.py` |
| Train Stage 1 neural selector | `selector_bench/scripts/21_train_neural_selector.py` |
| Run random/selector selection | `selector_bench/scripts/06_run_baseline_selection.py` |
| Export selection to train-info PKL | `selector_bench/scripts/19_export_diffusiondrive_info_subset.py` |
| Train DiffusionDrive checkpoint | `selector_bench/scripts/20_train_diffusiondrive_checkpoint.py` |
| Standalone DiffusionDrive eval | `selector_bench/scripts/18_eval_diffusiondrive_checkpoint.py` |
| Record feedback episode | `selector_bench/scripts/22_record_stage2_episode.py` |
| Fine-tune Stage 2 selector | `selector_bench/scripts/25_train_neural_selector_stage2.py` |
| Report Stage 2 collision sensitivity | `selector_bench/scripts/28_report_stage2_collision_sensitivity.py` |
| Reweight Stage 2 feedback rewards | `selector_bench/scripts/29_reweight_stage2_feedback.py` |
| Diagnose Stage 2 feedback generalization | `selector_bench/scripts/30_diagnose_stage2_feedback_generalization.py` |
| Score Stage 2 candidates with fold ensemble gate | `selector_bench/scripts/31_score_stage2_candidates.py` |
| Run high-frequency in-process loss-duels | `selector_bench/scripts/37_run_stage2_loss_duel_inprocess.py` |
| Real-eval aggregate selected micro-rounds | `selector_bench/scripts/38_run_stage2_aggregate_selected_real_eval.py` |
| Evaluate random-seed variance for an aggregate selector | `selector_bench/scripts/39_run_stage2_aggregate_random_variance.py` |

## Artifact Types

| Directory | Contents |
| --- | --- |
| `selector_bench/artifacts/features/` | Feature caches with `manifest.jsonl`, NPZ features, and `feature_stats.json`. |
| `selector_bench/artifacts/stage1_targets/` | Domain labels, cluster models, profile reports, and target-score JSONL files. |
| `selector_bench/artifacts/selectors/` | Selector checkpoint JSON artifacts. |
| `selector_bench/artifacts/selections/` | Selection JSON artifacts with selected tokens, hashes, domain budgets, and replay metadata. |
| `selector_bench/artifacts/diffusiondrive_configs/` | Generated DiffusionDrive Python configs for base, subset, random, and feedback runs. |
| `selector_bench/artifacts/diffusiondrive_subsets/` | Sidecars describing exported train-info subsets. |
| `selector_bench/artifacts/stage2_feedback/` | Feedback episode JSONL buffers and schema. |
| `selector_bench/work_dirs/` | Heavy run outputs: checkpoints, logs, eval outputs, and status JSON files. |

## Current Selector Loop Status

See `AGENT_STATUS.md` for the authoritative up-to-date values. In brief:

- Stage 1 cold-start on the current partial lost to random, so the useful path
  is not a one-shot supervised selector.
- Earlier full-eval Stage 2 v0-v6 experiments were noisy: some selector
  versions beat a paired random seed, later versions regressed, and negative
  feedback generalization was weak.
- The current strongest evidence comes from high-frequency Stage 2 loss-duels
  followed by aggregate real validation. With `selector_lr=0.01`, 40 rounds of
  8-sample selector-vs-random comparisons selected 320 unique selector tokens.
  The aggregate selector achieved `L2=6.309916` after 100-iter DiffusionDrive
  fine-tuning, beating the paired random aggregate (`L2=6.875033`) and three
  additional random-320 seeds (`mean L2=6.654263`, min `6.565195`, max
  `6.818086`). Selector beat 4/4 tested random aggregates.
- A lower selector learning rate ablation (`selector_lr=0.003`) failed:
  loss-duel mean reward was `-2.636652`, selector won only 15/40 micro-rounds,
  and the aggregate real eval lost to random (`selector L2=7.280338`, random
  L2=`6.605227`).
- The result supports the core Stage 2 hypothesis: selector-vs-random feedback
  contains learnable signal. It does not yet settle the best reinforcement
  learning update rule.
- Current recommended selector setting is the high-frequency loss-duel setup
  with `selector_lr=0.01`, fixed reset to
  `DiffusionDrive/ckpts/sparsedrive_stage1.pth`, planner lr `3e-5`, and
  aggregate real validation at 100 planner iterations.
- Current production-grade in-process loop keeps the DiffusionDrive worker
  resident, resets planner weights to the base checkpoint before each arm, and
  does not preserve planner updates across duels.
- Main successful loss-duel report:
  `selector_bench/artifacts/stage2_loss_duel_inprocess/stage2_loss_duel_inprocess_stage1_lr3e5_r40x8_train_test_20260707_report.json`.
- Successful aggregate r40 report:
  `selector_bench/artifacts/stage2_aggregate_selected_real_eval/stage2_loss_duel_aggregate320_real_eval_lr3e5_100iter_20260707_report.json`.
- Random-seed variance report:
  `selector_bench/artifacts/stage2_aggregate_random_variance/stage2_loss_duel_r40_random_variance_lr3e5_100iter_20260708_report.json`.
- Failed `selector_lr=0.003` ablation report:
  `selector_bench/artifacts/stage2_aggregate_selected_real_eval/stage2_loss_duel_aggregate320_selectorlr3e3_clean_real_eval_lr3e5_100iter_20260708_report.json`.
- Older v3-v7/rho100 artifacts and reports remain useful history for diagnosing
  negative feedback, collision-aware rewards, and candidate gating, but they are
  no longer the primary path forward.

## Recommended Next Step

Continue from the high-frequency Stage 2 path, not the older v5/v6/v7 one-shot
candidate loop. The next work should improve how the reinforcement signal is
used:

1. Keep validating against multiple random seeds; single random comparisons are
   not enough.
2. Explore reward shaping and update stability around the successful
   `selector_lr=0.01` setting instead of lowering directly to `0.003`.
3. Test candidate retirement/replay policies: selected selector tokens should
   leave the candidate pool for the current cycle, with controlled re-entry only
   after enough coverage has accumulated.
4. Treat train-loss proxy rewards as cheap, high-frequency hints, not final
   evidence. Real aggregate validation remains the deciding metric.
5. Consider periodic aggregate checkpoints, e.g. r10/r20/r30/r40, because the
   r40 aggregate was best in the current successful run but intermediate
   behavior matters for diagnosing update drift.

Do not introduce an external dataset yet. Do not scale to larger/full nuScenes
Trainval until the controlled partial loop has stable multi-seed evidence and a
clearer Stage 2 update rule.

## Testing And Sanity Checks

Common checks from the workspace root:

```bash
/root/autodl-tmp/domain_selector/.conda/diffusiondrive_py39/bin/python3.9 -m py_compile <files>
```

From `selector_bench/`:

```bash
/root/autodl-tmp/domain_selector/.conda/diffusiondrive_py39/bin/python3.9 -m unittest tests.test_nuscenes_partial tests.test_stage2_neural
/root/autodl-tmp/domain_selector/.conda/diffusiondrive_py39/bin/python3.9 -m unittest discover -s tests -p 'test_*.py'
```

Most recent known downstream result: high-frequency Stage 2 aggregate
`selector_lr=0.01` beat 4/4 random aggregate comparisons at r40. The
`selector_lr=0.003` ablation lost to random and should not replace the current
default. Recent code hardening added atomic JSON writes, a per-run lock for
high-frequency Stage 2 runs, and duplicate-episode filtering for feedback
training.

## Operational Notes

- Do not overwrite previous work dirs unless deliberately creating a rerun.
- New DiffusionDrive train/eval runs should use fresh, descriptive work dirs.
- Keep selection hashes in notes; they are the easiest way to match selector
  artifacts to subset/eval results.
- Record every selector-vs-random comparison as a Stage 2 episode JSONL row.
- Keep collision metric units straight: eval logs display percentages, episode
  JSONL stores fractions.

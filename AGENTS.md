# Drive-OPD ICLR 2027 Controller Contract

This file applies to the controller worktree and branch only.

## Identity and ownership

- Controller task: `01a00dab-5dc6-7413-8ca7-930353db0e0c` on 07G.
- Controller branch: `codex/iclr2027-drive-opd-controller`.
- Controller worktree: `/home/khwang/domain-selector-controller`.
- Dual-server dispatch revision branch: `codex/iclr2027-dual-server-controller`; it is a controller
  candidate branch derived from `3b6523de2f06f912e6e7834a0a5ce4166864f544`, not the original
  controller checkout.
- 07G research branch/worktree: `codex/iclr2027-drive-opd-07g-research` at
  `/home/khwang/domain-selector`.
- 06G execution task: `01a00e84-1f10-7331-ac52-fbc0c5d75a5a`.
- 06G execution branch: `codex/iclr2027-diffusiondrive-06g`.
- `01a00f54-8adc-7372-a8c5-bb3efe33fbef` is the completed P001 blind-review task. It is
  evidence, not the controller task and not an execution task.

## Controller scope

The controller branch owns only:

- cross-server plans and responsibility boundaries;
- prospective protocol/config/metric/hypothesis freezes;
- remote receipt intake, hash audit and promotion decisions;
- review decisions, paper milestone indices and user-audit requests;
- publication refs used by execution hosts.

Do not develop server-specific training code, write checkpoints/cache/logs, or launch GPU jobs from
this worktree. Those changes belong on the 07G research or 06G execution branch and enter the
controller only through reviewed manifests and receipts.

The controller schedules two independent claim lanes:

- 06G owns the primary pristine DiffusionDrive/NAVSIM continual-learning evidence.
- 07G owns Drive-OPD method research and a separate DiffusionDrive/nuScenes mechanism and external-
  validity lane. nuScenes results never substitute for, pool with, or unlock the NAVSIM main table.

Both servers may use computation after their own dataset-specific gates pass. A pass on one dataset
does not authorize the other dataset. The authoritative schedule and baseline tiers are frozen in
`ICLR2027_DUAL_SERVER_EXECUTION_PLAN_20260817.md`.

## Current hard gates

06G R1 commit `1bc9cc2957eeccc6927f3dc826135ce85bfae790` passes only reconstruction of
the official 103,288-token NAVSIM navtrain population and its official 85,109/18,179 train/validation
partition. It does not pass or authorize:

- a claim-bearing training cache;
- Failure-Patch-CL or Chronological-CL stage membership;
- T0 or any GPU gate/training;
- Drive-OPD/LwF performance evaluation.

The next authorization requires a controller-frozen session-atomic CL manifest, a published source
ref, an independently replayed cache-builder receipt and the real-adapter OPD/LwF budget preflight.

The 07G candidate `9dabf42371e8dde72e1b2061bac4fa9d11b34230` is not released. Its fresh
context-free review scored 3/10 reject and found executable identity/evidence bypasses involving seed,
CSV/checkpoint pairing, method identity, evaluation-domain registration, filename discovery and
duplicate receipts. Until a new frozen review passes those P0 items, 07G may perform source fixes,
CPU tests, resource inventory and nuScenes provenance/split audits only. It may not start claim-bearing
nuScenes cache, T0, training or GPU evaluation.

The NAVSIM and nuScenes gates are independent but fail closed. Historical nuScenes partial results and
the historical 14,951-token NAVSIM cache remain diagnostic only.

## Git and evidence discipline

- Begin and end every controller change by checking branch, HEAD and `git status --short`.
- Keep one auditable purpose per commit; never leave source/config/receipt files untracked.
- Preserve remote evidence verbatim when possible and verify its advertised SHA256 locally.
- Record evidence boundaries explicitly; a source-population receipt must never be promoted into a
  cache, training or performance claim.
- Do not merge, rebase, rewrite history, delete large artifacts, or unseal final-test data without
  explicit human audit.
- Push reviewed controller commits promptly. If publication credentials are unavailable, preserve the
  local commit and report the exact branch, SHA and authentication blocker.
- Never modify `/home/khwang/opd_temporal_multimodal_research_report.*`.

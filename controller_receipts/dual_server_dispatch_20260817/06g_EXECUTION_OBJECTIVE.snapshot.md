# ICLR 2027 06G pristine DiffusionDrive / NAVSIM execution objective

Status: **protocol repair pending; cache, CL, T0, and GPU stopped**

Execution branch: `codex/iclr2027-diffusiondrive-06g`

Pristine upstream: `hustvl/DiffusionDrive@9b52ed0ec06b073d82d6f392ab084c7b301c8681`

Coordination snapshot: `3013da9bd0189b599c558b25fd3f7a804de28f37`
(not an accessible source ref and not an execution unlock)

## 1. Research ownership

06G is the only executor for the paper's pristine official
DiffusionDrive/NAVSIM main-evidence line. It owns official-source verification,
NAVSIM source inventory, session-atomic continual-learning manifests, exact
cache construction, the real official-model gate, T0, all NAVSIM arms, NAVSIM
evaluation, and structured receipts.

07G owns an independent nuScenes method/external-validity line. NAVSIM and
nuScenes must remain separate in code provenance, data selection, checkpoints,
tables, statistical families, and conclusions. A pass, failure, or method
choice on one dataset cannot unlock the other. In particular, nuScenes
outcomes cannot alter the preregistered NAVSIM split or hypothesis family.

The 06G path must never consume RAP-modified, Waymo/rendered-camera, or
nuScenes V1SparseDrive agents or data.

## 2. Existing evidence and present boundary

The following narrow R1 result is retained:

- official navtrain population: 103,288 tokens;
- official train partition: 85,109 tokens;
- official validation partition: 18,179 tokens;
- source receipt:
  `reports/r1_pristine_navtrain_receipt_9b52ed0_20260817/receipt.json`;
- source receipt SHA256:
  `fa2f8c417957c22d3fce83d36544c373353a533302deeb5b9a618d9db4aedb1a`.

This evidence is only a source-population and official train/validation
partition receipt. It is not a three-stage continual-learning manifest and
does not authorize cache construction, T0 training, or GPU use.

The superseded 10,444-token asset remains preserved at
`/home/xlxia/datasets/navsim-exp/training_cache_failure_patch_cl_v1_d2238dbb`
but is prohibited from the pristine main path. Its report-computed post-scan
canonical index digest is
`a1e1c77acd886b23324d8d0f4f0ad6fb9b07ea6388dd9c3feb8f7b182d963b8a`.
That digest is not a file hash, and no `cache_index.json` path may be claimed.

## 3. R0-R4 objective and gate order

1. **R0 — pristine source/environment/data provenance.** Retain the exact
   official commit, independent environment lock, raw-tree hashes, and the
   controlled `env.sh` audit. No RAP or nuScenes source may enter the path.
2. **R1 — source inventory and protocol mapping.** Retain the passed official
   population/train-val receipt. Admit Failure-Patch and Chronological CL only
   through separately frozen, session-atomic manifests derived prospectively
   from that inventory.
3. **R2 — exact cache and real one-batch gate.** Only after an accessible
   reviewed implementation/transfer reference, complete content hashes, and
   controller authorization: construct a manifest-exact cache; replay its
   token/log/session membership; then run the official-model sequential-loss,
   teacher=student, query-budget, joint-backward/AdamW, and optimizer-resume
   checks on a freshly verified idle GPU.
4. **R3 — T0 and Seed-0 funnel.** Train T0 from scratch using only the frozen
   Stage-1 data. Audit numerical health and checkpoint/environment hashes.
   Then execute the preregistered Seed-0 Tier-A funnel with identical data
   order and budget instrumentation. Do not use an official full-data
   checkpoint that has seen later sessions.
5. **R4 — confirmatory main experiment and delivery.** Run the preregistered
   multi-seed NAVSIM experiment only after the Seed-0 funnel decision is
   frozen. Apply the immutable hypothesis family and global multiplicity rule,
   evaluate the final test once under the sealed policy, and return configs,
   SHA chains, JSON/CSV, curves, resource receipts, and structured tables.
   Model weights remain on 06G.

No later stage can retroactively satisfy an earlier gate.

## 4. Frozen method tiers

### Tier A: main NAVSIM evidence

- **T0 reference:** from-scratch Stage-1 teacher/reference, trained without
  exposure to later stages.
- **Sequential:** no-replay continual fine-tuning baseline.
- **Step-matched replay:** replay baseline whose optimizer-update budget is
  matched to the continual method budget.
- **LwF:** teacher-based distillation baseline with explicit teacher-query
  accounting.
- **Drive-OPD:** primary method, with the same optimizer/data-order contract as
  the other trainable arms and explicit student/teacher query counts.
- **Full-exposure replay:** replay baseline that exposes the agreed stored
  population, with exposure and storage cost reported rather than hidden.
- **DER++:** eligible only after its continuous-trajectory buffer item,
  logits/targets, replacement, sampling, and storage semantics are frozen and
  pass a real-adapter receipt.
- **Joint/all-seen oracle:** non-continual upper-bound reference trained on the
  preregistered all-seen data; label it as an oracle, never as a budget-matched
  continual method.

### Tier B: ablation or secondary evidence

- fixed-teacher OPD and EMA (`m=0.99`) OPD are ablations, not replacements for
  the registered primary Drive-OPD arm;
- EWC and A-GEM are secondary baselines only after a real-adapter receipt
  establishes parameter/Fisher or gradient-projection semantics and the same
  budget counters;
- ALER, TALR, O-LoRA, and GoalFlow are downgraded and are outside the current
  main-evidence tier unless a future prospectively frozen controller amendment
  and independent audit explicitly promotes them.

## 5. Fair-compute and storage ledger

Every run manifest and terminal receipt must bind, per stage and in total:

- sorted unique current-data sample/token IDs and their count;
- sorted unique old-data sample/token IDs and their count;
- total current and old presentations, including repeated replay exposure;
- model forward counts, backward counts, and optimizer-update counts;
- student denoiser/query counts and teacher denoiser/query counts;
- wall-clock duration measured over the same declared boundary;
- peak allocated/reserved VRAM under the same measurement convention;
- persistent and transient stored bytes, including replay items, logits,
  checkpoints, and method-specific state.

Trainable arms must share the frozen batch construction, data order, learning
rate schedule, epoch/update rule, and one declared joint backward plus AdamW
update semantics. A runtime OOM may trigger only a prospectively recorded,
uniform correction; it cannot silently change one arm's compute budget.

The ledger reports both nominal algorithm counts and runtime-observed counts.
Any mismatch fails closed before an optimizer update or claim receipt.

## 6. Run, evaluation, and claim identity

Each immutable run identity must bind at least:

- registered `method_id` and its one-to-one `training_arm` mapping;
- seed and deterministic-seed contract;
- immutable config bytes/SHA and protocol bytes/content SHA;
- source commit/tree, environment lock, data/cache receipt, stage, and run ID;
- completion target and observed completed epochs/steps/updates;
- endpoint checkpoint path, content SHA, optimizer state, and parent
  checkpoint identity.

Each evaluator-run receipt must additionally bind the loaded checkpoint, exact
evaluator source/environment, evaluation receipt, registered evaluation
domain/stage/split, CSV/result bytes and SHA, metrics, and completion status.
This receipt is the required semantic edge proving that a checkpoint generated
the cited CSV.

Claim inventory must use one registered root and schema, parse every
claim-eligible object irrespective of filename tricks, and match the immutable
Git family exactly. Deduplicate by semantic coordinate and by content of the
receipt, comparison spec, run/result, checkpoint, and CSV evidence. Distinct
nominal seeds may not reuse one realized run or raw evidence unless that
sharing was explicitly preregistered and excluded from independent-seed
counts.

Evaluation domains come from a frozen domain registry; free-text domains are
invalid. Confirmatory seed count and final-test chronology require prospective
power/precision and access-control receipts rather than declarative labels.

## 7. Immediate next admissible action

Remain stopped. The next admissible action is a read-only fetch and SHA audit
of a newly published, 06G-accessible source/transfer reference only after it
contains:

1. executable negative tests for all blind-review P0/P1 bypasses;
2. an independently reviewed clean source commit and complete transfer bundle;
3. frozen session-atomic NAVSIM CL manifests derived from the official R1
   population, with content hashes and an executable split rule;
4. an explicit controller statement identifying which gate, if any, is open.

Until all four are present, cache construction, official-model preflight, T0,
CL evaluation, and every GPU action remain prohibited.

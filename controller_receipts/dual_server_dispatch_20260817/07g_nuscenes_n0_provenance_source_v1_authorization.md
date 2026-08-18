# 07G nuScenes N0 provenance source-v1 authorization

Date: 2026-08-18

Decision: **FORWARD-ONLY SOURCE AND SYNTHETIC CPU WORK AUTHORIZED; REAL DATA ACCESS STOPPED.**

The controller has already frozen the nuScenes atomic unit as one complete `log_token`, accepted the
exact session-manifest producer-v2 identity at commit
`495bac670073fc3f46f0f773ca51033331078509`, and forbidden scene-level assignment and fallback
splits. The remaining first-order blocker is an independently auditable N0 dataset identity and
closure bundle. This authorization permits only the source needed to produce and validate that
bundle later.

## Exact forward path

- base branch: `codex/iclr2027-drive-opd-07g-nuscenes-manifest-source-v2`
- base commit: `495bac670073fc3f46f0f773ca51033331078509`
- base tree: `a7c86fbafea77ec647197c6d24d65edff83c95a8`
- new branch: `codex/iclr2027-drive-opd-07g-nuscenes-n0-provenance-source-v1`
- suggested worktree: `/home/khwang/domain-selector-nuscenes-n0-provenance-source-v1`
- construction: exactly one non-merge child commit; no rewrite, amend, reset or rebase

Only these four paths may change:

1. `selector_bench/selector_bench/continual/nuscenes_provenance.py`
2. `selector_bench/scripts/60_audit_nuscenes_n0_provenance.py`
3. `selector_bench/tests/test_nuscenes_provenance.py`
4. `paper_iterations/P001_scheduler_session_contract/NUSCENES_N0_PROVENANCE_SOURCE_V1.md`

## Required candidate semantics

The source must accept only explicit, content-addressed inputs and must not discover or infer a
dataset root, release, devkit source, official split, map source or CAN source. It must bind the
v1.0-trainval identity, all thirteen required metadata tables, official split source, devkit
package/source identity and a canonical metadata digest.

The validator must recompute all foreign keys and reciprocal chains, complete-log ownership,
official-partition containment, and every descendant scene, sample and `sample_data` token. One
canonical ledger row must represent each complete `log_token`; multiple scenes under the same log
must never be separated.

The later real bundle format must include content-addressed inventories for all registered six-camera
and `LIDAR_TOP` frame/sweep files plus explicit map and CAN linkage. File processing must be streaming
and must reject missing, duplicate, symlinked, escaping, non-regular or changed-during-read files.
Claims based on caller-asserted counts are forbidden: every count, membership and digest is observed
and recomputed.

Outputs must be canonical JSON/JSONL, staged privately, semantically revalidated, checked again after
validation and published atomically with no replacement. Any failure must leave no accepted output.
Focused synthetic tests must attack token/JSON duplication, broken chains and foreign keys,
cross-partition logs, fallback splits, partial descendants, sensor/map/CAN omission or substitution,
path and symlink attacks, changed-during-read files, non-finite/Boolean values and post-validation
mutation.

## Boundary

This is not authorization to read local nuScenes or NAVSIM metadata, sensors, maps, CAN, info PKLs or
raw data. It does not authorize a real N0 receipt, real manifest, Failure-Patch evidence, materialized
dataset, anchor, cache, model, T0, CUDA/GPU, training, evaluation, final-test access, performance
claim, publication, push or 06G dispatch.

The next gate is a controller audit of the exact clean one-commit source candidate. Only a later,
separate controller authorization may permit a real N0 scan.

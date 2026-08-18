# 07G nuScenes session-manifest source v2 controller audit

Date: 2026-08-18

## Decision

**PASS for a later approved-producer identity freeze only.**

The exact source candidate is `495bac670073fc3f46f0f773ca51033331078509`, sole parent
`37234dc194242a31459b0210bfbbd220c4d1a1ee`, tree
`a7c86fbafea77ec647197c6d24d65edff83c95a8`. It is one non-merge commit with seven changed paths
and a clean worktree. The seven normative controller-contract files match controller commit
`fd169e509af64da942ba9c63fc1e0e8ff5e584e9` byte-for-byte.

This is not producer approval by itself. A separate forward controller commit must bind the exact
commit, tree, producer path/hash and validator path/hash in the nuScenes requirements. That identity
freeze will still not authorize dataset access or manifest generation.

## Independent replay

- exact v1 bypass regression: 1 passed in 0.72 seconds; output SHA256
  `6262dd4ae1e1610248aa7a1d90c86e6b7861110a1a434aeede186690d3652770`
- detached exact-candidate suite: 71 passed in 12.12 seconds; output SHA256
  `9092d793d1fab5629e5e0b4de4ec9a9e0580f8675a9fc8a2f47a1c47e744e957`
- byte-identical controller-validator suite: 19 passed in 0.50 seconds; output SHA256
  `815a1a0dc1c9272190561f6057a08651e12ab643edcc13279562a4a4390bfbc9`
- CUDA hidden, bytecode disabled and pytest cache provider disabled; zero cache/bytecode paths
- `git show --check`, changed-Python AST parsing and changed-JSON parsing passed

## P0 closure on the synthetic source surface

Source v1 allowed a caller to replace arbitrary Failure-Patch statistics and regenerate a matching
`pre_access=true` receipt. Source v2 first binds the raw stream and receipt to controller evidence.
If those two hashes are also substituted, it recomputes the canonical development selection input
and rejects it against a separately frozen hash before `failure_patch_stages` can run.

The evidence surface additionally binds a canonical access ledger, family/access/evaluator commit
identities, evaluator source and metric/domain registries, with strict UTC chronology and canonical,
finite data. Synthetic mutations cover blocked or malformed authority, identity/hash substitution,
chronology rollback, duplicate/non-finite/noncanonical evidence, intermediate output mutation and
failed publication cleanup. Complete `log_token` membership includes every `sample_data` descendant,
including radar records not selected into the model-file inventory.

## Remaining authority and data boundary

The producer consumes controller authority; it does not create it. Before any real Failure-Patch
freeze, the controller must independently authenticate the named Git objects, access ledger,
evaluator and chronology. nuScenes release/devkit/split/CAN provenance, all-table expectations,
real file inventories, accepted population receipt, exact session assignments, DiffusionDrive
materialization lineage and nuScenes evaluator/metric registries are still missing.

Accordingly, authoritative Failure-Patch evidence, N0 metadata access, real manifest generation,
publication/transfer, 06G replay, cache, model, T0, CUDA/GPU, training, evaluation, final-test access
and performance claims all remain stopped.

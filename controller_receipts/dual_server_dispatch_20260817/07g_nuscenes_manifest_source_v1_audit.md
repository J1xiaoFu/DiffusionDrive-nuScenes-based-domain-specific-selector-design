# 07G nuScenes session-manifest source v1 controller audit

Date: 2026-08-18

## Decision

**REJECTED for producer approval: one P0 remains open.**

The exact candidate is `37234dc194242a31459b0210bfbbd220c4d1a1ee`, sole parent
`653e797a2381cdf017466ef95aefcc41961f6715`, tree
`c1d83c19ef2e08167b0014ac48c4828a95bb5651`. It is one non-merge commit with 11 changed paths and
a clean worktree. All seven controller-contract copies match the exact bytes authorized at controller
commit `04b2290979cb711ada763967ad55a62586f7d7be`.

This rejection does not authorize a rewrite of the candidate. Any repair must be a separately
authorized forward-only child after the controller freezes the missing evidence-binding semantics.

## Independent replay

- focused synthetic suite: 44 passed in 7.85 seconds; output SHA256
  `c4c40cd882547a6c315a144b7821915653427abce9f2c5714257af2518c5bfd1`
- adjacent NAVSIM/legacy-nuScenes regression: 10 passed in 0.12 seconds; output SHA256
  `6df23f5e7aa2f42be8c82e63634e296d396d4d919f2ef69394b6bb2129841712`
- CUDA hidden, bytecode disabled and pytest cache provider disabled
- exact candidate receipt SHA256:
  `316b2309c3671ea769d5edc0cb81f5c03a96f6d7fee39d3c61abd0a6c357fe63`

The log-atomic closure, canonical artifacts, source-hash comparison, frozen semantic validator,
post-validation output re-read, no-replace publication and failure cleanup all behaved as intended in
the synthetic surface.

## Open P0: self-attested Failure-Patch evidence

`build_session_atomic_manifest` accepts the Failure-Patch stream and receipt from the same input
spec. `_load_failure_stream` verifies exact development IDs, finite statistic fields, three asserted
Booleans and hashes that bind the receipt to that same stream. It never requires a controller-frozen
Failure-Patch receipt/stream hash, authoritative access ledger, authorization chronology,
evaluator/family/domain identity or source commit.

The controller replaced every Failure-Patch statistic with arbitrary values, handwrote a matching
receipt with `pre_access=true`, and ran the exact candidate. The frozen semantic validator returned
`passed` and the producer published a Failure-Patch assignment. Reproduction output SHA256:
`9a4da0777d7f3063af348d1eee37d814484bd3b50289de9e15494d516f501810`.

This is the same self-attestation class that earlier blind reviews rejected for final-test chronology
and seed-design evidence. An internally consistent receipt is not proof of prior authorization.

Required closure:

1. controller-owned requirements freeze exact Failure-Patch stream and receipt hashes;
2. the receipt binds authoritative pre-access chronology/access-ledger evidence, evaluation family,
   domain/metric registries, evaluator identity and source commit;
3. producer and independent validator reject any substituted stream or receipt even when internally
   consistent; and
4. synthetic coverage includes handwritten receipt, stream substitution, chronology rollback and
   identity mismatch negatives plus an executable controller-frozen positive fixture.

## P1 documentation/provenance findings

The paper report defines the atomic unit using only selected camera/LiDAR sample-data records, while
the implementation and frozen contract include every `sample_data_token` descendant in the ledger.
The report must state the all-sample-data membership rule and separately describe the selected
physical-file inventory.

The official split mapping and CAN-to-log inventory are content-addressed but their semantic
authority still depends on a later controller N0 freeze. That freeze must bind authoritative devkit
split and CAN provenance identities; source self-consistency cannot promote them.

## Boundary

Producer approval, real nuScenes N0 access, manifest generation, publication/transfer, 06G replay,
cache, model, T0, CUDA/GPU, training, evaluation, final-test access and claims all remain stopped.

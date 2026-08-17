# P000 Author Response

## Decision

We accept the `3/10, reject` calibration. The reviewed commit is a research program plus historical
diagnostics, not an ICLR-ready paper. No method-performance claim is promoted from P000.

## P0 responses

1. **Scheduler semantics — accepted.** The current OPD rollout initializes at timestep 8 but queries
   `(10, 0)`, while LwF independently noises GT at those query times. This invalidates the claim that
   support is the only difference. All new distillation training is blocked until P001 derives a
   scheduler-consistent state/time sequence and passes an identical-state loss-and-gradient negative
   control on the official model. Already trained distillation arms remain historical diagnostics.
2. **14,951 versus 103,288 tokens — accepted.** The local cache-derived protocol and the raw official
   NAVSIM inventory have not been reconciled. The 10,444-token portable train manifest must not define
   the pristine official main experiment until P001 explains every inclusion/exclusion at session,
   log and token levels and records the source policy that produced deployment metrics.
3. **Statistical atom — accepted.** The primary paired estimator will cluster by the same
   timestamp×vehicle session used for splitting. Segmented-log and token-weighted estimates become
   sensitivity analyses. Seeds are repeated model fits, not independent scene observations; P001 will
   preregister a seed/session hierarchical summary, invalid-row policy, effect-size gate and
   multiplicity policy.
4. **EWC/A-GEM validity — accepted.** EWC is removed from the main table until a per-example or
   explicitly named batch-gradient curvature estimator is implemented. A-GEM replay forward must not
   update BN state; its buffer behavior and gradient-only intervention require a regression test.
5. **Artifact inspectability — accepted.** Each claim-bearing round will commit portable manifests,
   machine-readable summaries, table sources and plotting code. Checkpoint paths and hashes alone are
   not treated as independently reviewable evidence.
6. **Official evidence — accepted.** The first performance-bearing table must include T0,
   sequential, step-matched replay, full-exposure replay, LwF and scheduler-correct Drive-OPD on old
   and new audit domains. A null-conflict result narrows or redirects the paper; it does not trigger
   an unstructured sweep.

## Clarifications without rebuttal

- The reviewer could not run tests in its read-only base environment. Before review, the four new
  continual-learning test files passed 16/16 in the recorded project environment. This is only an
  author-side receipt; P001 must make the environment and integration test independently reproducible.
- The context-swap procedure will be called an intervention diagnostic until representation alignment,
  off-manifold controls and a defensible identification argument exist.
- NAVSIM collision and compliance components will be called benchmark safety proxies, not real-world
  safety outcomes.

## Action entering P001

P001 is titled **Scheduler-Consistent Support Distillation and Session-Atomic Official Protocol**.
It must produce a mathematical state/time derivation, official-model invariant tests, a complete
token-reconciliation table, a session-level statistical analysis contract, one mechanism figure,
one provenance-linked result table, a frozen candidate commit and a new context-free review.

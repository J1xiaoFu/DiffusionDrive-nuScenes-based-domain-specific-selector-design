# Drive-OPD 07G P0 RC3 forward-only repair authorization

Date: 2026-08-18

Status: **AUTHORIZED FOR SOURCE/CPU REPAIR ONLY**

## Authoritative rejection and immutable base

- authoritative review: task `01a0106d-4a96-72d2-aa90-da0acd5f72c3`, fresh zero-parent
  `gpt-5.6-sol` at `xhigh`
- verdict: **3/10 REJECT / FAIL**, confidence 5/5
- preserved review report SHA256:
  `78664e96f07209691f633330bad51e6e642c6b3c33ec1954fa928ad0087c66bd`
- immutable RC2 branch: `codex/iclr2027-drive-opd-07g-p0-rc2`
- immutable RC2 commit: `d527274d368b6d6def5809b002324b31e88ae1bd`
- immutable RC2 parent: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`
- immutable RC2 tree: `dce36d261b8a0f4bde4519a3a393471c54b05356`
- authorized RC3 branch: `codex/iclr2027-drive-opd-07g-p0-rc3`
- authorized RC3 worktree: `/home/khwang/domain-selector-p0-rc3`
- required history: exactly one new, non-merge commit whose sole parent is exact RC2

At authorization time the RC3 ref equals exact RC2 and its worktree is clean. RC2, RC1, the 07G
research and recovery refs, and the controller/publication refs are immutable under this authorization.
Do not reset, rebase, amend, merge, delete, force-update or otherwise move any existing ref.

## Authorized repairs

### P0-1: monotonic Git history and strict final-test chronology

The chronology validator must derive canonical commit instants from raw Git commit objects, normalize
them to UTC, inspect the complete linear ancestry from family freeze through access, and independently
establish all of the following:

1. Every commit on the family-to-access chain has exactly one parent in the inspected interval and
   strictly increasing committer time. A later intermediate commit may never be followed by an earlier
   access commit.
2. The family-freeze commit contains the declared family specification and empty canonical access
   ledger; every intermediate commit preserves those ledger bytes exactly; the access commit appends
   exactly one canonical bound event.
3. The four instants satisfy the strict relation
   `family_commit_time < authorized_at_utc < access_commit_time < evaluator_started_at_utc`.
   Equality at any boundary is rejected.
4. UTC strings and Git time-zone offsets are parsed without caller-supplied chronology shortcuts;
   malformed, ambiguous or non-finite time representations fail closed.

Add executable negative fixtures for non-monotonic intermediate/access commit times, reversed endpoint
times and equality at each strict boundary. Preserve the merge/fork, non-empty freeze ledger,
add-delete-add, duplicate event and wrong-ledger-path negatives already closed by RC2.

### P0-2: active-family binding of sealed final-test access

The evaluator receipt's sealed-test-access payload must bind the active comparison family, not merely
report a self-selected verified status. Bind and verify at minimum:

- family ID, canonical family-spec repository path and SHA256;
- family-freeze commit;
- canonical ledger path and frozen/access ledger SHA256 values;
- access commit, event ID and event index;
- evaluation cell, protocol, final-test domain and split.

`load_evaluator_run_contract` must receive the expected family identity derived from the current frozen
comparison family and verify every field exactly for test evidence. The comparison/claim path must pass
that controller-derived identity to every evaluator receipt and require all evidence in the family to
share it. A receipt valid for family A must fail when transplanted into a distinct frozen family B.

Add executable negatives for cross-family transplantation and mutation of every bound identity field;
each must exit nonzero before a claim artifact exists.

### P0-3: mandatory multi-candidate common-baseline cells

The family schema must explicitly designate every shared confirmatory cell, its common baseline identity
(method and training arm), and a minimum of at least two distinct candidate methods/arms. For every
designated cell the validator must require:

- at least two distinct declared candidates compared with exactly the same declared baseline;
- unique comparison IDs and evidence identities;
- all seven registered metrics for every candidate comparison;
- exact declared projected coordinates, with no implicit/default cell or baseline substitution.

One comparison, duplicated candidates, different baselines or an undeclared cell cannot satisfy the
family. Update the checked example and positive fixture to contain at least two candidates. Add
executable negatives for all four cases.

### P0-4: finite seed-design inputs, replay and outputs

Introduce one reusable finite-number validation path across the generator and claim validator. Every
numeric value that can affect critical values, power, held-out FWER or confirmatory seed selection must
be finite before random-number generation/statistic computation and again before receipt comparison or
serialization. This includes generator inputs, pilot-matrix cells, normalized invocation values,
simulation intermediates and replay outputs. Use explicit `math.isfinite`/`numpy.isfinite` checks and
reject booleans where an integer or real parameter is required.

Generator, audit and claim JSON serialization must use `allow_nan=False` where applicable. Preserve exact
normalized deterministic replay. Add executable negatives for `NaN`, `+Infinity` and `-Infinity` in the
minimum effect, target power, family alpha, pilot matrix, critical values, per-metric power and held-out
FWER. Every mutation must exit nonzero before claim output.

## Closed invariants that may not regress

RC3 must retain and regression-test all source/CPU closures already present in RC2, including measured
main/A-GEM presentation and unique-identity budgets, deterministic replay targets, ALER query capture,
measured transient storage, ledger append-only history, executable seed replay, metric/method registry
binding, filename/order invariance and all earlier P0 attack families. Passing only the four new
reproductions is insufficient if any closed invariant is weakened.

## Frozen attack and CPU acceptance

Promote the four reviewer reproductions into four named harness families after the existing 15. The RC3
harness must therefore report at least 19 named attack families and at least 36 individual mutation
receipts; every listed mutation must have a nonzero CLI exit and no claim output before assertion. The
positive fixture must simultaneously exercise strict chronology, current-family-bound evaluator
receipts, two or more candidates against one common baseline, finite seed design, and filename/
hypothesis-order invariance.

Run the existing six source/CPU suites plus focused new tests with `CUDA_VISIBLE_DEVICES=''`, bytecode
disabled and the pytest cache provider disabled. Also run in-memory compilation, JSON parsing and
`git show --check`. Scratch outputs remain outside the repository and no cache, bytecode, model or
checkpoint artifact may be created.

## Authorized actions

- edit source, configuration, tests and paper/audit response files needed only for the four repairs and
  non-regression evidence above;
- run static checks, synthetic temporary-Git fixtures and source-only CPU unit/integration tests;
- expand the named P0 harness and generate temporary CPU receipts;
- create exactly one auditable RC3 commit whose parent is exact RC2;
- report commit/parent/tree, clean status, exact path and SHA256 inventories, commands, outputs, harness
  receipts and artifact-boundary attestation to the controller.

## Not authorized and subsequent gates

This authorization does not permit publication or push, review-task creation, controller promotion,
06G dispatch, NAVSIM/raw-data/cache reads or construction, dataset-adapter execution, real-model import
or preflight, T0, CUDA/GPU activity, training, evaluation, final-test unsealing or performance claims.

RC3 cannot self-promote. After exact controller intake, the controller may separately freeze and publish
the source bundle and request a new context-free `gpt-5.6-sol` `xhigh` exact-RC3 review. No isolated 06G
replay is dispatched unless that review closes every P0. Even a review pass does not unlock data,
session-atomic CL manifests, cache, model, T0, GPU, training, evaluation or claim gates without later
explicit controller authorization.

# Drive-OPD 07G P0 RC2 forward-only repair authorization

Date: 2026-08-17

Status: **AUTHORIZED FOR SOURCE/CPU REPAIR ONLY**

## Immutable base and destination

- immutable RC1 branch: `codex/iclr2027-drive-opd-07g-p0-rc`
- immutable RC1 commit: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`
- immutable RC1 parent: `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`
- immutable RC1 tree: `0e6e9a0adda280765ce41ab377de9bbbaea0cff1`
- authorized RC2 branch: `codex/iclr2027-drive-opd-07g-p0-rc2`
- authorized RC2 worktree: `/home/khwang/domain-selector-p0-rc2`
- required history: exactly one new, non-merge repair commit whose sole parent is exact RC1

RC1, `codex/iclr2027-drive-opd-07g-research`,
`codex/iclr2027-drive-opd-07g-p0-pre-squash`, and every recovery ref are immutable under this
authorization. Create the RC2 branch/worktree without checking out, resetting, rebasing, amending,
deleting, force-updating, or otherwise moving any existing branch or ref.

## Authorized repairs

### P0-A: authoritative final-test chronology

Replace scalar or self-asserted prior-access state with a canonical access ledger declared by the
frozen family. The validator must independently establish all of the following:

1. The family commit contains the declared canonical family specification and an empty canonical
   access ledger, each bound by repository path and SHA256.
2. The access commit descends from the family commit and contains exactly one unique ledger event for
   the family, evaluation cell, protocol and final-test domain being evaluated. Event sequence is
   derived from canonical ledger position, never from a caller-supplied count.
3. Strict UTC parsing and Git-object checks establish
   `family_commit_time <= authorized_at <= access_commit_time <= evaluator_start_time`.
4. The receipt binds family ID, family-spec path/hash, ledger path/hash, event ID, evaluation cell,
   protocol and domain. No asserted-count substitution, alternate path, duplicate event or
   non-canonical ledger is accepted.

Add executable negative fixtures for a 1900 authorization, future authorization, non-empty freeze
ledger, duplicate event, wrong canonical ledger/path/family and asserted-count substitution. Preserve
a positive fixture that constructs the valid history from real temporary Git commits.

### P0-B: measured production budgets

Production observed budgets must be derived only from actual batches consumed by the runner:

- return stable cache-token identity with every training sample;
- count actual current and old presentations and unique identities from every main batch;
- count A-GEM replay presentations and identities from every replay batch actually consumed;
- never initialize or complete observed fields by spreading or copying target-budget values;
- require promotion-time equality between measured observations and the applicable frozen target,
  while preserving explicit incomplete status for partial or `max_steps` runs.

Add runner-level CPU fixtures for complete and partial runs, step-matched and full replay, A-GEM and a
loader-delivery mismatch that must fail rather than certify completion.

### P0-C: executable seed-design replay

Factor the seed power/calibration computation into one deterministic importable pure module. The
generator and validator must share that implementation. Bind and replay:

- canonical pilot-matrix content and hash;
- normalized generator inputs, deterministic seed and repetitions;
- effect-size assumptions and held-out calibration design;
- generator/module source path and computed source SHA256;
- normalized invocation and exact normalized output, including critical values, per-metric power and
  held-out family-wise error rate.

The positive family fixture must invoke the executable generator. Add negative fixtures for
handwritten outputs and mutations to seed, repetitions, effect assumptions, pilot content/hash,
program source/hash, invocation, critical values, power and held-out FWER. Deterministic vectorization
or memoization is allowed only when the replay result remains exact and the cache key binds every
semantic input and program identity.

### P1 hardening included in RC2

- Put the complete ALER search and repair path inside `capture_query_audit`; promotion uses measured
  query counts before `optimizer.step`, never hardcoded counts.
- Permit multiple registered method-versus-common-baseline comparisons over one evaluation cell while
  retaining unique comparison IDs and evidence identities. Required crossed cells operate on the
  declared projected coordinates and may not collapse distinct comparisons.
- Define and measure transient storage at runtime. A literal declared zero is not evidence.
- Promote the new chronology and seed-design bypasses into the named P0 attack harness.

## Authorized actions

- create the new RC2 worktree/branch from exact RC1;
- edit source, configuration, tests and paper/audit response files needed for the repairs above;
- run source-only static checks, synthetic temporary-Git fixtures and CPU unit/integration tests with
  `CUDA_VISIBLE_DEVICES=''`, bytecode disabled and the pytest cache provider disabled;
- expand the P0 harness and generate final CPU audit receipts in a temporary directory;
- create exactly one auditable RC2 commit and report its commit, parent, tree, changed paths, path
  SHA256 inventory, commands, outputs and receipt hashes to the controller.

The RC2 commit must be cleanly reproducible from RC1, contain no cache/checkpoint/model/data payload,
and leave its worktree clean. Generated scratch outputs stay outside the repository. Only concise,
final source/CPU audit receipts explicitly referenced by the paper response may be tracked.

## Not authorized

This authorization does **not** permit:

- modifying or moving RC1, the current research branch, the recovery branch or controller branches;
- merge, rebase, amend, reset, force-push, deletion or history rewrite;
- publication, push, review-task creation, controller promotion or 06G dispatch;
- NAVSIM/raw-data/cache reads or construction, dataset adapter execution, real-model import or
  preflight, T0, CUDA/GPU activity, training, evaluation or performance claims;
- changing official NAVSIM population/partition or unsealing final-test content.

## RC2 acceptance and subsequent gates

The implementation report must show exactly one commit from RC1, a clean worktree, exact path/blob and
SHA256 inventories, static checks, the frozen five-suite replay, expanded P0 harness results and
individual fail-closed mutation receipts. Passing those checks does not self-promote RC2.

After controller intake, the controller may separately freeze and publish the exact RC2 source,
create a new context-free `gpt-5.6-sol` `xhigh` review task and request an isolated 06G source/CPU
replay. RC2 is rejected unless that exact-RC2 review has no open P0 and independently closes every
source/CPU-testable historical issue. Even an RC2 review PASS cannot unlock dataset, manifest, cache,
model, T0, GPU, training, evaluation or claim gates without later explicit controller receipts.

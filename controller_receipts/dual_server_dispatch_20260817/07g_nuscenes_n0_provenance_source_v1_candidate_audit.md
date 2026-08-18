# 07G nuScenes N0 provenance source-v1 candidate audit

Date: 2026-08-18

## Decision

**PASS for source-identity freeze only.**

The exact candidate is `ac38c9ffca9d4a428cee97606a92f93fb5ff8d82`, sole parent
`495bac670073fc3f46f0f773ca51033331078509`, tree
`07900590456acb6e523f7d815f969dccc11ae226`. It is one non-merge commit containing only the four
authorized paper, CLI, library and test paths, and its worktree is clean. The controller freezes that
exact commit/tree and the CLI/library content hashes as the accepted synthetic/source implementation.

This does **not** authorize a real nuScenes scan. A later, separate controller decision must first
freeze a content-addressed real input specification. Real N0 evidence, real session membership,
publication, transfer and execution all remain stopped.

## Verification

- source executor: 54/54 focused provenance attacks, 71/71 adjacent nuScenes manifest tests and
  19/19 controller-contract tests passed
- source executor full repository: 227 tests plus 132 subtests passed in 388.90 seconds; output
  SHA256 `1f0c18a70b21c23ef8e6fb4d8a8ff4c447616644c6793d714f9b4cf0f0272993`
- independent controller detached-Git replay: 132/132 focused and adjacent tests passed in 71.65
  seconds; output SHA256 `559b07f6481d9b859cbbb1fd329f62d7608809dad2947977fd586056526588a5`
- CUDA was hidden, Python bytecode was disabled and the pytest cache provider was disabled; no
  bytecode/cache residue or worktree status drift was observed
- `git show --check`, exact changed-path inventory and per-path SHA256 inventory passed

An earlier archive-only controller diagnostic is recorded transparently but is not a candidate
failure: 79 adjacent tests passed, while 53 provenance fixtures could not fetch the frozen producer
commit because a Git archive has no object database. Repeating the same scope in a detached Git
worktree at the exact SHA passed 132/132.

## Closed synthetic/source attacks

The source binds one unique local origin, an isolated Git environment, exact ordinary tracked HEAD
blobs for the library and CLI, and the frozen manifest-producer commit/ancestry/blob. It rejects
untracked, assume-unchanged, gitlink, ambient-Git and credential-bearing repository substitutions.
The producer command must begin with the exact interpreter and tracked CLI and bind every explicit
root, input-spec hash and output path exactly once.

All metadata, split, sensor, map and CAN inputs are content-addressed. Descriptor-safe component
walks reject symlinks, path escapes, file/root/parent replacement and changed-during-read objects.
The validator recomputes all-table closure, complete-log atomicity and registered-file coverage. The
output must be isolated from every read-only root, survive a second private-staging semantic replay,
and publish with atomic no-replace semantics; the attack fixtures cover partial, post-validation and
pre-publication mutations.

## Remaining gate

No raw nuScenes/NAVSIM data was read, no real input specification or N0 receipt was generated, and
no manifest, cache, model, checkpoint, T0, CUDA/GPU, training, evaluation or final-test operation was
performed. The exact-RC4 context-free `gpt-5.6-sol`/xhigh review still has no valid zero-P0 verdict.
Both real session-atomic manifests and a later explicit controller gate remain prerequisites for any
06G source replay.

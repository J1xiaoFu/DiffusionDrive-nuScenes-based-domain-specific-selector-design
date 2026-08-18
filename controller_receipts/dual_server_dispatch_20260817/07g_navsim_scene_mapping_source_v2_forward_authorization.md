# 07G NAVSIM token-level scene-mapping source-v2 forward authorization

Date: 2026-08-18

## Decision

**AUTHORIZED only for one forward-only source/CPU candidate that replaces the unsupported v1
log-to-scene bijection with exact token-level evidence.**

This authorization does not read NAVSIM or nuScenes, generate a real mapping or manifest, alter
official counts, approve Failure-Patch evidence, publish, dispatch 06G, use a cache/model/CUDA/GPU,
train, evaluate, access final test or support a performance claim.

## Immutable base

- base branch: `codex/iclr2027-drive-opd-07g-navsim-manifest-source-v1`
- base commit: `64447d3a9cc60f61b8596aed41086b5b4311c889`
- base parent: `653e797a2381cdf017466ef95aefcc41961f6715`
- base tree: `5ce81ef06d71189d1ff46676ecc1b4d01e415728`
- authorized branch: `codex/iclr2027-drive-opd-07g-navsim-scene-mapping-source-v2`
- suggested worktree: `/home/khwang/domain-selector-navsim-scene-mapping-source-v2`
- required history: exactly one non-merge commit whose sole parent is the exact v1 candidate

Every existing RC, research, recovery, producer, controller, publication and 06G ref is immutable.
Reset, rebase, amend, merge, force update, history rewrite and deletion are forbidden.

## Why v1 must change

Official NAVSIM v1.1 code stores `log_name`, `scene_token` and `initial_token` as separate source
fields and keys filtered scene windows by a frame token. The approved v1 producer accepts one
`scene_token` for each segmented log and rejects any repeated scene token. That silently imposes a
bijection not established by the source. Equal log/scene counts in the frozen requirements are
unproven because the accepted R1 ledger contains no `scene_token`.

## Canonical evidence row

The v2 source must consume canonical JSONL sorted by unique `token`. Every row has exactly:

- `schema = drive_opd.navsim_scene_mapping_row.v2`;
- `dataset_id = navsim_official_navtrain`;
- `token`, `segmented_log_name`, `scene_token`, `initial_token`;
- `official_partition` and complete-session `atomic_unit_id`;
- `source_identity_sha256`, recomputed from the canonical four-field source identity
  `{initial_token,log_name,scene_token,token}` plus a trailing LF.

There must be exactly one row for every token in the accepted R1 ledger and no extra token. The
segmented log, partition and complete session must match the ledger row containing that token. The
producer derives each session's scene descendants from the sorted unique evidenced scene tokens;
it never copies logs into scenes or asserts that their cardinalities are equal.

Multiple tokens per scene, multiple scenes per segmented log and one scene spanning multiple
segmented logs are valid only when the exact rows keep every descendant inside one complete
session. A scene token crossing sessions or official partitions is always rejected.

## Authority and proof boundary

The mapping proof must bind the exact R1 ledger SHA256
`cf12df3e215874362c67a6ce404a8bb8ae054fdc5c393f7a58385abd2453fa08`, mapping bytes, row and
unique-set counts, per-partition hashes, official semantic-source blobs, builder source identity and
complete invocation. Synthetic tests may build temporary proofs; the committed production
requirements must remain blocked and cannot self-approve the candidate.

Real metadata access is not part of this authorization. After candidate audit and producer-identity
freeze, the controller must separately authorize the exact source metadata surface and invocation,
then independently replay the resulting mapping before updating any frozen scene count or allowing
a real NAVSIM manifest.

## Required tests

Negative coverage includes legacy v1 mappings; missing, duplicate, unknown or extra tokens; wrong
log/session/partition; altered scene or initial token; wrong source identity; a scene crossing
sessions; asserted count/hash substitution; proof/source/program/invocation mutation; fail-open JSON
types; and all staged-write/post-validation/pre-publication output attacks.

Positive coverage must prove that the implementation supports multiple tokens per scene, multiple
scenes per log and a scene spanning logs inside one complete session while deriving exact scene
counts and hashes from the rows.

## Gate boundary

The candidate may run only static and synthetic CUDA-hidden CPU checks and create one auditable
commit. Controller audit/freeze, separate real-metadata authorization, exact mapping generation and
replay, Failure-Patch authority, a complete NAVSIM manifest, a complete nuScenes manifest and the
valid exact-RC4 xhigh zero-P0 review all remain required. 06G source replay and every
cache/model/T0/CUDA/GPU/training/evaluation/final-test/claim gate remain stopped.

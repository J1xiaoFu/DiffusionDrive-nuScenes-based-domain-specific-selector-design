# 07G nuScenes manifest producer v2 forward-only repair authorization

Date: 2026-08-18

## Decision

**AUTHORIZED only for one source/CPU, synthetic-only, forward repair of rejected source v1.**

This authorization is not producer acceptance and does not unlock nuScenes/NAVSIM access, real
membership generation, publication, 06G replay, cache, model, T0, CUDA/GPU, training, evaluation,
final-test access, or a performance claim.

## Immutable rejected base

- base branch: `codex/iclr2027-drive-opd-07g-nuscenes-manifest-source-v1`
- exact rejected commit: `37234dc194242a31459b0210bfbbd220c4d1a1ee`
- sole parent: `653e797a2381cdf017466ef95aefcc41961f6715`
- tree: `c1d83c19ef2e08167b0014ac48c4828a95bb5651`
- authorized branch: `codex/iclr2027-drive-opd-07g-nuscenes-manifest-source-v2`
- required history: exactly one new non-merge commit whose sole parent is exact source v1

All existing controller, publication, RC, research, recovery and 06G refs remain immutable. Reset,
rebase, amend, merge, force update, history rewrite and deletion are forbidden.

## Normative contract

Replace the seven v1 contract copies with byte-identical files from controller commit
`fd169e509af64da942ba9c63fc1e0e8ff5e584e9`, tree
`3ebe4edbc1072e19c1d666cadd89c426357f343e`:

- `MANIFEST_CONTRACT.md` — `04d3109b3f7f33290d563699f0980550ddc1a35e1479ab1049b9d70234b9e0e0`
- `session_atomic_manifest.schema.json` — `6e43ae2af5a981368c9e4bd764691d3d9d8849031e4a02910c5db982ab04e791`
- `population_ledger_row.schema.json` — `e5ac1719c08b37351fb42576cabb63fcf261b97112185e57e2ed3f78df2b7752`
- `protocol_assignment_row.schema.json` — `fa8d3f93a57bc33dcb3655a7c8a782e9c727b18b651a608a9252463938986f96`
- `selection_input_row.schema.json` — `0b05cd114277e98e267953a0d94f3514715e1018ff7767c5222e2537575afc28`
- `nuscenes_manifest_requirements.v1.json` — `fbbf99edce7362fd0537ca694f573e0ed2372927f992da0ad29ac1df11100440`
- `validate_session_atomic_manifest.py` — `70b53893c3bc0704fbf09945a677412effdf4a2b0258ac56c2f8084798805ba3`

The committed production requirements must retain both producer and Failure-Patch evidence blockers.
Only temporary synthetic tests may construct `CONTROLLER_AUTHORIZED` evidence.

## Required repair

Source v1 accepted arbitrary failure statistics plus an internally matching `pre_access=true`
receipt. The repair must treat the controller requirements—not input-spec booleans or an internally
consistent stream/receipt pair—as the authority. For an executable temporary fixture it must:

1. require the exact `failure_patch_evidence` key set and status `CONTROLLER_AUTHORIZED`;
2. hash the canonical Failure-Patch selection JSONL and require equality with the controller-frozen
   `selection_input_sha256` before stage calculation or output publication;
3. bind raw stream, evidence receipt, family freeze/access commits and access ledger, evaluator
   commit/source, and metric/domain registry hashes;
4. enforce strict UTC chronology `family freeze <= authorization <= access commit <= observation
   start <= observation finish <= receipt creation`;
5. include the full evidence object in `selection_parameters_sha256` and surface its canonical hash
   in the semantic validation receipt;
6. reject a changed stream even when the input spec supplies a newly matching receipt and every
   self-asserted boolean is true; and
7. correct the report formula so descendant closure describes all `sample_data` records included by
   the frozen ledger contract, rather than only selected camera/LIDAR files.

The controller will later freeze real evidence only after independently verifying commit ancestry,
access-ledger append history, evaluator/registry identities and exact bytes. Source v2 cannot create
or approve that authority itself.

## Required synthetic attacks

Add fail-before-publish tests for arbitrary statistics plus matching receipts; stream/canonical-input
substitution; blocked/missing/unknown/Boolean status; wrong dataset/protocol/family/registry;
changed stream/receipt/ledger/evaluator hashes; every chronology rollback; duplicate-key,
non-finite and noncanonical evidence; and staging failure. Every rejection must leave no accepted
manifest, assignment, receipt or published directory.

Tests must remain synthetic temporary-directory CPU tests with bytecode/test caches disabled. They
must not inspect conventional dataset roots, environment dataset paths, DiffusionDrive models or
CUDA.

## Later gates

The candidate must report exact branch/commit/parent/tree, changed paths and hashes, byte-identical
contract replay, clean status, test receipts and boundary attestation. It then requires a new
controller source audit. Producer approval, authoritative Failure-Patch evidence, N0 metadata-only
access, official population freeze, real manifest generation, publication and every execution gate
remain separately stopped.

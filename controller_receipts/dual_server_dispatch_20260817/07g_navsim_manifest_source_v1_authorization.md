# 07G NAVSIM session-manifest producer v1 source-only authorization

Date: 2026-08-18

## Decision

**AUTHORIZED only for one forward-only source/CPU candidate that implements a dataset-specific
NAVSIM session-manifest producer.**

This is not approval of a producer or a dataset manifest. It does not authorize reading NAVSIM,
nuScenes or the real R1 evidence files; generating membership; using a cache index or failure
metrics; importing a model; using final-test data; creating a cache; using CUDA/GPU; training;
evaluation; publication; transfer; 06G dispatch; or a performance claim.

## Immutable base and destination

- exact base: `codex/iclr2027-drive-opd-07g-p0-rc4@653e797a2381cdf017466ef95aefcc41961f6715`
- base parent: `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`
- base tree: `53943615137849d3338cc037914b302d6e3db363`
- authorized branch: `codex/iclr2027-drive-opd-07g-navsim-manifest-source-v1`
- suggested worktree: `/home/khwang/domain-selector-navsim-manifest-source-v1`
- required history: exactly one new non-merge commit whose sole parent is exact RC4

All existing RC, research, recovery, controller, publication and 06G refs are immutable. Reset,
rebase, amend, merge, force update, history rewrite and deletion are forbidden.

## Normative contract

Copy byte-identically under `controller_protocols/drive_opd_session_atomic_v1/` from controller
commit `2c0694213b1db0289dd3fec9094d3f1531ee9a23`, tree
`f9a929e5530b342dd3779ac1bea63b8c0f4e39b2`:

- `MANIFEST_CONTRACT.md` — `d9034a7d8829fcc7cfc98b409ca41ba08d482bb518f2e5c2fafdc2b194e7ac87`
- `session_atomic_manifest.schema.json` — `6e43ae2af5a981368c9e4bd764691d3d9d8849031e4a02910c5db982ab04e791`
- `population_ledger_row.schema.json` — `e5ac1719c08b37351fb42576cabb63fcf261b97112185e57e2ed3f78df2b7752`
- `protocol_assignment_row.schema.json` — `fa8d3f93a57bc33dcb3655a7c8a782e9c727b18b651a608a9252463938986f96`
- `selection_input_row.schema.json` — `0b05cd114277e98e267953a0d94f3514715e1018ff7767c5222e2537575afc28`
- `navsim_manifest_requirements.v1.json` — `bf2e6cbf385892bf1aa8316d0ebd12957c3bf5367948899de6d51acfc42f5cbb`
- `validate_session_atomic_manifest.py` — `70b53893c3bc0704fbf09945a677412effdf4a2b0258ac56c2f8084798805ba3`

The committed production requirements deliberately retain blocked producer and Failure-Patch
evidence states. Synthetic tests may create temporary requirements, but the production file must
not be edited to self-approve the candidate.

## Required source surface and behavior

Implement the producer in
`selector_bench/selector_bench/continual/navsim_protocol.py`, a thin CLI at
`selector_bench/scripts/59_build_navsim_session_atomic_manifest.py`, and focused tests at
`selector_bench/tests/test_navsim_continual_protocol.py`.

The producer must:

1. accept only the exact R1 session-ledger and population/session receipts as its eventual real
   evidence surface. It must not load a pickle cache index, infer membership from filenames or use
   the historical 10,444/14,951-token caches;
2. validate every canonical R1 row and recompute row counts, list lengths, uniqueness, splits,
   chronology status and official full/development/sealed expectations;
3. use the complete `session_key` as the atomic unit. All segmented logs and tokens inherit one
   official partition, stage and cell;
4. handle `scene_token` fail-closed. The R1 row has no independent scene-token field, so the producer
   must either bind an accepted dataset-specific proof that scene identity is exactly segmented-log
   identity or consume an independent exact mapping. Silent copying is forbidden;
5. derive strict UTC order keys deterministically from validated chronology keys and reproduce the
   exact frozen three-stage chronological and deterministic cell rules;
6. leave Failure-Patch generation blocked unless the requirements contain the exact separately
   controller-authorized development-only pre-access evidence object and canonical selection-input
   SHA256. A generic metrics CSV or internally matching receipt is not authority;
7. emit the common canonical population, selection-input, assignment and manifest artifacts, bind
   source/invocation/input/output identities, and pass the exact normative validator;
8. validate all output before atomically publishing a previously nonexistent output directory; any
   failure leaves no receipt or partial output; and
9. expose pure standard-library-first replay functions without importing NAVSIM, DiffusionDrive,
   caches, models, checkpoints, CUDA or GPUs.

## Required synthetic adversarial coverage

Use temporary synthetic fixtures only. Cover at least:

- cache-index/pickle, historical-default and filename-derived membership rejection;
- official train/validation crossover and any sealed identifier in selection/assignment;
- unknown, missing, duplicate or reused session/log/scene/token IDs;
- count, list, split, row-hash, receipt-hash and official-expectation mutation;
- missing, duplicate, invalid or rollback chronology;
- silent log-to-scene alias, wrong alias proof and wrong independent scene mapping;
- selection, stage, cell or descendant substitution;
- blocked/self-declared/wrong-family/wrong-access Failure-Patch evidence;
- source, producer or validator hash substitution;
- Boolean counts, NaN/Infinity, duplicate JSON keys and noncanonical JSONL; and
- pre-existing output, staged-write failure and post-write mutation with no accepted partial result.

Tests must hide CUDA, disable bytecode and pytest caches, and must not probe real dataset paths or
environment-derived data roots.

## Candidate receipt and later gates

Report exact branch/commit/parent/tree, clean status, changed paths and per-path SHA256, exact
contract-copy replay, test commands/counts/exits/durations/output hashes, and a no-data/no-cache/
no-model/no-CUDA attestation.

The one candidate commit cannot self-approve. Controller source audit, approved-producer freeze,
authoritative Failure-Patch evidence, real NAVSIM manifest generation, independent replay, the
separate complete nuScenes manifest and a valid exact-RC4 gpt-5.6-sol/xhigh zero-P0 review all
remain required before any distinct 06G source-replay authorization. Every execution gate remains
stopped.

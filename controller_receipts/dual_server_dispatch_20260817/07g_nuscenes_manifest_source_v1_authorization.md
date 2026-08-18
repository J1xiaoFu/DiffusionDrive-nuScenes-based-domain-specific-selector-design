# 07G nuScenes session-manifest producer v1 source-only authorization

Date: 2026-08-18

## Decision

**AUTHORIZED only for one forward-only source/CPU candidate that implements the prospective
nuScenes session-manifest producer.**

This authorization is not an acceptance of a producer or a dataset manifest. It does not authorize
reading local nuScenes or NAVSIM metadata, enumerating sensor files, generating real membership,
materializing info PKLs, importing a real model, using final-test data, creating a cache, using
CUDA/GPU, training, evaluation, publication, transfer, or a performance claim.

## Immutable source base and destination

- source lane: `codex/iclr2027-drive-opd-07g-p0-rc4`
- exact base commit: `653e797a2381cdf017466ef95aefcc41961f6715`
- sole parent: `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`
- base tree: `53943615137849d3338cc037914b302d6e3db363`
- authorized branch: `codex/iclr2027-drive-opd-07g-nuscenes-manifest-source-v1`
- required history: exactly one new non-merge commit whose sole parent is the exact RC4 commit

RC4, RC3, RC2, RC1, research, recovery, controller, publication and 06G refs are immutable under
this authorization. Reset, rebase, amend, merge, force update, history rewrite and deletion are
forbidden.

## Normative controller contract

The candidate must carry an exact, byte-identical copy of the following controller-frozen contract
surface from controller commit `04b2290979cb711ada763967ad55a62586f7d7be`, tree
`131f601bccf9c7846346c341323a471d3189a624`:

- `MANIFEST_CONTRACT.md` — `826da2e22066bcebf712e64e7dc2c583983252b948441a60eca629bfd59c1a67`
- `session_atomic_manifest.schema.json` — `6e43ae2af5a981368c9e4bd764691d3d9d8849031e4a02910c5db982ab04e791`
- `population_ledger_row.schema.json` — `e5ac1719c08b37351fb42576cabb63fcf261b97112185e57e2ed3f78df2b7752`
- `protocol_assignment_row.schema.json` — `fa8d3f93a57bc33dcb3655a7c8a782e9c727b18b651a608a9252463938986f96`
- `selection_input_row.schema.json` — `0b05cd114277e98e267953a0d94f3514715e1018ff7767c5222e2537575afc28`
- `nuscenes_manifest_requirements.v1.json` — `72ec753d36bb5383a4eaee3854cfa991a56406d66920e0f4920996f768665162`
- `validate_session_atomic_manifest.py` — `6799593056840ae60a1b900079f35ea043db9612d5531a4347f9ad88cb30d029`

The copy must remain under `controller_protocols/drive_opd_session_atomic_v1/`. The source candidate
must not edit these bytes. The requirements deliberately retain a blocked producer, missing
authoritative population expectations and blocked Failure-Patch parameters; source tests may use
temporary synthetic requirements, but the committed production requirements must remain blocked.

## Required producer behavior

Implement a standard-library-first, deterministic producer in
`selector_bench/selector_bench/continual/nuscenes_protocol.py`, a thin CLI in
`selector_bench/scripts/59_build_nuscenes_session_atomic_manifest.py`, and focused synthetic tests
in `selector_bench/tests/test_nuscenes_continual_protocol.py`.

The producer must:

1. require explicit `v1.0-trainval` metadata, official split mapping, devkit version/source identity,
   input paths and SHA256 values; disable every fallback split and reject duplicate-key/non-finite
   JSON;
2. group the full release by complete `log_token`; assign every child `scene_token`, `sample_token`
   and `sample_data_token` to its owning log; fail if a scene is unknown, duplicated, omitted, has an
   invalid sample chain, or if one log spans official train and validation partitions;
3. verify exact foreign-key and chain closure across log, scene, sample, sample_data,
   calibrated_sensor, ego_pose, sensor, sample_annotation, instance, category, attribute,
   visibility and map records, with explicit CAN linkage evidence;
4. create a canonical provenance inventory that binds every metadata table's byte hash and row
   count, one overall canonical metadata digest, official split source, selected six-camera and
   `LIDAR_TOP` keyframe/sweep file inventories, sizes and hashes, and missing/duplicate/path-escape
   counts;
5. emit canonical full/development/sealed population-ledger rows whose descendant sets and hashes
   satisfy the frozen row schema, with strict UTC first/last timestamps, location, ordered samples
   and source-row hashes;
6. reproduce the frozen chronological three-stage and deterministic train/audit/test cell rules
   exactly. Failure-Patch generation must fail while its controller requirement is blocked, and later
   must accept only a separately hashed development-only pre-access stream;
7. bind the producer and validator source identities, invocation and every input/output hash in the
   manifest without treating self-declared hashes as controller approval;
8. validate every output before an atomic publication of a previously nonexistent output directory;
   on any failure, leave no manifest, ledger, assignment, receipt or partial output;
9. expose pure functions so a later independent destination can replay population closure, official
   ID digests, stage/cell assignments and content hashes without importing DiffusionDrive or a
   model; and
10. produce no claim, cache, checkpoint, info PKL, anchor, evaluation cell or model artifact.

## Required adversarial synthetic coverage

The committed tests must include a positive multi-log fixture and fail-closed cases for:

- two scenes from one log assigned to different official partitions, stages or cells;
- unknown, missing, duplicate and multiply assigned scenes/samples/sample-data;
- broken sample `prev`/`next` chains and inconsistent first/last sample links;
- missing or foreign calibrated-sensor, ego-pose, sensor, annotation, instance, category,
  attribute, visibility, map or CAN references;
- missing sensor/sweep file, duplicate physical path, root escape, changed file size/hash;
- fallback split use, changed official split hash, official-population subset and sealed identifier
  in selection or assignment;
- chronology rollback/tie substitution, selection-payload mutation, cell substitution and
  cross-log descendant reuse;
- blocked Failure-Patch parameters, self-declared population receipt, unapproved producer,
  source-hash mismatch, Boolean counts, NaN/Infinity, duplicate JSON keys and noncanonical JSONL;
- pre-existing output, mid-write failure and post-write byte mutation, each leaving no accepted
  receipt or partial published directory.

Tests must use synthetic temporary directories only, with bytecode and test caches disabled. No
test may probe conventional nuScenes roots, environment-derived dataset paths, model packages,
CUDA or GPUs.

## Candidate receipt and later gates

The candidate receipt must report the exact branch/commit/parent/tree, clean status, exact changed
path set and per-path SHA256 inventory, contract-copy hash replay, test names/counts/exits/durations
and output hashes, and a no-data/no-cache/no-model/no-CUDA boundary attestation.

The resulting commit is only an intake candidate. A separate controller audit must review the code
and synthetic coverage before replacing `approved_producer`. A later, separately authorized N0
metadata-only run must establish authoritative nuScenes population/provenance evidence before the
controller can fill official counts/digests or permit real manifest generation. Materialization,
info-PKL lineage, train-only anchors, evaluator/metric/family freezes and all execution gates remain
separate and stopped.

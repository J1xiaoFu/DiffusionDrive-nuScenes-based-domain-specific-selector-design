# 06G DiffusionDrive / NAVSIM execution rules

These instructions apply to this repository and every path below it.

## Identity and scope

- The execution branch is `codex/iclr2027-diffusiondrive-06g`.
- The pristine upstream baseline is `hustvl/DiffusionDrive` at commit
  `9b52ed0ec06b073d82d6f392ab084c7b301c8681`.
- 06G is the sole main-evidence executor for pristine official
  DiffusionDrive on NAVSIM. Keep official NAVSIM code, data, caches,
  checkpoints, metrics, and receipts on this server.
- 07G independently owns the nuScenes method and external-validity line.
  The two datasets must have separate tables, statistics, manifests, caches,
  checkpoints, and claims. Neither line unlocks the other, and nuScenes
  results must never influence NAVSIM split construction or model selection.
- Never mix this path with RAP/Waymo/rendered-camera modifications, the
  migrated nuScenes V1SparseDrive code, or the old nuScenes adapter.

The durable 06G objective and gate order are defined in
`ICLR2027_06G_EXECUTION_OBJECTIVE.md`. When prose conflicts, the stricter
stop condition wins.

## Current hard stop

- R1 proves only the official NAVSIM source population and official
  train/validation partition: 103,288 navtrain tokens, split into 85,109
  train and 18,179 validation tokens.
- The failed claim-protocol audit of the 07G candidate at `9dabf423` does not
  invalidate that narrow R1 fact, but it leaves all downstream gates closed.
- Do not build or replay a main-experiment cache, run a real-model preflight,
  train T0, evaluate a continual-learning arm, or start any GPU process until
  an accessible frozen source/transfer reference and session-atomic NAVSIM
  manifests pass an independent audit and the controller explicitly opens the
  corresponding gate.
- Controller candidate `3013da9bd0189b599c558b25fd3f7a804de28f37`
  is an informational coordination snapshot only. It was not published over
  an accessible source reference and must not be fetched, reconstructed, or
  treated as execution authorization.
- Never infer authorization from a passing test, a reviewer's conditional
  source-only opinion, a result on nuScenes, or the availability of idle GPUs.

## Preserved non-main asset

- Preserve without deletion or overwrite:
  `/home/xlxia/datasets/navsim-exp/training_cache_failure_patch_cl_v1_d2238dbb`.
- This 9.5 GB cache belongs only to the superseded portable 10,444-token
  manifest. It is forbidden for the pristine NAVSIM T0 or main experiment.
- `a1e1c77acd886b23324d8d0f4f0ad6fb9b07ea6388dd9c3feb8f7b182d963b8a`
  is the report-computed `post_scan.canonical_cache_index_sha256`, derived
  from ordered token paths and builder-file sizes. It is not the SHA256 of a
  `cache_index.json` file. Do not invent or cite such a file path.

## Method and evidence contract

- Tier A/B method status, prerequisites, and downgrades are frozen in the
  execution objective. Do not silently substitute or promote methods.
- Every comparison must bind unique current/old exposure, forward and backward
  counts, optimizer updates, student and teacher queries, wall time, peak
  VRAM, and stored bytes. Record both unique sample identities and repeated
  presentations where applicable.
- Every run and evaluation receipt must fail closed on registered method/arm,
  seed, immutable config and protocol hashes, run identity, checkpoint,
  evaluator source/environment, completed result, and registered evaluation
  domain.
- Claim receipts and underlying result/CSV/checkpoint evidence must be
  content-deduplicated, not merely path-deduplicated. A checkpoint and CSV must
  be joined by an evaluator-run receipt proving that the checkpoint generated
  that CSV.
- No result is claim-bearing until the current claim-protocol P0/P1 defects
  have executable negative tests and an independent frozen review passes.

## Compute and repository safety

- Do not fetch an unpublished controller reference. A future handoff must name
  an accessible full commit, tree, transfer-manifest path, and complete SHA256
  chain before any adaptation.
- Do not modify upstream `navsim/` tracked source merely to document scope.
  Training-code changes require their own reviewed implementation commit after
  the protocol gate opens.
- Before any future GPU action, re-read all applicable `AGENTS.md`, inspect all
  eight GPUs and their process owners, and use only a genuinely idle card.
  Never interrupt another user's process. GPU 1 has hosted VLLM and must not be
  assumed free from stale information.
- Preserve unrelated and untracked files. Never delete or overwrite data,
  caches, checkpoints, receipts, or reports without exact authorization.
- Keep each gate transition atomic and auditable: report branch, parent and
  full commit, changed-file hashes, verification output, clean/dirty status,
  publication status, and the exact remaining blocker.

# NAVSIM scene/log semantic audit

Recorded: 2026-08-18

## Decision

**Official NAVSIM v1.1 source rules out treating `scene_token` and segmented `log_name` as an
established identity. The real NAVSIM manifest remains blocked pending an exact R1 mapping.**

This is a read-only source audit. It does not read NAVSIM or nuScenes data, generate membership,
revise the frozen producer, authorize Failure-Patch inputs, or unlock source replay or execution.

## Source identity

The controller queried only the official repository. `refs/tags/v1.1` resolves to
`0811876c274e8b058ab2be9b3dcd4d37bd23f177`, tree
`db55d2e6f03fadbcae1c6c34207063eea00a4cae`; `refs/heads/v1.1` resolves to
`3e8291bfa89ff247231e0227778840cd0a036896`, tree
`6f930e75636f0ea9ee911c4e1bdfd8d4381f06f9`. The two relevant files are byte-identical at both
identities:

- `navsim/common/dataloader.py`: blob `66332b6365f95899a5ec1d5cf741c21bf25f0769`,
  SHA256 `f9c1af29cd08d3da7b8b892db0e1d6676b0ab3aab0447c36a474b4a8c08e0696`;
- `navsim/common/dataclasses.py`: blob `7e8321a316912a320ad2db761db1848c1b0a5552`,
  SHA256 `17d96f48101982d4a0842bb4cc1de441672207ad9fed3e460bbbdcf74440a32b`.

## Semantic result

The official loader reads one log pickle, splits its frame list into filtered scene windows and keys
each window by the center-frame `token`. It separately groups those tokens by the source-row
`log_name`. `SceneMetadata` independently stores `log_name`, `scene_token` and `initial_token` from
three source-row keys. The source therefore supplies no rule equating `scene_token` with a segmented
log name.

The current requirements list scene-token counts of 978/1,192/214, numerically equal to the
segmented-log counts, but the accepted R1 session ledger has no scene-token field. Those equal counts
remain unproven and cannot support manifest acceptance.

## Required closure evidence

The identity-alias option is closed. A later evidence intake must provide one content-addressed row
for every accepted R1 token containing at least:

- exact frame `token`;
- exact segmented `log_name`;
- source `scene_token`;
- official partition and complete timestamp-vehicle session identity.

The controller must independently recompute token coverage, unique log and scene sets, per-partition
counts and hashes, and session closure. Only after that replay may a forward requirements update bind
the exact scene-token expectations. The frozen producer and requirements are not edited by this
audit.

## Gate boundary

Exact-RC4 zero-P0 xhigh review, authoritative NAVSIM Failure-Patch inputs, exact NAVSIM scene
mapping, complete NAVSIM membership, authoritative nuScenes origin/N0 evidence and complete nuScenes
membership remain independent blockers. 06G source replay, dataset/cache/model/T0/CUDA/GPU,
training, evaluation, final-test and performance-claim gates remain stopped.

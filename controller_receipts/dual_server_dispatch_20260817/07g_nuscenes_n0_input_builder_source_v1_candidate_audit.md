# 07G nuScenes N0 input-builder source-v1 candidate audit

Date: 2026-08-18

## Decision

**PASS for input-specification builder source identity only, with two documentation-scope
corrections required before paper or publication promotion.**

The exact candidate is `2734b602e6ed92b6c1111bb36b4a9c02eacd471a`, sole parent
`ac38c9ffca9d4a428cee97606a92f93fb5ff8d82`, tree
`cc8780eabb763ee8d05f355e96df55d71ecf4d82`. It is one non-merge commit containing exactly the
four authorized paper, CLI, library and test paths, and its worktree is clean. The controller
freezes that exact source identity and the CLI/library content hashes.

This does **not** authorize construction of a real input specification or any real nuScenes read.
Independent upstream release provenance is still absent. A later, separate controller decision must
bind one canonical real request, the upstream-origin receipt and all six explicit nonnested roots
before real-mode construction can begin.

## Verification

- source executor: 185/185 exact candidate, adjacent provenance and manifest tests passed in 126.44
  seconds; output SHA256
  `9a6232f44a5aa8307e1fdaf042ac4c74b1f2c4ae9faf601724ef76921ee7ea0f`
- independent controller detached replay: 185/185 passed in 127.01 seconds; output SHA256
  `86ad1b018c4c3370475dce65b5f254e5a1b1fa01c5a74bd967c9ffd604c3f264`
- `git show --check`, exact path/hash inventory, AST checks and source-surface scans passed
- no shared-root discovery, recursive walk, model/torch/CUDA import or hardcoded real-data root is
  present; directory listing is limited to staged/published output verification and cleanup
- the detached verifier remained clean and created no Python bytecode or pytest cache path

The evidence-mode contract distinguishes `synthetic_fixture` from `controller_authorized_real`.
Real mode requires an addressed controller authorization whose SHA256 binds the canonical request
subject with only the authorization object removed. Reusing authorization after any other request,
root, source or content-address mutation is rejected.

## Accepted implementation boundary

Source-v1 accepts six explicit, distinct and nonnested request, metadata, devkit-package,
devkit-dist-info, data and source roots. It binds required metadata/devkit/split identities and
derives registered sensor, map and CAN expectations only from trainval-reachable tokens. It does not
discover roots or recursively inventory shared sensor trees.

Two paper statements are broader than the accepted implementation and must be corrected before the
paper or these source bytes are promoted:

1. Source-v1 accepts an explicitly addressed **extracted metadata directory only**. It does not
   implement or accept an archive input mode.
2. In real mode, this builder validates the graph needed to derive explicit sensor/map/CAN path
   expectations. It does not itself prove complete all-table closure. That proof remains the later,
   separate frozen script-60 N0 gate. Synthetic mode runs script 60 only as a compatibility replay.

These limitations do not create a source-interface bypass because no real evidence is accepted at
this gate; they are recorded as mandatory scope corrections, not silently promoted claims.

## Remaining gate

No raw nuScenes/NAVSIM data was read, no real input specification or N0 receipt was generated, and
no manifest, cache, model, checkpoint, T0, CUDA/GPU, training, evaluation or final-test operation was
performed. The exact-RC4 context-free `gpt-5.6-sol`/xhigh review still has no valid zero-P0 verdict.
Both complete session-atomic manifests and a later explicit controller gate remain prerequisites for
any 06G source replay.

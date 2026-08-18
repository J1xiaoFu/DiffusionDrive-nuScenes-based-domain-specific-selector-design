# 07G nuScenes N0 input-builder documentation-scope v2 candidate audit

Date: 2026-08-18

## Decision

**PASS for the authorized documentation correction only.**

The exact candidate is `b4c3ffd8b197c6fdd4a7ef7416cc8852714a5717`, sole parent
`2734b602e6ed92b6c1111bb36b4a9c02eacd471a`, tree
`068b319424359eb43bb35fc24948ea52df3c5c51`. It is one non-merge commit changing only
`paper_iterations/P001_scheduler_session_contract/NUSCENES_N0_INPUT_BUILDER_SOURCE_V1.md`; its
worktree is clean.

The corrected document now matches the accepted implementation boundary:

1. source-v1 accepts one explicitly addressed extracted metadata directory; archive input is not
   implemented or accepted;
2. real mode validates only the graph needed to derive explicit sensor, map and CAN path
   expectations; complete all-table closure remains the later frozen script-60 N0 gate; and
3. synthetic script-60 replay proves interface compatibility only.

## Identity and replay

- corrected document SHA256:
  `431a18c660667909323266e3df59110e725ab202ba62f206dd8bef1c97f6866a`
- full-index diff SHA256:
  `fb8e53c1ebaa2b224105b61618e8515b01e25ce79fe1eb659c785cded993796c`
- exact one-path inventory SHA256:
  `a3f2acf4723dc29c463deedb85be49697f4e834f720102c37e84e915bef3fe1f`
- CLI, library and input-builder test Git blobs are identical to base; their SHA256 values remain
  `5796fb47…61242`, `d802e854…f0100` and `e65dadf6…a7169`
- independent exact-commit detached replay: 185/185 input-builder, provenance and continual-protocol
  tests passed in 127.15 seconds; stdout SHA256
  `f138d3d94dac3e51e779c3b8da1ac88c0ec91e4e89f68bb3c90032b8857a437e`
- `git show --check` passed; the detached verifier stayed clean and created no Python bytecode or
  pytest cache path

## Evidence boundary

This closes only the two wording discrepancies recorded in the source-v1 audit. It does not promote
the paper, publish source, authorize archive support, establish upstream dataset origin, read real
nuScenes/NAVSIM content, construct a real input bundle or N0 ledger, or create a session manifest.

The upstream-origin decision remains with the human auditor. The exact-RC4 context-free
`gpt-5.6-sol`/xhigh review is still waiting for local approval and has no verdict. Complete frozen
session-atomic NAVSIM and nuScenes manifests and a later explicit controller gate remain mandatory
before any 06G source replay, cache, model, T0, CUDA/GPU, training or evaluation operation.

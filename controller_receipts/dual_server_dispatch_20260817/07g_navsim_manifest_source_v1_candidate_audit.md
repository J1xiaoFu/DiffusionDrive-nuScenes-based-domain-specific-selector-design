# 07G NAVSIM session-manifest source v1 candidate audit

Date: 2026-08-18

## Decision

**PASS for a separate approved-producer identity freeze only.**

The exact source candidate is `64447d3a9cc60f61b8596aed41086b5b4311c889`, sole parent
`653e797a2381cdf017466ef95aefcc41961f6715`, tree
`5ce81ef06d71189d1ff46676ecc1b4d01e415728`. It is one non-merge commit with 12 changed paths and a
clean worktree. The seven normative controller-contract files match controller commit
`2c0694213b1db0289dd3fec9094d3f1531ee9a23` byte-for-byte.

This audit does not itself approve the producer. A separate forward controller commit must bind the
exact commit, tree, producer path/hash and validator path/hash. Even after that identity freeze,
scene identity, Failure-Patch authority, real input transfer and real manifest generation remain
separate stopped gates.

## Independent replay

- exact candidate producer suite: 44 passed in 4.64 seconds; stdout SHA256
  `ab022635f1110d3ff024e7942a0a37f47789bfc432a943d9d2571c327175c9cf`
- byte-identical controller-validator suite: 19 passed in 0.56 seconds; stdout SHA256
  `d0654e6b420aef5ffb33b81c839cd73cd87307576aa967d1be275e3ca8fe8bce`
- independent full repository replay: 143 tests and 132 subtests passed in 304.57 seconds
- CUDA hidden, bytecode disabled and pytest cache provider disabled; the temporary detached audit
  worktree was removed after replay
- `git show --check`, changed-Python AST parsing and changed-JSON parsing passed

## P0 closure on the synthetic source surface

The producer consumes the exact content-addressed R1 session ledger and accepted receipt bytes.
Historical cache/pickle/filename membership and generic metrics CSV interfaces now reject directly.
Every `session_key` is an atomic unit; segmented logs and tokens are exact descendants, duplicate or
cross-session descendants fail, and official-validation sessions are excluded from both protocol
assignment artifacts.

Because the accepted R1 ledger has no independent `scene_token`, the producer does not silently
alias segmented logs to scenes. It requires either a controller-authorized semantic proof bound to
the exact ledger or a complete, unique, content-addressed log-to-scene mapping. Failure-Patch stage
calculation likewise requires controller-frozen raw stream, receipt, access ledger, evaluator,
metric/domain registries and canonical selection-input hashes.

Output is written into a private staging directory, validated with the frozen controller validator,
re-read to detect post-validation mutation, checked against an exact file inventory and published
with atomic no-replace semantics. Synthetic mid-write, post-validation and pre-publication attacks
leave no accepted output.

## Remaining authority boundary

The committed production requirements intentionally retain blocked producer, scene-identity and
Failure-Patch authority fields. Real R1 input objects have not been transferred to this source lane,
and no real manifest has been generated. A valid exact-RC4 `gpt-5.6-sol`/xhigh zero-P0 verdict is
also still absent.

Accordingly, real manifest generation, publication/transfer, 06G source replay, cache, model, T0,
CUDA/GPU, training, evaluation, final-test access and performance claims all remain stopped.

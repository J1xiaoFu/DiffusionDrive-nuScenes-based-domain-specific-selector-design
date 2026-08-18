# Controller branch publication replay

- Local controller identity: `codex/iclr2027-dual-server-controller` at
  `1175e026c33c744891fd1ad22585fdacb7fd4c77`, tree
  `903b413ea358e39f2a222cfde6b855c16ca09c5c`.
- Public controller ref: `05cbe17679f144a5b4effa7c27a6360f2de14376`, sole parent
  `f93e53868b322b391b8a43ce4ff96ffae823f3cb`, tree
  `825f3fca4d07ca230aeba13d5669b1e8a861c8ad`.
- The public ref was created and advanced without force. No local branch, ref, index or worktree was
  moved by publication.
- The local controller base was not present in the publication repository, so this is the established
  receipt-snapshot publication form rather than a claim that the unrelated whole trees are equal.
- Exact controller receipt subtree on both sides:
  `48d2581fee646f6a53f47c3ced2f370566e8fca7`.
- Isolated public replay passed all 88 `SHA256SUMS` entries. Replay-output SHA256:
  `44f88680354f7975d463db6c3634d2e42c89870bbb762057ea94e6b2360be009`.

Decision: **CONTROLLER RECEIPT SNAPSHOT PUBLICATION PASSED; NO EXECUTION PROMOTION.** The publication
proves public reachability and exact receipt-subtree identity only. It does not authorize NAVSIM or
nuScenes manifest generation, source replay, cache, model, T0, CUDA/GPU, training, evaluation,
final-test access or claims.

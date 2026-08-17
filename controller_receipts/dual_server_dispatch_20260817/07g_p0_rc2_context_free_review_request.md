# Drive-OPD P0 RC2 context-free blind-review request

## Review identity

- Review ID: `P001-DRIVE-OPD-P0-RC2-EXACT`
- Required model: `gpt-5.6-sol`
- Required reasoning effort: `xhigh`
- Required isolation: a newly created task with no parent conversation, forked turns or author summary
- Mode: read-only source/CPU audit
- Exact target: `d527274d368b6d6def5809b002324b31e88ae1bd`
- Target parent: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`
- Target tree: `dce36d261b8a0f4bde4519a3a393471c54b05356`
- Public source commit: `d262e3c14ba2bea31b48bd85ab1d721987b826b3`
- Public source parent: `fda179f007e64614467791f1ab9ed3c229c8f0bb`
- Public source tree: `c125b76a2f579016c2d4bcdafd9eb0db996f0a15`
- Contract: `controller_receipts/dual_server_dispatch_20260817/07g_p0_rc2_context_free_review_manifest.json`

Do not substitute a branch tip, precursor, uncommitted worktree or summary for these objects. Resolve
the public commit independently and verify the 41-file manifest and logical bundle before behavioral
review.

## Independence protocol

First inspect exact code, configs and tests and record candidate findings. Only then inspect
`RC1_INTERNAL_REVIEW.md`, the RC2 response and paper audit files. Existing PASS labels and tests are
evidence to challenge, not conclusions. Use isolated scratch for any temporary negative fixtures and
make no tracked writes, commits, pushes, branch/ref changes, merges or rebases.

## Mandatory RC1 finding reaudit

Independently determine whether RC2 closes each defect that rejected RC1:

1. canonical ledger-backed final-test authorization, full linear history and strict Git/UTC ordering;
2. production observations measured from every actual main and A-GEM replay batch, never copied from
   a target budget;
3. deterministic executable replay of seed power/calibration from pilot inputs, invocation and exact
   program identity;
4. ALER query counts captured from the actual search-and-repair path before the optimizer step;
5. multiple registered method-versus-common-baseline comparisons over a shared evaluation cell;
6. runtime-measured transient storage rather than a declared literal.

Attempt new bypasses beyond shipped fixtures, especially alternate/intermediate ledger histories,
authorization-time edge cases, loader-delivery and replay-buffer discrepancies, and semantically
altered seed-design inputs or program identity.

## Broader claim-integrity audit

Audit immutable run registration, checkpoint-to-evaluator-to-CSV lineage, evidence reuse and semantic
deduplication, method/metric/domain binding, session/stage chronology and leakage, confirmatory family
construction and multiplicity, filename/order invariance, and Drive-OPD/LwF loss and query/step budget
neutrality. The official one-batch real-adapter equivalence check remains unopened and must not be
reported as passed.

Disposition all twelve historical requirements from the machine-readable contract as `closed`,
`open_p0`, `open_p1`, `open_p2` or `not_source_testable`, with exact file/line or executable evidence.

## Required CPU replay

With `CUDA_VISIBLE_DEVICES=''`, bytecode disabled and the pytest cache provider disabled, run:

```text
selector_bench/tests/test_navsim_continual_protocol.py
selector_bench/tests/test_drive_opd_losses.py
selector_bench/tests/test_drive_cl_baselines.py
selector_bench/tests/test_drive_cl_statistics.py
selector_bench/tests/test_drive_cl_claim_protocol.py
selector_bench/tests/test_drive_cl_run_budget.py
```

Also run `selector_bench/scripts/57_audit_drive_cl_p0_attacks.py`. Verify that every mutation exits
nonzero before a claim output exists and that the positive fixture passes. Record interpreter,
commands, exit codes, counts, time and SHA256 for stdout, stderr and generated JSON. Do not read/build
a cache, access raw data, import a real model or invoke CUDA/GPU.

## Required verdict

Return Markdown plus a machine-readable JSON receipt, score 1–10, confidence 1–5, findings with exact
references and reproductions, and verdict `PASS` or `FAIL`. Answer explicitly:

- A: Does exact RC2 satisfy the source/CPU P0 claim-integrity contract?
- B: Is it safe for non-promoting source-only publication and isolated replay?
- C: Does this review unlock any dataset, cache, model, T0, GPU, training or evaluation gate?

PASS requires no open P0, independent closure of every source/CPU-testable historical P0, successful
identity/hash and CPU/harness replay, explicit disposition of P1/P2, and an honest gate boundary.

Regardless of verdict, answer C must remain NO. A PASS can clear only the exact-RC2 review prerequisite
for a later controller decision. Dataset provenance/adapters, the session-atomic CL manifest, cache,
real-model preflight, T0, CUDA/GPU, training, evaluation and performance claims remain stopped.

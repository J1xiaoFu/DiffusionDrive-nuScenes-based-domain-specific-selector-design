# Drive-OPD P0 RC context-free blind-review request

## Review identity

- Review ID: `P001-DRIVE-OPD-P0-RC-EXACT`
- Required model: `gpt-5.6-sol`
- Required reasoning effort: `xhigh`
- Required task isolation: a newly created task with no parent conversation, no forked turns and no
  preloaded prior-review summary
- Review mode: read-only source/CPU audit
- Target RC: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`
- Target parent: `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`
- Target tree: `0e6e9a0adda280765ce41ab377de9bbbaea0cff1`
- Target branch label: `codex/iclr2027-drive-opd-07g-p0-rc`
- Machine-readable contract:
  `controller_receipts/dual_server_dispatch_20260817/07g_p0_rc_context_free_review_manifest.json`

The enclosing Git publication commit and `SHA256SUMS` entry bind this request and its manifest.
Do not substitute a later branch tip, an earlier precursor, an uncommitted worktree or an author
summary for the exact target above.

## Independence protocol

Start from this request in a fresh task. Do not read the controller conversation, inherit a parent
task transcript, ask an author agent for conclusions or treat existing PASS labels as findings.
Independently fetch or resolve the immutable published source and verify all advertised identities
and hashes before reviewing behavior. Use a detached or temporary extraction. Do not modify a
tracked source ref, controller ref or execution branch.

Conduct the review in two phases:

1. Inspect the exact code, configs and tests independently and record candidate findings before
   reading the author response or prior internal audit.
2. Then inspect the exact-RC paper audit files and explicitly disposition every historical required
   fix against the implementation and adversarial replay. A response document is evidence of intent,
   never proof of closure.

## Immutable evidence anchors

- Public source commit: `02e2784cb39cd636af6f582c2ff3f72d9274c58f`
- Public source parent: `94a30b148e3a7c6fe1ecdd2f8ce493ab8440946b`
- Public source tree: `b95c40170c29c8408cc258ea22a173d2167d8a66`
- Final publication receipt commit: `d1665712535e101c948bec2bb7f7c21e64fe1f36`
- Final publication receipt tree: `7ba0facb01a1dc7f4354d32964127a252f5dc2c2`
- Transfer manifest SHA256:
  `70b9f99bc96633a6be9a9437b106e48dc1d8effdac55fbc8439869c3dbc2ed78`
- Canonical 37-file source-bundle SHA256:
  `ee681bf3cdb0cc2dfd198f860ce927971d738904250daacb2d8cfbf47349de93`
- Controller exact-RC acceptance commit: `4e5822bc68744d0bfecbb620a04fea4e63315c7b`
- 06G destination-replay receipt SHA256:
  `89503b3763e31a8383fe02bb6746783344b474a2cd8e11108730466fdef045e5`
- 06G destination-replay audit commit: `7bcfbf5f173cc567228904bc76413c508194dfe7`

The destination replay passed public reachability, byte identity, 49 CPU tests and the P0 fixture
harness. It did not validate dataset provenance, a session-atomic CL manifest, the real adapter,
model execution, training, evaluation or performance.

## Required audit

### Identity and evidence

1. Verify the exact RC commit, sole parent, tree and one-commit distance from its declared base.
2. Verify the source publication commit/parent/tree and the final publication receipt identity.
3. Verify the transfer manifest SHA256, every one of its 37 path hashes and the canonical logical
   source-bundle hash. Report missing, extra or mismatched paths.
4. Verify the reconstructed 06G destination receipt SHA256 and audit its boundary claims. Do not
   infer a data, model or performance result from a source/CPU replay.

### Claim-integrity protocol

5. Audit run identity and registration for uniqueness, immutability, semantic binding and resistance
   to aliasing or replay.
6. Audit checkpoint-to-evaluator-to-CSV lineage, metric and method registries, evidence inventory,
   filename/order invariance and rejection of missing, reused, duplicated or stale evidence.
7. Audit domain derivation, stage chronology, session boundaries and resistance to future-domain or
   future-stage leakage.
8. Audit confirmatory seed design, crossed comparisons, metric directionality, family construction,
   multiplicity correction and the claim-root selection rule.
9. Audit Drive-OPD/LwF runtime neutrality: teacher stop-gradient, official-plus-distillation loss,
   matched teacher-query and optimizer-step budgets, and method-arm binding. Treat the official
   one-batch real-adapter equivalence check as an unopened execution gate, not a source-level PASS.
10. Independently attempt plausible bypasses beyond the shipped fixtures. Temporary tests and
    mutated copies are allowed only in isolated scratch space and must not alter the target refs.

### Mandatory historical dispositions

After the independent pass, disposition all twelve historical requirements:

1. unique immutable run identity and rerun/reuse prevention;
2. evaluator receipt binding checkpoint through evaluation output;
3. rejection of reused evidence;
4. method-registry binding of an experiment to its arm;
5. evaluator-derived domain identity;
6. confirmatory verification of the frozen family;
7. claim-root selection independent of filename suffix;
8. semantic and evidence-identity deduplication;
9. chronology binding;
10. confirmatory seed power/calibration design;
11. complete distillation-loss and budget audit, while keeping the official real-adapter batch gate
    closed;
12. unambiguous round status and evidence boundary.

For each item, return `closed`, `open_p0`, `open_p1`, `open_p2` or `not_source_testable`, with exact
file/line or test evidence and a concise rationale.

## Required CPU replay

In an isolated extraction with GPU visibility disabled, bytecode disabled and the pytest cache
provider disabled, run these five suites against the exact published source:

```text
selector_bench/tests/test_navsim_continual_protocol.py
selector_bench/tests/test_drive_opd_losses.py
selector_bench/tests/test_drive_cl_baselines.py
selector_bench/tests/test_drive_cl_statistics.py
selector_bench/tests/test_drive_cl_claim_protocol.py
```

Also run `selector_bench/scripts/57_audit_drive_cl_p0_attacks.py` and verify, rather than merely
repeat, that all mutated CLIs return nonzero before a claim output exists and that the positive
fixture succeeds. Record commands, interpreter identity, exit codes, counts, elapsed time and
SHA256 of stdout, stderr and produced audit JSON. Do not read or build a cache and do not import a
real model.

## Finding and verdict contract

Classify findings as:

- `P0`: an executable claim-integrity bypass, target/evidence substitution, leakage, invalid
  statistical promotion, unfair method comparison, or false gate promotion;
- `P1`: a material defect that blocks the next data/adapter gate but does not currently permit a
  claim because all execution gates remain closed;
- `P2`: bounded maintainability, clarity or defense-in-depth work with no current claim consequence.

Return:

- a Markdown review with exact file/line references and reproduction steps;
- a machine-readable JSON receipt conforming to the output fields in the review manifest;
- score from 1 to 10, confidence from 1 to 5, and verdict `PASS` or `FAIL`.

`PASS` requires no open P0, independent closure of all source/CPU-testable historical P0 items,
successful immutable identity/hash replay and an honest boundary around non-executed gates. Every P1
and P2 must be explicitly dispositioned. A test pass alone is insufficient.

## Non-promotion boundary

Even a review PASS clears only `fresh_context_free_exact_rc_review`. It does not authorize or imply
dataset provenance, cache construction, a session-atomic CL manifest, real-model preflight, T0, GPU,
training, evaluation or a performance claim. Those require separate controller freezes and receipts.

The reviewer must make no writes, commits, pushes, fetch-side ref changes, merges, rebases or branch
moves in an owned source worktree. Network fetch to an isolated temporary destination is permitted
only for immutable evidence verification.

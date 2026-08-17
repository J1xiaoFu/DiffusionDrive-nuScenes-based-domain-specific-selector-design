# Drive-OPD RC2 blind review

**Score: 3/10 — Reject**
**Verdict: FAIL**
**Confidence: 5/5**

Exact RC2 does not satisfy the source/CPU P0 claim-integrity contract. The frozen tests pass, but four independently reproduced executable bypasses remain:

1. non-monotonic Git/access chronology is accepted;
2. sealed-test authorization from family A can be reused in family B;
3. a one-comparison family passes without the required common-baseline multi-method cell;
4. an infinite minimum relevant effect passes executable seed-design replay with estimated power 1.0.

No independent P1 or P2 findings remain after accounting for these P0s.

## Gate decisions

- **A — Source/CPU P0 closed? NO.**
- **B — Safe for non-promoting source-only publication and isolated CPU replay? YES, conditionally**, only when clearly labeled rejected/non-promoting.
- **C — Any dataset, cache, model, T0, GPU, training, evaluation or performance gate unlocked? NO.**

## Identity and hash audit

All immutable source identities passed:

- Review freeze: `81973da8c2768e71f5f79cefc17888e19d2f6cbd`
  - parent: `d262e3c14ba2bea31b48bd85ab1d721987b826b3`
  - tree: `211d50d5e31b517a0062012068c07d5169d757d9`
- Logical source: `d527274d368b6d6def5809b002324b31e88ae1bd`
  - parent: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`
  - tree: `dce36d261b8a0f4bde4519a3a393471c54b05356`
  - one commit from RC1; 22 changed paths
- Public source: `d262e3c14ba2bea31b48bd85ab1d721987b826b3`
  - parent: `fda179f007e64614467791f1ab9ed3c229c8f0bb`
  - tree: `c125b76a2f579016c2d4bcdafd9eb0db996f0a15`
  - 20 changed paths
- Independent `ls-remote` returned the exact review freeze at the public branch.
- The stale local publication ref pointed at `389e07b…`; it was not used or modified.
- 28/28 SHA256SUMS receipts verified; 29 adjacent files including `SHA256SUMS`.
- 41/41 logical blobs and 41/41 public blobs matched the transfer manifest.
- Logical/public source-blob differences: zero.
- Transfer manifest: `9eadb5064f5bebd8234e910fb3a6e6aa8dc16aeb7d077fbd30803cda0af66932`
- Recomputed logical bundle: `5bc6f3339f1af5ccde9c21a8068b94053ae00ea920eed9f3ade7613d70a5cd88`
- Canonical bundle algorithm: `958d537d6707f130f72834790ba8bdfcd3192431d214d5fdb4241ea61e023a78`
- Review manifest: `619a12ba27fe1c140da6f1c5b7c1e7019ee7ae91a115ea9969f4cf467abe669d`
- Public full-index diff: `25e031f4ef7b78cd99b2b8e7ee3e0ecdf027a79560310366a7adb0d41ee37362`
- Sorted changed paths: `acac582db6b9086505058c4c776cbaccf660bd921ba149037c84d02a03d8217a`
- Public/local-equivalent content diff: empty.

The source worktree was clean at the initial check. No repository or ref writes were performed; generated artifacts were confined to `/tmp`. A final status command was not rerun because the report-only continuation explicitly prohibited further commands.

## CPU replay

Interpreter:

`/home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10`

The exact patch-level `--version` output was recorded in `/tmp` but is unavailable in the retained report context.

Each suite was run independently with:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONHASHSEED=0 \
PYTHONPATH=/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench \
/home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10 \
-m pytest -q -p no:cacheprovider <suite>
```

| Suite | Result | Measured elapsed | stdout SHA256 |
|---|---:|---:|---|
| `test_navsim_continual_protocol.py` | 3 passed | 207,697,190 ns | `c2cba0e5b1275bde99432021fd22a26635493fa35d0886fcd5cd38708b0f7cca` |
| `test_drive_opd_losses.py` | 13 passed | 1,608,852,777 ns | `0516f6ef8f9270b56985dab9c0c9c9877ee54ec0a71b4655372d2fbaa4415314` |
| `test_drive_cl_baselines.py` | 8 passed | 1,540,790,247 ns | `2963c080eaf15eb33f54f8ba947759f6f4106d0f0d10bcff7a1ae7e80667d24d` |
| `test_drive_cl_statistics.py` | 8 passed | 293,402,296 ns | `092afb193e0565b502e8c29ec6c9f2e9ed67befee919192e41fe9bd294ef0cc2` |
| `test_drive_cl_claim_protocol.py` | 23 passed, 21 subtests | 117,580,263,739 ns | `5bfe86292407dc4253af1a8faa74ec324d7192818593fb10beabe54b865570d2` |
| `test_drive_cl_run_budget.py` | 7 passed | 1,163,349,192 ns | `e555813f9983ff591aee5e97ab7ad4c928dcfe9fe0c82735ddd5a82a43ef21e1` |

Total: **62 tests and 21 subtests passed**. Every suite’s stderr SHA256 was the empty-stream hash `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Expanded harness command:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONHASHSEED=0 \
PYTHONPATH=/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench \
/home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10 \
/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/scripts/57_audit_drive_cl_p0_attacks.py \
--repo /tmp/drive-opd-rc2-review.EJtTp6/source \
--python /home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10 \
--output /tmp/drive-opd-rc2-review.EJtTp6/test_runs/p0_attack_audit.json
```

Results:

- exit `0`, status `PASS`
- elapsed: `115082767834` ns
- 15/15 attack nodes passed
- 32/32 mutation instances exited nonzero
- 32/32 had no claim output before rejection
- positive fixture exit `0`
- stdout: `f23c4734d85915869e4bd48a7aeeff3246d49143f10ef2cf282e66d0b3506d0f`
- stderr: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- generated JSON: `d68ba4fd2b86db2f481dc9656804995bcd8af47b76e111e4ca6c4196f818833e`

The harness therefore passes its shipped mutations, but those mutations do not cover the four accepted substitutions below.

## Open P0 findings

### P0-1 — Full Git/UTC chronology is not enforced

[54_run_drive_cl_navsim_evaluator.py](/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/scripts/54_run_drive_cl_navsim_evaluator.py:186) verifies a linear parent chain and unchanged intermediate ledger bytes. It does not inspect intermediate commit timestamps. Lines 255–260 check only the family and final access endpoints and use non-strict `<=`.

A temporary fixture inserted an unrelated intermediate commit dated `2030-01-01T00:00:00+00:00`, followed by an access commit dated `2026-08-17T16:06:20+00:00`. Authorization, family and access were also accepted at the same normalized instant. Evaluator exit was `0` and its receipt existed.

- stdout: `ae462c71425a2ace71583ee1ea6168e5af3839c0ca4fe688066fb7f781a739c7`
- stderr: empty-stream hash
- elapsed: `1237405624` ns

Exploitability: an access history can violate strict parent/child Git chronology and strict family/authorization/access ordering while still producing a verified sealed-test evaluator receipt.

Disposition: **open_p0**.

### P0-2 — Sealed authorization is not bound to the comparison’s family

[load_evaluator_run_contract](/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/selector_bench/continual/claim_protocol.py:1013) validates checkpoint, cell, CSV, domain and sealed-access status, but lines 1147–1152 only require `sealed_test_access.status == "verified"`. It accepts no expected family ID, family hash or freeze commit.

[52_compare_drive_cl_crossed_pdm.py](/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/scripts/52_compare_drive_cl_crossed_pdm.py:408) consumes those evaluator receipts without the current family identity; the separate confirmatory-family registration occurs at lines 632–653.

A minimal fixture evaluated every run under family `p003_primary`, then supplied those same evaluator receipts to a distinct frozen family `p003_distinct_family_b`. Comparison exit was `0`, the claim output existed, and the reproduction reported `family_mismatch_accepted: true`.

- stdout: `ce07a017d1764aa7dbcefebfad6ddf1b6855ea7e59478f43e9b8301da6c01b7e`
- stderr: empty-stream hash
- elapsed: `10060845931` ns

Exploitability: valid sealed-test evidence can be transplanted between confirmatory families, bypassing the family-specific access authorization.

Disposition: **open_p0**.

### P0-3 — The common-baseline multi-method requirement is optional

[validate_family_registries_and_cells](/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/selector_bench/continual/claim_protocol.py:420) requires only a nonempty comparison list. Lines 489–503 deduplicate semantic comparisons, and lines 505–530 require an off-diagonal cell, but there is no requirement that a cell contain two or more distinct candidate methods sharing one baseline.

A one-comparison family with only `drive_opd_fixed` versus `sequential` passed the family validator:

```json
{
  "validator_result": "ACCEPTED",
  "comparison_count": 1,
  "required_cell_count": 1,
  "baselines": ["sequential"],
  "candidates": ["drive_opd_fixed"]
}
```

- stdout: `7ab9ad4d8a9a8f3018316971511cf63497d0bee97b95051e5aced70a6e02660e`
- stderr: empty-stream hash
- elapsed: `1010174425` ns

Exploitability: a selectively chosen single method comparison can be promoted as the complete confirmatory family without the frozen common-baseline comparison structure.

Disposition: **open_p0**.

### P0-4 — Non-finite seed-design inputs pass exact replay

[seed_design.py](/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/selector_bench/continual/seed_design.py:203) rejects an effect only when `minimum_relevant_effect <= 0.0`; it never requires finiteness. The CLI has the same omission. [validate_seed_design](/tmp/drive-opd-rc2-review.EJtTp6/source/selector_bench/selector_bench/continual/claim_protocol.py:318) exactly replays the malformed input, and line 353 again checks only `<= 0.0`.

A frozen invocation using `Infinity` produced and validated:

```json
{
  "validator_result": "ACCEPTED",
  "design_status": "passed",
  "minimum_relevant_effect_is_infinite": true,
  "estimated_power": 1.0,
  "designed_seed_count": 9
}
```

- stdout: `4f0b331ba16cb726e8d88a30926a4a858b38f772f690c016173a8c2b679fb2c2`
- stderr: `8996aafcfb07c1e40b94a9d07c6e3e2c2a73ba4162c066a9cb1783341fa53b4c`
- elapsed: `1120565000` ns

The stderr contains NumPy invalid-subtraction warnings, but the validator still accepts the design.

Exploitability: a non-finite effect assumption trivially yields estimated power 1.0 while satisfying exact program, invocation and receipt replay.

Disposition: **open_p0**.

## Mandatory RC1 finding dispositions

| Mandatory item | Disposition | Evidence |
|---|---|---|
| Canonical ledger, full history, strict Git/UTC chronology | **open_p0** | Future-dated intermediate history and endpoint equality accepted. |
| Measured production/A-GEM delivery budgets | **closed** | Main deliveries measured before forward at training script lines 924–931; A-GEM delivery is compared with its plan and measured at 1069–1081; executed calls/queries are accumulated at 1138–1143. |
| Executable seed power/calibration replay | **open_p0** | Original handwritten-receipt attack is closed, but non-finite invocation replay passes. |
| ALER actual query capture | **closed** | Search and repair are inside `capture_query_audit` at lines 977–997; actual counts are emitted at 1013–1018 before `optimizer.step`. |
| Shared common-baseline comparison cells | **open_p0** | Multiple comparisons are supported but not mandatory; a one-comparison family passes. |
| Runtime-measured transient storage | **closed** | Atomic JSON/checkpoint temporary sizes are observed at training lines 172–178 and 366–374 and emitted at 1282–1286; zero is rejected by the claim contract. |

## Historical requirement dispositions

| Historical requirement | Disposition |
|---|---|
| Unique immutable run identity and rerun/reuse prevention | **closed** |
| Evaluator receipt binding checkpoint through evaluation output | **closed** |
| Rejection of reused evidence | **closed** |
| Method registry binding to executable arm | **closed** |
| Evaluator-derived domain identity | **closed** |
| Confirmatory verification of the frozen family | **open_p0** — access evidence is transplantable between families |
| Claim-root selection independent of filename suffix | **closed** |
| Semantic and evidence identity deduplication | **closed** |
| Authoritative chronology binding | **open_p0** |
| Executable confirmatory seed power/calibration design | **open_p0** |
| Complete distillation loss and measured budget audit | **closed at source/CPU** |
| Official one-batch real-adapter equivalence | **not_source_testable; unopened and not passed** |
| Unambiguous round status and evidence boundary | **closed** |

The shipped P0-01 through P0-15 mutations are closed against their exact fixtures. P0-13 and P0-15 remain incomplete because the new chronology and non-finite-input variants pass.

## Evidence boundary and next action

This review covers immutable source, configuration, synthetic CPU fixtures and `/tmp` Git reproductions only. It provides no NAVSIM or nuScenes provenance, cache validity, real-model behavior, one-batch official-loss equivalence, T0, CUDA/GPU execution, training, evaluation, score or performance evidence.

Required next action:

1. require strictly increasing normalized UTC times across every ancestry commit and strict `family < authorization < access < evaluator`;
2. bind each test evaluator receipt to the active family ID, family SHA256 and freeze commit at comparison time;
3. require at least two distinct candidates against one common baseline in every designated shared confirmatory cell;
4. reject all non-finite invocation, simulation and replay values and serialize JSON with non-finite values forbidden;
5. add these four reproductions to the frozen harness, publish a new exact forward RC, and obtain another context-free review.

```json
{
  "schema": "drive_opd.context_free_exact_rc2_review_result.v1",
  "review_id": "P001-DRIVE-OPD-P0-RC2-EXACT",
  "score": 3,
  "decision": "REJECT",
  "verdict": "FAIL",
  "confidence": 5,
  "identities": {
    "review_freeze": "81973da8c2768e71f5f79cefc17888e19d2f6cbd",
    "review_parent": "d262e3c14ba2bea31b48bd85ab1d721987b826b3",
    "review_tree": "211d50d5e31b517a0062012068c07d5169d757d9",
    "logical_source": "d527274d368b6d6def5809b002324b31e88ae1bd",
    "logical_parent": "7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4",
    "logical_tree": "dce36d261b8a0f4bde4519a3a393471c54b05356",
    "public_source": "d262e3c14ba2bea31b48bd85ab1d721987b826b3",
    "public_parent": "fda179f007e64614467791f1ab9ed3c229c8f0bb",
    "public_tree": "c125b76a2f579016c2d4bcdafd9eb0db996f0a15",
    "remote_public_head_verified": true
  },
  "hashes": {
    "receipt_rows_verified": 28,
    "receipt_hash_mismatches": 0,
    "source_blob_count": 41,
    "logical_blob_hash_mismatches": 0,
    "public_blob_hash_mismatches": 0,
    "transfer_manifest_sha256": "9eadb5064f5bebd8234e910fb3a6e6aa8dc16aeb7d077fbd30803cda0af66932",
    "logical_bundle_sha256": "5bc6f3339f1af5ccde9c21a8068b94053ae00ea920eed9f3ade7613d70a5cd88",
    "canonical_algorithm_sha256": "958d537d6707f130f72834790ba8bdfcd3192431d214d5fdb4241ea61e023a78",
    "review_manifest_sha256": "619a12ba27fe1c140da6f1c5b7c1e7019ee7ae91a115ea9969f4cf467abe669d"
  },
  "cpu_replay": {
    "interpreter": "/home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10",
    "interpreter_patch_version": "unavailable_in_retained_report_context",
    "suite_processes": 6,
    "test_functions_passed": 62,
    "subtests_passed": 21,
    "suite_failures": 0,
    "harness_exit_code": 0,
    "harness_status": "PASS",
    "harness_attack_count": 15,
    "harness_mutation_count": 32,
    "mutations_rejected_nonzero": 32,
    "mutations_without_claim_output": 32,
    "positive_fixture_exit_code": 0,
    "harness_stdout_sha256": "f23c4734d85915869e4bd48a7aeeff3246d49143f10ef2cf282e66d0b3506d0f",
    "harness_stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "harness_json_sha256": "d68ba4fd2b86db2f481dc9656804995bcd8af47b76e111e4ca6c4196f818833e"
  },
  "findings": [
    {
      "id": "RC2-P0-01",
      "level": "P0",
      "title": "non-monotonic Git and non-strict authorization chronology accepted",
      "disposition": "open_p0",
      "reproduction_exit_code": 0,
      "stdout_sha256": "ae462c71425a2ace71583ee1ea6168e5af3839c0ca4fe688066fb7f781a739c7"
    },
    {
      "id": "RC2-P0-02",
      "level": "P0",
      "title": "sealed-test authorization can be transplanted between families",
      "disposition": "open_p0",
      "reproduction_exit_code": 0,
      "stdout_sha256": "ce07a017d1764aa7dbcefebfad6ddf1b6855ea7e59478f43e9b8301da6c01b7e"
    },
    {
      "id": "RC2-P0-03",
      "level": "P0",
      "title": "single comparison satisfies the common-baseline family gate",
      "disposition": "open_p0",
      "reproduction_exit_code": 0,
      "stdout_sha256": "7ab9ad4d8a9a8f3018316971511cf63497d0bee97b95051e5aced70a6e02660e"
    },
    {
      "id": "RC2-P0-04",
      "level": "P0",
      "title": "infinite effect size passes executable seed-design replay",
      "disposition": "open_p0",
      "reproduction_exit_code": 0,
      "stdout_sha256": "4f0b331ba16cb726e8d88a30926a4a858b38f772f690c016173a8c2b679fb2c2",
      "stderr_sha256": "8996aafcfb07c1e40b94a9d07c6e3e2c2a73ba4162c066a9cb1783341fa53b4c"
    }
  ],
  "finding_counts": {
    "open_p0": 4,
    "open_p1": 0,
    "open_p2": 0
  },
  "gate_decisions": {
    "A_source_cpu_p0_contract": "NO",
    "B_non_promoting_source_only_publication_and_replay": "YES_CONDITIONAL_ON_REJECTED_NON_PROMOTING_LABEL",
    "C_execution_unlocked": "NO"
  },
  "execution_boundary": {
    "dataset": "STOPPED",
    "cache": "STOPPED",
    "real_model_preflight": "STOPPED",
    "t0": "STOPPED",
    "gpu": "STOPPED",
    "training": "STOPPED",
    "evaluation": "STOPPED",
    "performance_claim": "STOPPED"
  }
}
```

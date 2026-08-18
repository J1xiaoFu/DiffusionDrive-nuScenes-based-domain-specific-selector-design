# Fresh adversarial ICLR 2027 review

## Verdict

**Score: 3/10 — Reject**
**Confidence: 5/5**

The advertised immutable identities, public source transfer, six CPU suites, and 19-family attack harness reproduce. RC3 closes the four specific RC2 attacks and most historical lineage/sealing problems.

However, two independently executable P0 failures remain:

1. The seed-design receipt is calibrated for one anonymous seven-metric comparison, while the frozen confirmatory family requires two comparisons and 14 hypotheses.
2. The claim gate accepts malformed resource receipts—including `transient_bytes: true` and `wall_seconds: "nan"`—and I reproduced a complete 14-hypothesis global-Holm output from such evidence.

Therefore RC3 does not pass its frozen claim-protocol contract.

### Binary gates

| Gate | Decision |
|---|---|
| **A) Exact RC3 passes frozen P0 source/claim-protocol contract?** | **NO** |
| **B) Exact source-only publication/transfer/replay is safe?** | **YES, narrowly**—only as immutable source bytes and CPU fixtures, explicitly labeled rejected/non-promoting. |
| **C) Isolated 06G CPU/source replay warranted next?** | **NO**—repair the two P0s first; replaying known-failing source adds little. |
| **D) Any dataset/cache/real-model/T0/GPU/training/evaluation/performance-claim execution authorized?** | **NO** |

## Exact identity and publication replay

Local identity passed:

- Detached `HEAD`: `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`
- Sole parent: `d527274d368b6d6def5809b002324b31e88ae1bd`
- Tree: `3ec47127a6d7c85326bbcf1d72ef76e2971f100d`
- Named branch resolves to the same commit.
- Exactly 29 paths differ from the parent.
- Initial and final `git status --porcelain=v1` were empty; SHA256 of empty status: `e3b0c442…b855`.
- No `.pytest_cache` or `__pycache__` appeared.

I read AGENTS.md, the complete P001 round and preserved RC1/RC2 reviews, and all 29 changed paths: six paper files, one config, scripts 40–58, two continual-learning modules, and the claim-protocol test file.

Public identity also passed. I used:

```bash
git ls-remote https://github.com/J1xiaoFu/DiffusionDrive-nuScenes-based-domain-specific-selector-design.git \
  refs/heads/codex/iclr2027-dual-server-dispatch-publication

git fetch --no-tags \
  https://github.com/J1xiaoFu/DiffusionDrive-nuScenes-based-domain-specific-selector-design.git \
  refs/heads/codex/iclr2027-dual-server-dispatch-publication
```

The fetch updated only `FETCH_HEAD`:

- Public commit: `b1e1991afe7b9b14fd83f7510394c0ffb77ad700`
- Sole parent: `7892cbcd06b99f44dc593813459b12e5eb84f902`
- Public tree: `1090673c5e8f148f2ec1b127044fa306b51b19cd`
- Existing remote-tracking ref remained at `fda179f…`; no ref was created or moved.
- Public commit has 28 changed paths. I did not equate its whole tree with the local tree.

The transfer manifest replayed exactly:

- Manifest SHA256: `ff8ca94ebd3dd39abb048dabe8c8a87eaa383465b5cdb899d74cc6fbe2bcd437`
- `source_commit`: exact local RC3 commit
- Source files: **41/41 local hashes matched**
- Source files: **41/41 public hashes matched**
- Local/public source-byte differences: **0**
- Declared and recomputed bundle: `68259f07dc97db312c00c71dcd18a899502588b4eb303ee12cce585e1cc894b1`
- Manifest data gate: `BLOCKED_UNTIL_06G_REPLAY_RECEIPT_PASSES`

## P0 findings

### P0.1 — Seed design is not bound to the complete confirmatory family

The family requires two distinct candidates sharing one baseline and registers 14 hypotheses: two comparisons times seven metrics ([family config](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/configs/drive_cl_global_holm_family.example.json:25), [comparisons](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/configs/drive_cl_global_holm_family.example.json:31), [second hypothesis block](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/configs/drive_cl_global_holm_family.example.json:226)). The family validator correctly requires all seven metrics per comparison and at least two candidates against one common baseline ([claim_protocol.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:772)).

The seed-design path does not consume comparison IDs, candidate identities, the hypothesis manifest, or hypothesis count:

- Pilot input contains only one `candidate_minus_baseline[metric]` matrix ([seed_design.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/seed_design.py:237)).
- Null calibration takes a maximum over seven metric columns only ([seed_design.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/seed_design.py:283)).
- Tail resolution divides alpha by `len(PDM_DEFAULT_METRICS)`, i.e. seven ([seed_design.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/seed_design.py:406)).
- Yet downstream Holm consumes all 14 registered p-values ([50_apply_drive_cl_global_holm.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/50_apply_drive_cl_global_holm.py:571)).

My executable reproduction produced:

- comparisons: 2
- hypotheses: 14
- implemented tail floor: 9 seeds
- full-family version of the stated rule: 10 seeds
- \(2^{1-9}=0.00390625\)
- first Holm threshold: \(0.05/14=0.00357143\)

Thus the paper’s stated discrete rule cannot cross the first Holm threshold for the actual family. Moreover, the downstream p-values are Monte Carlo bootstrap p-values with floor \(1/(R+1)\), not the stated sign-resolution floor ([statistics.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/statistics.py:506)). The seed-design calculation and final inferential procedure are internally misaligned.

Reproduction stdout SHA256: `b28bc3fde2cfeef0848d5133ed6e110df20ea21316ac224ecc4941f1fd3b306f`.

This is not merely low power. A favorable pilot for one unnamed candidate can authorize the same seed count for every candidate comparison without joint family calibration.

### P0.2 — Invalid resource receipts remain globally claim-eligible

Production code measures transient storage from temporary artifacts ([run_budget.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/run_budget.py:85)) and emits the measured value ([41_train_drive_cl_diffusiondrive.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:1287)). The validator, however, uses coercive `float()` and `int()` checks ([claim_protocol.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:1543)).

I verified that `load_training_run_contract` accepts all of:

- `transient_bytes: true`
- `transient_bytes: 1.5`
- `wall_seconds: NaN`
- `wall_seconds: "nan"`
- `persistent_bytes: true`

Direct-bypass stdout SHA256: `d3e3633854f9a9758ea6baeef4e699264e8ba2f76fcb5a2a89ec077e87b0713e`.

I then constructed the full synthetic confirmatory chain with:

```json
"transient_bytes": true,
"wall_seconds": "nan"
```

The evaluator, both comparison producers, evidence inventory and global Holm all exited zero. Two confirmatory comparison receipts and a final 14-hypothesis global result were emitted.

- End-to-end stdout SHA256: `ed39777ac687dc5da674bf39c84623e54b26eac3513089b21d0eee92a5cd46c8`
- Emitted global-result SHA256: `4ba3e605804fa11b9696c793aef9caf7d1bccb01fdf23f19bdfc695ccf08c15d`

The existing test only checks that integer zero is rejected ([test_drive_cl_claim_protocol.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/tests/test_drive_cl_claim_protocol.py:1632)). Hash lineage cannot repair this: if malformed bytes exist before evaluator execution, every downstream receipt faithfully hashes the malformed object.

## P1/P2 findings

### P1 — ALER equivalence coverage is split across different paths

The production ALER search and repair do use the real adapter audit and assert measured counts before the official forward/backward/optimizer step ([41_train_drive_cl_diffusiondrive.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:959)). The hot query audit records only shape/dtype/device metadata and does not copy CUDA timestep values to CPU ([drive_opd.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/drive_opd.py:495)).

However, the ALER-specific test checks query counts and a gradient ([test_drive_opd_losses.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/tests/test_drive_opd_losses.py:146)), while the audit-on/off loss, gradient, parameter and AdamW-state equivalence test exercises ordinary `distillation_loss`, not ALER search plus repair ([test_drive_opd_losses.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/tests/test_drive_opd_losses.py:190)). Add one exact ALER audit-on/off optimizer-state equivalence test.

### P1 — Statistical effect-size justification remains scientifically incomplete

The same raw `minimum_relevant_effect` is added to every metric during power simulation ([seed_design.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/seed_design.py:285)). The response correctly concedes that literature does not justify the selected effect or prove nine seeds are powered ([P0 response](/home/khwang/.codex/worktrees/6189/domain-selector/paper_iterations/P001_scheduler_session_contract/P0_ACCEPTANCE_REPAIR_RESPONSE.md:231)). A future design needs comparison-specific or explicitly standardized minimum effects derived from frozen audit evidence.

### P2 — Literature and novelty positioning need an ICLR 2027 refresh

The scheduler-consistent semi-gradient and evidence protocol are careful, but the novelty is currently procedural/mechanistic rather than empirically demonstrated. The paper correctly cites DiffusionDrive, student-distribution policy distillation and crossed resampling; nevertheless, an ICLR 2027 positioning should discuss newer diffusion-driving work such as [Hyper Diffusion Planner](https://arxiv.org/abs/2602.22801) and recent autonomous-driving continual-learning work such as [H2C](https://arxiv.org/abs/2508.01158). The pigeonhole-bootstrap citation supports crossed resampling, but not this candidate’s full-family power calculation ([Owen](https://arxiv.org/abs/0712.1111)).

## Contract-by-contract audit

| Requirement | Result |
|---|---|
| 1. Raw commit chronology, full linear ancestry, append-only ledger, strict evaluator ordering | **PASS.** Raw-object parsing and strict four-time ordering are at [claim_protocol.py:69](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:69); full ancestry/ledger replay at [134](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:134). |
| 2. Actual delivery/presentation counters; A-GEM before replay forward; deterministic partial behavior | **PASS.** Tokens are delivered by `TokenDataset`; observation precedes forward, and A-GEM delivery is checked before replay forward ([training script](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:929), [A-GEM](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:1069)). |
| 3. Executable seed design, exact source/input/invocation/output, critical value/power/FWER | **FAIL.** Exact replay itself works ([claim_protocol.py](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:418)); its statistical family is incomplete. |
| 4. Real ALER query path and paired equivalence | **PASS with P1 test gap.** Counts are measured, not used as observed hardcoded counts. |
| 5. Shared evaluation cell, common baseline, crossed-cell multiplicity | **PARTIAL FAIL.** Family enumeration and final Holm are complete; seed calibration omits the comparison dimension. |
| 6. Genuinely measured transient storage | **FAIL.** Producer measures it; claim validator accepts boolean/fractional impostors. |
| 7. Post-write immutability | **PASS for sequential threat model.** Independent checkpoint mutation, symlink, stale-descriptor replacement and alias attacks all failed before global output. |
| 8. Repository/evidence-set sealing | **PASS.** Recursive discovery, exact path sets, byte/raw-evidence dedupe and rehashing are enforced ([inventory script](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/56_build_drive_cl_evidence_inventory.py:59), [Holm rediscovery](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/scripts/50_apply_drive_cl_global_holm.py:517)). Ignored/untracked evidence also failed in my independent test. |
| 9. Family/ledger/final-test/query/seed/statistical lineage | **PASS structurally, FAIL semantically where noted.** Training → checkpoint → evaluator argv → CSV binding is explicit at [claim_protocol.py:1589](/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:1589). |
| 10. Method, math, experiment, literature, novelty and claim boundary | **Boundaries PASS; paper readiness FAIL.** The paper explicitly makes no performance claim ([CLAIMS_EVIDENCE.md](/home/khwang/.codex/worktrees/6189/domain-selector/paper_iterations/P001_scheduler_session_contract/CLAIMS_EVIDENCE.md:9)). |

## CPU/source reproduction

Common environment:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
PYTHONPATH=/home/khwang/.codex/worktrees/6189/domain-selector/selector_bench \
/home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10
```

Interpreter: Python 3.10.20.

Each suite used:

```bash
<common-environment> -m pytest -q -p no:cacheprovider <suite>
```

| Suite | Result | stdout SHA256 |
|---|---:|---|
| `test_navsim_continual_protocol.py` | 3 passed | `c2cba0e5b1275bde99432021fd22a26635493fa35d0886fcd5cd38708b0f7cca` |
| `test_drive_opd_losses.py` | 13 passed | `966c787c5b84cff1f2940469590ae6aaeac6fc4b1240c747205f8f8b0dff3d82` |
| `test_drive_cl_baselines.py` | 8 passed | `98e8124148330ddb6708ee670f805e418d50c4d4e70c5753d2aed5547314b2a5` |
| `test_drive_cl_statistics.py` | 8 passed | `2258da9311f6395f328963456cede3d56523ef15c0585df78fd82e95fa259756` |
| `test_drive_cl_claim_protocol.py` | 28 passed, 73 subtests | `4ea0ef379520d5bbca247866508f1a9ac85e5517e5af3ab2cc5a34b929eceaec` |
| `test_drive_cl_run_budget.py` | 7 passed | `8f72f6ef79c4ba95867373899f0d4b43c07b227d042cfa7261eac1f688b3f613` |

Total: **67 passed plus 73 subtests**. Every suite’s stderr was empty, SHA256 `e3b0c442…b855`.

Attack command:

```bash
<common-environment> selector_bench/scripts/57_audit_drive_cl_p0_attacks.py \
  --repo /home/khwang/.codex/worktrees/6189/domain-selector \
  --python /home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10 \
  --output /tmp/drive-opd-rc3-review.IvvV5V/p0_attack_audit.json
```

Result: harness subprocess exit 0; **19/19 families, 87/87 mutations nonzero, 87/87 claim outputs absent, positive fixture exit 0**.

- Standalone stdout SHA256: `ec6b73c2329ffdbd471034e393882a8a3c016983a52ffd2703380d10b54d0b51`
- Immediate standalone receipt SHA256: `1f7b4b7d9a8a4bad15ed7e59d8c74f1fa31b99905129800e1c4d30d767a34e1e`
- A later claim-audit rerun regenerated the path-sensitive receipt as `82ea25dcf72f628f353435a2b3ed06ce20acb2a31136d56296c53c7363722a5b`.

Higher-level audit command:

```bash
<common-environment> selector_bench/scripts/53_audit_drive_cl_claim_protocol.py \
  --repo /home/khwang/.codex/worktrees/6189/domain-selector \
  --python /home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10 \
  --output /tmp/drive-opd-rc3-review.IvvV5V/claim_protocol_audit.json
```

It exited 0 after 517.505 seconds:

- Combined test output: 67 plus 73, SHA256 `6d55bfbd8f88cd94b0342a4c871ea54b675b182504e9893603eafff7de21608d`
- Attack output SHA256: `9d6245bf1d296bbc5e28b9e4e4fbfd33a7fabf902d399d87e4abfc521fd96556`
- Whole audit stdout SHA256: `14ed9a94735321630e8431be433659500cbe416475213eb665699b2373764e68`
- Audit JSON SHA256: `31f3ef590ae064f9ac7b89ae44a601450e0329bbc0d5af94a9158f0376b0b28f`

Independent static checks exited 0:

- 2 changed JSON files parsed
- 22 changed Python files AST-compiled
- 19 JSON-emitting scripts inspected; zero missing `allow_nan=False`
- `git diff --check` and `git show --check` passed
- stdout SHA256: `cc077b0547bb817fe815d55307192a3f463a8db677b5c70f19385242f011aaf6`

Independent post-write/sealing attacks exited 0 as an audit driver: checkpoint mutation, ignored untracked evidence, symlink alias, hard-link alias and stale-descriptor replacement all caused the claim CLI to exit 1 without output. Malformed raw commit timezones/headers were rejected. Output SHA256: `82e7ae9a597dedc9b7f3ac89e2f017c92924a07e41a283de048c522819a9d13e`.

## Method and empirical assessment

The registered `(10,0)` scheduler construction and student-support semi-gradient are clear and appropriately bounded ([METHOD_AND_THEORY.md](/home/khwang/.codex/worktrees/6189/domain-selector/paper_iterations/P001_scheduler_session_contract/METHOD_AND_THEORY.md:3)). The distinction between student-rollout OPD support and exogenous LwF support is technically meaningful. The A-GEM buffer-preservation intervention and per-example EWC proxy are also carefully specified.

These are source-level mechanisms, not empirical evidence. No official cache, driving batch, real model, checkpoint, T0, GPU training, NAVSIM evaluation, nuScenes evaluation, safety result or superiority result was reproduced or authorized. Consequently, efficacy, external validity, stability/plasticity and novelty relative to modern diffusion planners remain unestablished.

## Required actions and remaining blockers

1. Bind seed design to the exact frozen comparison/hypothesis family: family SHA, ordered hypothesis IDs, comparison IDs, candidate/baseline identities and total hypothesis count.
2. Supply pilot matrices per comparison or a preregistered joint crossed array; calibrate power/FWER against the same 14-hypothesis procedure used downstream.
3. Correct or remove the \(2^{1-n}\) tail-resolution claim so it matches the actual bootstrap p-value procedure.
4. Replace resource coercions with strict finite-real and strict non-boolean integer validation. Require non-empty host/device identities.
5. Add end-to-end confirmatory attacks for boolean, fractional, string and non-finite resource fields.
6. Add exact ALER search-plus-repair audit-on/off loss, gradient, parameter and AdamW-state equivalence.
7. Refresh ICLR 2027 related work and keep all efficacy language explicitly untested.
8. Produce a new forward-only immutable candidate, rerun the safe CPU suite and adversarial harness, obtain another fresh review, and only then re-freeze/publicize the bundle.
9. Even after a future P0 pass, dataset provenance, adapter, cache, real-model, T0, GPU, training, evaluation and performance claims require their own dataset-specific gates.

**Final disposition: RC3 source identity and transfer integrity pass; RC3 claim-protocol acceptance fails. All execution gates remain closed.**

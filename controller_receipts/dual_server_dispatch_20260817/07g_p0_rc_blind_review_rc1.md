# ICLR 2027 adversarial review

## Verdict

**Score: 4/10 — Reject / P0 remains open**  
**Confidence: 5/5**

The candidate fixes all 12 previously enumerated attacks, but it does **not** pass the complete frozen acceptance contract. I found three remaining claim-integrity failures:

1. sealed-test chronology is not actually enforced;
2. the production runner reports some “observed” budget fields by copying their targets;
3. seed power/calibration receipts can be handwritten without replaying the executable design.

No cache, model loading, T0, GPU use, training, or performance evaluation is authorized.

## Reviewed object and reproducibility

Exact candidate verified:

- Ref: `codex/iclr2027-drive-opd-07g-p0-rc`
- SHA: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`
- Parent: `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`
- Tree: `0e6e9a0adda280765ce41ab377de9bbbaea0cff1`
- Changed paths: exactly 34, including 25 `selector_bench` implementation/config/test paths
- Worktree: clean before and after
- Checkout: detached, but the named local branch points to the exact SHA
- Changed-path content-hash manifest SHA-256: `b07949de23b0361c8a362d000d0d0cbe824f9649689e865f855a723f078cb44d`
- `git diff --check HEAD^ HEAD`: pass
- Temporary P0 v1/v2 diagnostic directories: absent
- No `__pycache__` or `.pytest_cache` created

Frozen contract:

```bash
git show 37448d7bac7721857148d49d6899d515c5dd0438:ICLR2027_P0_CLAIM_PROTOCOL_ACCEPTANCE_20260817.md | sha256sum
```

Result:

```text
374d81fd57e4bd40bed396b4af26f86f1fa52f8f38e712de2b880fe45d295b7e
```

### Five frozen suites

```bash
CUDA_VISIBLE_DEVICES="" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/selector_bench" \
/home/khwang/domain-selector/.conda/diffusiondrive_py310/bin/python3.10 \
-m pytest -q -p no:cacheprovider \
  selector_bench/tests/test_navsim_continual_protocol.py \
  selector_bench/tests/test_drive_opd_losses.py \
  selector_bench/tests/test_drive_cl_baselines.py \
  selector_bench/tests/test_drive_cl_statistics.py \
  selector_bench/tests/test_drive_cl_claim_protocol.py
```

Result:

```text
49 passed, 9 subtests passed in 96.38s
```

Terminal-output SHA-256:

```text
157694437a1e90a5edc3d07432796019f753487e3dc307704f5ae684fe943f4e
```

### Attack harness

Executed with CUDA hidden, bytecode disabled, pytest cache disabled, and receipt/output placed outside the worktree.

Result:

- Harness exit: `0`
- Status: `PASS`
- Families: 12/12
- Mutation instances: 19/19 rejected before claim output
- Positive filename/order-invariant fixture: pass
- Harness stdout SHA-256: `03b91fa3109dc391a26eb753bdeaeb1df857d42ffb8120107fb5d9c1330bc0d6`
- Generated receipt SHA-256: `8a36ef29ba2c7d2a5acafc23a800e86dabbe734828e2bde073db12130bee6dc8`
- Test-source SHA-256: `5cf4f6747755f3df5364a4aa6acfac7cd301b031cb38825e7bc524c9f456a5de`

| Attack | Independent result |
|---|---|
| P0-01 seed reuse | Rejected before claim |
| P0-02 CSV swap | Rejected before claim |
| P0-03 arm mismatch | Rejected before claim |
| P0-04 unknown domain | Rejected before claim |
| P0-05 suffix hide | Rejected before inventory |
| P0-06 byte-identical receipt | Rejected |
| P0-07 duplicate complete raw bundle | Rejected |
| P0-08 missing cross cell | Rejected |
| P0-09 duplicate/extra/missing/invalid row | All four rejected |
| P0-10 mutable config/protocol/registries/registration | All five rejected |
| P0-11 serialized incomplete budget | Rejected |
| P0-12 hidden extra claim object | Rejected |

## P0 findings

### P0 — Final-test chronology is declarative and bypassable

[54_run_drive_cl_navsim_evaluator.py](/home/khwang/.codex/worktrees/f14a/domain-selector/selector_bench/scripts/54_run_drive_cl_navsim_evaluator.py:69) verifies commit ancestry and merely checks that `authorized_at_utc` ends in `+00:00`. It never compares:

- authorization time with the family-freeze commit time;
- authorization time with the access-receipt commit time;
- evaluator start time with authorization time;
- the asserted prior-access count with an authoritative append-only ledger.

I mutated the repository’s own sealed-test fixture so that authorization was declared at `1900-01-01T00:00:00+00:00`, committed it after the family, and ran four evaluator fixtures. All four were accepted:

```json
{
  "result": "ACCEPTED",
  "declared_authorized_at_utc": "1900-01-01T00:00:00+00:00",
  "evaluators_completed": 4
}
```

Output SHA-256:

```text
c0b466b84814afd79baa425c5761c341ff5bd73bbdb62c8b58f7bd0f07925e7b
```

This violates the frozen requirement for executable final-test chronology.

### P0 — Production “observed budget” is partly copied, not observed

The runner derives the target budget at [41_train_drive_cl_diffusiondrive.py:684](/home/khwang/.codex/worktrees/f14a/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:684). When producing the completion receipt, it initializes `observed_budget` with `**target_budget` at [line 1127](/home/khwang/.codex/worktrees/f14a/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:1127).

Consequently:

- `current_unique_identities` and `old_unique_identities` are copied unchanged;
- current/old presentations are set to their target values whenever step count completes;
- only optimizer steps, forward/backward calls and some query counts are genuinely accumulated.

The P0-11 fixture proves that a manually lowered serialized counter is rejected. It does not prove that the production runner measures those presentations. A loader or replay-accounting discrepancy can therefore be certified by construction as complete.

### P0 — Seed-design mathematics is not replayed at the claim gate

[validate_seed_design](/home/khwang/.codex/worktrees/f14a/domain-selector/selector_bench/selector_bench/continual/claim_protocol.py:223) checks that receipt fields are mutually consistent but does not:

- bind the receipt to script 55’s source hash or exact invocation;
- rerun the simulation from the pilot matrix;
- recompute critical values, power, or held-out FWER.

The positive fixture directly handwrites `estimated_power=0.85`, metric powers, simulation counts, and calibration status at [test_drive_cl_claim_protocol.py:577](/home/khwang/.codex/worktrees/f14a/domain-selector/selector_bench/tests/test_drive_cl_claim_protocol.py:577). It never invokes `55_design_drive_cl_confirmatory_seeds.py`, yet the full confirmatory chain and Holm decision pass.

Thus the claimed executable power/calibration receipt can be fabricated while remaining schema-valid.

## P1 findings

- ALER query counts are hardcoded as two student/two teacher calls and bypass `capture_query_audit`; only the other distillation arms perform real-path assertion before the optimizer step. See [41_train_drive_cl_diffusiondrive.py:925](/home/khwang/.codex/worktrees/f14a/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:925).
- `validate_family_registries_and_cells` rejects two comparisons sharing a checkpoint-stage/evaluation-domain cell. That makes ordinary multi-method-versus-baseline families over one common test cell difficult or impossible without artificial cell duplication.
- Storage accounting records `transient_bytes: 0` rather than measuring transient storage.
- The chronology fixture and seed-design positive fixture should be adversarial negative tests, not accepted positive paths.

## Passing components

The following repairs are substantively sound at source/CPU-fixture level:

- distinct seed/run/RNG/config/protocol/result identities;
- frozen run-registration consumption and v3 production/validator schema parity;
- checkpoint → evaluator invocation → CSV lineage;
- one-to-one method/arm registry;
- protocol-derived dataset/version-qualified domains;
- frozen metric registry;
- suffix-independent recursive claim discovery;
- exact expected receipt inventory;
- byte-identical receipt and complete raw-bundle duplicate rejection;
- row/token/metric exactness;
- immutable family and evidence-inventory ancestry;
- filename and hypothesis-order-invariant positive result;
- CUDA hot-query counter records shape/device/dtype only and does not copy timestep values to CPU;
- CPU synthetic audit-on/off equivalence for loss, gradient, AdamW parameters and optimizer state;
- CPU LwF/OPD identical-state and equal-support controls.

These do not replace an official-model, real-batch adapter gate.

## Evidence and publication boundary

The checked-in precursor receipts are internally hash-consistent:

- `claim_protocol_audit.json`: `7aaa90d901872bacef77d35a0a3eed9b8ac85fda7fc115d9a3061193cb3920c9`
- `p0_attack_audit.json`: `bd414ac17797506ce1cbdcbc8c9f6f1dc24c426847b531cf85d07726a8f3ea1d`
- `ROUND_MANIFEST.json`: `9de6d3da2270cc1f7dbb371994c5e57a610486f89df5293b10ba84f4708f863c`

They describe precursor `7b3b1e0d…`, not independent evidence for the exact RC.

I could not independently verify the controller’s abbreviated `02e2784c…` source publication or `d1665712…` replay receipt. Neither commit was present in local history, and `git ls-remote` exposed neither requested publication nor RC branch on the configured origin. Therefore the reported 37/37 bytes, 49 tests, 12/12 families and 19/19 mutations remain external claims.

## Explicit decisions

**A) Does the P0 repair pass the frozen acceptance contract? — NO.**

The named P0-01…P0-12 suite passes, but final-test chronology, production observed-budget measurement, and executable seed-design verification remain open P0 failures.

**B) Is source-only publication/replay safe? — Conditional YES.**

It is safe only as clearly labeled, non-promoting source material for independent audit/replay. It is not safe to publish as a passed claim protocol, and the reported external transfer replay was not independently verified here.

**C) Does this authorize cache/model/T0/GPU/training/evaluation? — NO.**

No dataset-specific provenance, cache, real-adapter, model, or GPU gate was reviewed or passed.

## Remaining blockers

For the paper:

- repair and adversarially test authoritative final-test chronology;
- make the claim gate replay or cryptographically bind the seed-design computation;
- distinguish genuinely observed production counters from derived targets;
- obtain a new context-free review of the repaired exact SHA;
- independently verify full publication and destination replay objects;
- retain NAVSIM and nuScenes as separate statistical and promotion families.

For execution:

- all dataset-specific provenance and scene/session split gates;
- cache-builder replay;
- real official-model one-batch loss/query/gradient/resume gate;
- immediate GPU ownership/UUID/utilization audit;
- separate explicit controller authorization.

Until those are satisfied, cache, T0, GPU, training, and evaluation remain **STOPPED**.

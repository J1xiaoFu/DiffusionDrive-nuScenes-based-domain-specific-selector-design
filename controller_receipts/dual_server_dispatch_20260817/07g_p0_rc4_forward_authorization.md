# 07G Drive-OPD P0 RC4 Forward-Only Authorization

Date: 2026-08-18

## Decision

**AUTHORIZED only for one forward-only 07G source/CPU repair candidate.**

This authorization does not promote RC3, publish RC4, create or accept a review, dispatch 06G
work, read a dataset or cache, import a real model, unseal final-test data, use CUDA/GPU, train,
evaluate, or support a performance claim.

## Immutable base and destination

- rejected base branch: `codex/iclr2027-drive-opd-07g-p0-rc3`
- exact base commit: `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`
- base parent: `d527274d368b6d6def5809b002324b31e88ae1bd`
- base tree: `3ec47127a6d7c85326bbcf1d72ef76e2971f100d`
- authorized new branch: `codex/iclr2027-drive-opd-07g-p0-rc4`
- suggested isolated worktree: `/home/khwang/domain-selector-p0-rc4`
- required history: exactly one new non-merge commit whose sole parent is the exact RC3 commit

RC3, RC2, RC1, research, recovery, controller and publication refs must not move. No reset,
rebase, amend, merge, history rewrite, force update or deletion is authorized.

## P0-A: full-family seed-design alignment

The executable seed-design receipt and validator must bind and replay the actual frozen
confirmatory family, not a seven-metric anonymous proxy:

- bind the canonical family path/hash, ordered comparison IDs, ordered hypothesis IDs, method and
  common-baseline identities, metric IDs and total family size; the current family has two
  comparisons and fourteen hypotheses;
- require a pilot matrix covering every comparison-by-metric cell, or one explicitly preregistered
  joint crossed array that identifies the same complete family;
- replay a deterministic power/FWER calculation using the same fourteen-hypothesis downstream
  familywise inference procedure, or a formally exact preregistered surrogate whose equivalence is
  checked in source and tests;
- remove or correct the `2^(1-n)` sign-resolution seed rule. Any resolution bound must be derived
  from the downstream bootstrap p-value floor `1/(R+1)` and the first applicable Holm threshold;
- bind the pilot bytes, program source hash, invocation, simulation seed, repetition count, effect
  specification, critical values, per-hypothesis power, family power/FWER and normalized output;
- reject non-finite or structurally incomplete inputs and outputs.

Required adversarial cases include reuse of a one-comparison pilot, an omitted comparison or
hypothesis, family path/hash/order/size mutation, comparison/candidate/baseline/metric mutation,
procedure or p-value-floor mismatch, and changed bootstrap repetitions, simulation seed, effect,
critical value, power or FWER.

## P0-B: strict resource-evidence types and finite values

Resource evidence must be validated without coercion at production, load, evidence inventory,
comparison and global-Holm boundaries:

- `transient_bytes` and `persistent_bytes` must be exact positive integers; Boolean, floating,
  string, zero and negative values are invalid;
- `wall_seconds` must be a finite positive real number that is not Boolean or a string; NaN and
  positive/negative infinity are invalid;
- required host and device identity/type fields must be non-empty strings with the registered
  semantics;
- malformed pre-existing evidence must fail before any comparison, claim or global-result output
  is written.

Required end-to-end adversarial cases include `true`, `false`, zero, negative, fractional and string
byte counts; Boolean, string, zero, negative, NaN and positive/negative-infinite wall time; blank
host/device identity; and altered host/device types. Each mutation must exit nonzero with no claim,
comparison or global-Holm artifact.

## P1 non-regression and paper boundary

- Add exact ALER audit-on versus audit-off equality coverage for loss, gradients, parameters and
  AdamW optimizer state around search and repair; keep measured query accounting.
- State the statistical effect-size choice and its evidence limitations explicitly. Refresh the
  relevant autonomous-driving literature, including Hyper Diffusion Planner and H2C, without
  claiming unmeasured efficacy.
- Preserve all previously closed chronology, active-family, shared-cell, observed-budget,
  fail-closed JSON and output-sealing invariants and all nineteen RC3 attack families.

## Authorized verification and candidate receipt

07G may edit source, config, tests and the bounded paper response; run static checks, synthetic
temporary-Git fixtures and CUDA-hidden CPU tests; and create exactly one auditable RC4 commit. The
candidate receipt must include exact branch/commit/parent/tree, clean status, exact changed-path and
per-path SHA256 inventories, suite and attack outputs/hashes, and an artifact-boundary attestation.

Acceptance requires all six existing CPU suites plus new targeted tests, an expanded named attack
harness covering both P0s, and a positive complete-family/strict-resource fixture. Python bytecode,
pytest cache, data, cache, checkpoint and model artifacts must not remain in the candidate tree.

## Controller-held gates after candidate construction

RC4 cannot self-promote. Publication, source-transfer freeze, a new context-free exact-RC4 review,
and every 06G action remain controller-owned. A future review must use a newly created task whose
actual metadata is `gpt-5.6-sol` with `xhigh` reasoning and must independently attack both P0 areas.

Before **any 06G source replay**, both of these conditions must be independently satisfied:

1. the fresh exact-RC4 xhigh review has no open P0; and
2. the controller has frozen dataset-specific, session-atomic CL manifests for both NAVSIM and
   nuScenes, with distinct provenance, population, partition, stage, session and hash identities.

Neither dataset lane may substitute for the other. Even those two conditions do not authorize cache
construction, a real-model preflight, T0, GPU, training, evaluation or a performance claim; each
still requires its later explicit controller gate and the applicable dataset/adapter/cache-builder,
disk and hardware receipts.

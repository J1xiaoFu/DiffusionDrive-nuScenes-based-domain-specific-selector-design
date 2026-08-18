# NAVSIM session-manifest source inventory and legacy-source decision

Date: 2026-08-18

## Decision

**The existing NAVSIM continual-protocol implementation is not eligible for a controller
approved-producer freeze. A separate forward-only dataset-specific producer candidate is required.**

This is a read-only source and accepted-evidence audit. It does not authorize dataset access,
membership generation, source replay on 06G, cache construction, model import, T0, CUDA/GPU,
training, evaluation, final-test access or a performance claim.

## Accepted boundary available to a future producer

The controller has already accepted only the official NAVSIM navtrain population and its official
train/validation partition:

- population commit `1bc9cc2957eeccc6927f3dc826135ce85bfae790`
- evidence commit `af9213b216b0260b184d3d3ae109a4ed1bed70b8`
- population receipt SHA256 `fa2f8c417957c22d3fce83d36544c373353a533302deeb5b9a618d9db4aedb1a`
- session-integrity receipt SHA256 `7156370a742df6328b03350f48c7b55a1608e9d0498dbd131f17ac075a2e63bd`
- canonical 162-row session-table SHA256
  `cf12df3e215874362c67a6ce404a8bb8ae054fdc5c393f7a58385abd2453fa08`
- full: 162 sessions, 1,192 segmented logs, 103,288 tokens
- official development: 101 sessions, 978 segmented logs, 85,109 tokens
- official sealed validation: 61 sessions, 214 segmented logs, 18,179 tokens

These identities do not contain or authorize a Chronological-CL or Failure-Patch-CL assignment.

## Why the existing source is rejected

The exact RC4 implementation hashes are:

- `selector_bench/selector_bench/continual/navsim_protocol.py` —
  `568ffc9f57f81f25241f6bdc12db584f8c3f3cf1038c6fdde92f701f4cf234ce`
- `selector_bench/scripts/40_build_drive_cl_protocol.py` —
  `e7ceb775f8341dcdee4349cdb38d2582b3b99d978f72951138ff1830d1a59079`
- `selector_bench/tests/test_navsim_continual_protocol.py` —
  `37b072a66913416ea65948d334682664f958a1c5b8eaf5690301317f62834257`

That surface is a legacy prototype. It loads population membership from a pickle cache index, the
CLI defaults to a historical `valid_cache_train.pkl`, and Failure-Patch scores come from a generic
metrics CSV. It does not require the exact accepted 162-row session ledger and receipts, does not
authenticate a controller-authorized pre-access failure stream, and does not emit or validate the
common controller-frozen population, selection, assignment and manifest artifacts. A cache-derived
population or an internally consistent metrics CSV is forbidden evidence under the contract.

## Required R1 ledger conversion guard

Each accepted R1 session row contains `session_key`, split, log IDs and token IDs plus chronology
counts and minimum/maximum chronology keys. It does not contain an independent `scene_token` field.
A future producer must therefore:

1. bind each output population row to the exact canonical source row and recheck every count, list,
   split and chronology invariant;
2. use `session_key` as the atomic unit and preserve official train as development and official
   validation as sealed evaluation;
3. use the exact sorted log and token sets as descendants;
4. derive strict UTC order deterministically from the validated minimum chronology key; and
5. either prove, from an accepted dataset-specific source identity, that NAVSIM `scene_token` is the
   same segmented-log identity or consume an independent exact scene-token mapping. Merely copying
   `log_ids` into a `scene_token` field is not acceptable.

## Remaining boundary

The controller may authorize a synthetic source/CPU candidate, but cannot approve it in advance.
Approved-producer identity, exact Failure-Patch evidence, real manifest generation, independent
replay and every 06G or execution gate remain stopped.

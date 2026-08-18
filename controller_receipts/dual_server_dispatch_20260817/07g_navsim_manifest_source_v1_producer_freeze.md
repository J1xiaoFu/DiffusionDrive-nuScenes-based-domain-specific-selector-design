# NAVSIM session-manifest producer v1 identity freeze

Date: 2026-08-18

## Decision

**PRODUCER IDENTITY FROZEN; REAL MANIFEST STILL BLOCKED.**

The controller reviewed audit at commit `00d3853e11105e0b5c4653e62482ae5c29f3c2c6` accepts the exact
source candidate `64447d3a9cc60f61b8596aed41086b5b4311c889`, tree
`5ce81ef06d71189d1ff46676ecc1b4d01e415728`. The NAVSIM requirements now bind:

- producer: `selector_bench/selector_bench/continual/navsim_protocol.py`, SHA256
  `75f683cb8a7931783f1342a9b479d37470bc215b297efc421b003e3a2ab6cfae`
- frozen semantic validator:
  `controller_protocols/drive_opd_session_atomic_v1/validate_session_atomic_manifest.py`, SHA256
  `70b53893c3bc0704fbf09945a677412effdf4a2b0258ac56c2f8084798805ba3`
- updated requirements SHA256:
  `54d3f5f203a3db568c5cc5d18305c3dc4b00b81863e353c9dbd30bdefc833d64`

The controller-validator suite passes 19/19 after this update; stdout SHA256 is
`b84a5f56832006ce4e2bf9f759666949900308668a781b362e5aab41c6c2b8e8`.

This identity freeze is not a manifest freeze. The requirements still lack a controller-authorized
scene-identity proof or exact mapping and retain blocked Failure-Patch evidence. Real R1 objects have
not been transferred to the producer lane, and no real membership artifacts exist.

The exact-RC4 xhigh review still waits for user approval and has no valid zero-P0 verdict. Therefore
real manifest generation, publication/transfer, 06G source replay, cache, model, T0, CUDA/GPU,
training, evaluation, final-test access and performance claims remain stopped.

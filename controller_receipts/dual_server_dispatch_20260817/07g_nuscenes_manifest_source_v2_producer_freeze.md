# nuScenes manifest producer-v2 identity freeze

Date: 2026-08-18

The controller freezes only this reviewed source identity:

- source commit `495bac670073fc3f46f0f773ca51033331078509`
- source tree `a7c86fbafea77ec647197c6d24d65edff83c95a8`
- producer `selector_bench/selector_bench/continual/nuscenes_protocol.py`, SHA256
  `a5e509d9fb33e5bbc00bdea71ecc51521eeeee39335f18e8667bea124ac2d773`
- semantic validator `controller_protocols/drive_opd_session_atomic_v1/validate_session_atomic_manifest.py`,
  SHA256 `70b53893c3bc0704fbf09945a677412effdf4a2b0258ac56c2f8084798805ba3`

The source-v2 audit is controller commit `145d6504ed09457e0cd094fad41341bd4a07306c`; its
JSON receipt SHA256 is `f3b9f870ad6d82a31644503508c0c0c92e5422eb563f4802840daa3aa3b891ea`.
The updated nuScenes requirements SHA256 is
`08968b1af352210fb57bfdc7db0946f4f75191ba865b2889e476fb5fc8409507`.

This freeze only removes the producer-identity blocker. The same requirements still reject every
production attempt because no population receipt is accepted, official counts/hashes are absent,
and Failure-Patch parameters/evidence remain blocked. A separate controller authorization is
required even for metadata-only N0 intake; real manifest generation requires additional reviewed
evidence and a later explicit gate.

RC4 blind review, both complete dataset manifests, source publication/destination replay and the
distinct 06G source-replay gate remain pending. Raw-data access, cache, model, T0, CUDA/GPU,
training, evaluation, final-test access and performance claims remain stopped.

# 07G nuScenes session-manifest inventory intake

Recorded: 2026-08-18

## Decision

**Accepted only as a read-only gap inventory. No dataset evidence, producer approval, manifest
generation, source replay, or execution gate is promoted.**

The controller does not narrow the claim to scene atomicity. Its frozen contract already defines the
nuScenes atomic unit as one complete `log_token`: every child scene, sample, and selected sample-data
record inherits one official partition, protocol stage, and development cell. The historical
whole-scene helpers are useful implementation substrate, but a tool that selects scenes independently
cannot satisfy this contract because multiple scenes may share one capture log.

## Intake identity and independent checks

- source task: `01a00dab-5dc6-7413-8ca7-930353db0e0c`
- exact source inspected by the sender: RC4
  `653e797a2381cdf017466ef95aefcc41961f6715`
- exact delegated inventory text: 10,562 UTF-8 characters, 103 lines, SHA256
  `de6204c1b729872b006b7796aac27cf6bf820a4beae73970e0678574989ab385`
- controller check: all 18 advertised RC4 file hashes independently matched `git show` bytes from
  the exact commit
- frozen controller contract SHA256:
  `04d3109b3f7f33290d563699f0980550ddc1a35e1479ab1049b9d70234b9e0e0`
- frozen nuScenes requirements SHA256:
  `fbbf99edce7362fd0537ca694f573e0ed2372927f992da0ad29ac1df11100440`

## What exists

- presence-only nuScenes counts and limited metadata hashes, explicitly not an N0 pass;
- scene-level metadata, balancing, archive, sample-subset and custom-scene converter prototypes;
- the DiffusionDrive nuScenes converter, dataset loader and stage-2 configuration surface;
- a NAVSIM-specific session-atomic protocol, strict claim scaffold and training-manifest exporter that
  provide reusable design patterns but cannot authorize nuScenes;
- a historical mini-feasibility report that remains diagnostic and unbound to current data.

## What remains missing

The blocked surface is dataset identity and closure, not merely another split file. Before a complete
nuScenes manifest can be accepted, the controller still needs:

1. an all-table release/devkit/root provenance receipt and canonical metadata digest;
2. all-table foreign-key closure plus map and CAN linkage;
3. a registered six-camera, LIDAR_TOP and sweep inventory with file-integrity checks;
4. a canonical complete-log population ledger and official-split containment proof;
5. a controller-reviewed complete-log producer with strict fallback, overlap and closure rejection;
6. exact Chronological-CL and authorized pre-access Failure-Patch-CL inputs and assignments;
7. complete stage-by-cell membership, descendant closure, coverage and disjointness receipts;
8. manifest-to-info-PKL-to-config lineage and train-only derived-artifact lineage;
9. nuScenes-specific evaluator, metric/domain registries, comparison family and seed design;
10. adversarial cross-log, split-fallback, materialization-swap, modality-omission and anchor-leakage
    tests, followed by an independent destination replay.

## Program evidence readiness

| Area | Status | Current boundary |
|---|---|---|
| Problem/protocol freeze | Partial | Semantic contract is frozen; both dataset memberships remain incomplete. |
| Method math/causal estimand | Partial | Semi-gradient and implementation estimands are explicit; no causal performance or convergence claim exists. |
| Core baselines | Partial | Roster and code semantics exist; no official fair claim-bearing runs exist. |
| Query/compute/storage fairness | Partial | Source/CPU contracts exist; real-adapter and real-model measurements remain blocked. |
| Ablations | Missing | Prospective arms are named; no claim-bearing outputs exist. |
| Cross-architecture nuScenes | Blocked | Prototype source exists; session membership, materialization, evaluator and result lineage do not. |
| Statistics/seed design | Partial | Family and seed-design source exists; exact RC4 review and nuScenes freezes remain pending. |
| Paper claim boundary | Partial | P001 disclaims performance claims, but its round disposition predates RC4 and current dataset gates. |
| Fresh blind/human audit | Blocked | Exact-RC4 xhigh task has no valid verdict and is waiting on approval. |
| Publication/replay | Partial | RC4 source identity is frozen; this is not execution authorization. |

## Gate state

Exact-RC4 zero-P0 xhigh review, complete NAVSIM manifest, and complete nuScenes manifest are all
unresolved. 06G source replay and every dataset/raw-data, cache, real-model, T0, CUDA/GPU, training,
evaluation, final-test and performance-claim gate remain stopped.

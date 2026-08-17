# P001 Claims-to-Evidence Matrix

| ID | proposed paper statement | evidence | status | boundary / next falsifier |
|---|---|---|---|---|
| C1 | Drive-OPD and LwF use a registered constant-stride `(10,0)` timetable with equal denoiser-query counts. | Actual Diffusers numerical audit; scheduler unit tests; identical-state loss/gradient negative control. | Supported as implementation claim. | Does not imply PDMS benefit; P002 must compare endpoints. |
| C2 | Historical modified-RAP two-step inference is not a controlled OPD/LwF support comparison. | Source hash plus observed `noise=8`, query `10`, scheduler `10→9`, next label `0`; RMS 0.0198078 to declared state. | Supported for the frozen source. | Must not generalize to pristine upstream without source audit. |
| C3 | EWC uses a per-example official-loss gradient second moment, not squared aggregate batch gradients. | Enforced batch-one accumulator, artifact schema v2, regression tests. | Supported as code semantics. | It is not claimed to be the true Fisher. |
| C4 | A-GEM replay cannot mutate running buffers beyond the current-data forward. | Full buffer snapshot/restore and BatchNorm gradient regression test. | Supported as code semantics. | Real model cost and performance remain unmeasured. |
| C5 | Main uncertainty is clustered by complete timestamp–vehicle sessions and invalid rows fail closed. | Session parser, paired/hierarchical bootstrap tests, strict CSV tests, Holm implementation. | Supported as protocol claim. | Three-seed PDM results do not yet exist. |
| C6 | The local 14,951 cache and CL manifest are exact, reproducible derivatives of the declared NAVSIM scene rule. | 13.17 GB raw reconstruction, cache/manifest hashes, per-log/session tables. | **Refuted.** Cache/manifest match each other but not the declared rule; selection receipt is absent. | 06G official regeneration receipt is required. |
| C7 | Scheduler-consistent Drive-OPD improves continual-driving stability/plasticity. | None in P001. | Not tested. | P002 official Seed-0 stage matrix after data gate. |

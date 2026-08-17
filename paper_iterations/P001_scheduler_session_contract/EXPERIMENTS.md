# P001 Experimental Setup

## Questions and preregistered outcomes

P001 asks two paper-critical questions before any new performance comparison.

1. **Can OPD and LwF be compared on correctly registered diffusion states at equal forward cost?**
   Expected outcome: initial-noise time equals first query time; every scheduler step lands on the next
   query time; corrected numerical RMS to the declared next state is below (10^{-6}); both methods use
   two student and two teacher denoiser queries.
2. **Can the local 14,951-scene inventory be regenerated from the declared NAVSIM source rule?**
   Expected outcome before the audit: 103,288 allowlist frames reduce reproducibly to the 14,951 cache
   scenes, whose 70/10/20 session split then explains 10,444 training scenes. Failure criterion: any
   cache token not attributable to the frozen loader/config, or an undeclared post-hoc cache subset.

The second expectation was falsified. We did not redefine the expected rule after observing the data.

## Scheduler mechanism probe

The probe uses Diffusers 0.21.4, `DDIMScheduler(num_train_timesteps=1000,
beta_schedule="scaled_linear", prediction_type="sample")`, four synthetic batches of 20 eight-pose
trajectories, and a fixed seed (20260817). Historical and corrected schedules receive the same clean
states and Gaussian noise. Given an exact (x_0) prediction, the returned state is compared with a
forward-noised reference at the declared next timestep.

The frozen sources are:

- modified-RAP official heuristic source SHA256:
  `ee33c8eee9a4466743ecd9ffdb757479c1377522784c58ee626797c5d18f1d3d`;
- P001 Drive-OPD source is recorded in the generated audit JSON and will be frozen by the candidate
  commit.

## Inventory reconstruction

The audit reads all 1,192 configured raw log pickles (13,170,865,067 bytes), hashes each file, and
reimplements the frozen `SceneLoader` ordering of checks: 14-frame completeness, current-frame
validity, non-empty route, and allowlist membership. It separately computes unit-stride eligibility
and the YAML-declared interval-14 sample. It then compares exact token sets and token-to-log mappings
against:

- `/home/rguo/rap_workspace/exp/training_cache/valid_cache_train.pkl`;
- the full Failure-Patch-CL manifest;
- the portable derivative and its train-only contract.

The aggregate selected raw-tree content hash is
`503e0dd763371a8893fe20b675cfbc399d332d57560f3644e31bfa2eda5c06ca`.
The full per-log and per-session tables are retained as round artifacts.

## Statistical descriptive audit

For each of 162 complete timestamp–vehicle sessions, cache coverage is the number of cached tokens
divided by unit-stride valid allowlist tokens. The reported interval uses 10,000 complete-session
bootstrap resamples with seed 20260817. It is explicitly descriptive and is not used to infer model
quality.

## Software verification

CPU unit tests cover the scheduler contract, identical-state loss/gradient negative control,
per-example EWC enforcement, A-GEM buffer restoration, session clustering, strict invalid-row
handling, hierarchical seed/session bootstrap, and Holm adjustment. No GPU performance experiment is
part of P001; P002 remains blocked by the failed data-provenance gate.

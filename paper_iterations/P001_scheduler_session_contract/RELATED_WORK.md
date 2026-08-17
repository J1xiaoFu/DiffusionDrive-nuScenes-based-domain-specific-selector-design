# P001 Related Work Closure

## Pre-experiment search

The 2026-08-17 search focused on diffusion timestep subsequences, student-distribution distillation,
and curvature proxies.

- [DDIM](https://arxiv.org/abs/2010.02502) defines accelerated reverse processes on selected
  timestep subsequences. Its construction supports our requirement that a skipped transition and the
  next denoiser label refer to the same registered time; it does not prescribe Drive-OPD's two-step
  truncation.
- [Distilling Policy Distillation](https://proceedings.mlr.press/v89/czarnecki19a.html) analyzes the
  distinction between teacher- and student-policy state distributions and shows that naive on-policy
  distillation updates need not form a gradient vector field. This directly motivates describing our
  detached student rollout as a semi-gradient and testing long-horizon stability rather than asserting
  convergence from the scalar-looking loss.
- [Limitations of the Empirical Fisher Approximation](https://papers.nips.cc/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html)
  shows that empirical-Fisher substitutions generally lack the geometry often attributed to the true
  Fisher. Because DiffusionDrive uses a composite surrogate rather than one normalized likelihood,
  our EWC statistic is named and interpreted only as a per-example gradient second moment.
- The [A-GEM ICLR paper](https://iclr.cc/virtual/2019/poster/715) motivates a computationally light
  replay-gradient constraint. In an end-to-end driving network, however, an auxiliary replay forward
  also mutates normalization buffers unless explicitly controlled; P001 adds this driving-model state
  audit rather than changing the A-GEM projection.

## Post-anomaly search

After the 14,951-token reconstruction failed, the search shifted to dataset provenance and mutable
network state.

- [Datasheets for Datasets](https://www.microsoft.com/en-us/research/uploads/prod/2019/01/1803.09010.pdf)
  argues that dataset motivation, composition, collection, processing, intended use, distribution, and
  maintenance should be documented. Our cache receipt is a narrower executable counterpart: it records
  the exact source tree, selection rule, failures, builders, and token hashes required to regenerate a
  claim-bearing driving stream.
- The original [Batch Normalization paper](https://arxiv.org/abs/1502.03167) makes mini-batch
  statistics part of the model's training behavior. This supports treating A-GEM's replay forward as a
  state-changing operation, not merely an extra gradient query, when the model contains running buffers.

## Evidence boundary

These works support the audit criteria and terminology. None proves that scheduler-consistent OPD will
outperform LwF or replay on NAVSIM, and none resolves the missing local cache-generation rule. Those are
empirical and provenance obligations for the next paper round.

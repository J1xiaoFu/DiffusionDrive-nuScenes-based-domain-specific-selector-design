# Table P001-B. Local NAVSIM inventory contract

| inventory layer | tokens | relation established by P001 | reproducible source rule? |
|---|---:|---|---|
| Raw valid unit-stride allowlist | 103,288 | all configured allowlist tokens are valid 14-frame windows | yes |
| Declared `frame_interval=14` output | 7,457 | deterministic subset of the allowlist | yes |
| Existing training cache | 14,951 | 14.48% subset of allowlist; not the interval-14 output | **no** |
| Full Failure-Patch-CL manifest | 14,951 | token set and token-to-log map equal cache | inherits cache failure |
| Portable training union | 10,444 | exact union of the three train cells | inherits cache failure |
| Audit + sealed test union | 4,507 | disjoint from training | inherits cache failure |

Across 162 timestamp–vehicle sessions, the equal-session mean cache fraction is 0.1481 with a
descriptive 95% session bootstrap interval [0.1456, 0.1506]. This interval describes cache coverage;
it is not a model-performance confidence interval.

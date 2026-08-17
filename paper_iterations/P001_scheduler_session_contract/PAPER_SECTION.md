# P001 Paper Section: Making Student-Support Distillation Identifiable

## Motivation

An OPD–LwF comparison is interpretable only if both methods query valid diffusion states under the same
time convention and if the driving stream itself can be regenerated. We audit these prerequisites
before reporting another method ranking. This round makes four implementation-level contributions:
(i) a registered constant-stride trajectory-distillation schedule, (ii) explicit semi-gradient
semantics, (iii) state-neutral EWC/A-GEM baselines, and (iv) session-level inference. It also reports a
negative data-provenance result that blocks the next main-table experiment.

## Method

For query times (mathcal T=(10,0)), we require the initial forward-noise time to equal 10 and configure
the 1,000-step DDIM grid with 100 inference steps. Hence the single non-final scheduler call lands
exactly at (t=0). Student-support OPD and exogenous-support LwF share the same response MSE, forward
mode KL, teacher, contexts, query implementation, and two student/two teacher calls. OPD alone advances
the detached student-generated state; this is reported as a semi-gradient, not as a guaranteed
gradient flow.

The EWC control estimates a diagonal per-example official-loss gradient second moment. A-GEM computes
its replay gradient in train mode but restores every model buffer before applying the projected current
gradient. Statistical comparisons first average paired token differences within complete
timestamp–vehicle sessions and then bootstrap sessions; multi-seed intervals additionally resample
seeds. Invalid rows fail closed and metric families use Holm adjustment.

## Results

The scheduler mechanism test passes. Under identical clean trajectories and noise, the historical
heuristic returns a state 0.0198078 RMS from its declared (t=0) label; the corrected transition's RMS
is (1.538\times10^{-8}), with no extra denoiser query.

The data hypothesis fails. The 103,288 configured tokens are exactly the valid unit-stride allowlist,
whereas `frame_interval=14` produces 7,457 scenes. The existing cache contains a different 14,951-token
subset (14.48%) with no recorded selection rule or seed. Cache and full CL manifest are exactly equal,
and the portable 10,444-token set is exactly the training union, so the protocol arithmetic is correct
but its source selection is not reproducible. Across 162 sessions, mean cache coverage is 0.1481 (95%
descriptive session-bootstrap CI [0.1456, 0.1506]).

## Interpretation and limitation

The corrected loss is now an identifiable comparison of student and exogenous support, but its driving
benefit remains untested. The local historical checkpoints cannot support a pristine-paper claim
because both the old timestep heuristic and undeclared cache subset are upstream of their results.
The next round starts only after 06G regenerates the official scene inventory and cache from frozen
upstream code with a replayable receipt. This negative gate costs time, but prevents an irreproducible
data subset from becoming the paper's hidden experimental definition.

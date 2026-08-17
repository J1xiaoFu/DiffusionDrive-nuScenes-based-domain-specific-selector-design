# ICLR 2027 Blind Calibration Review — P000

**Reviewed artifact:** frozen commit `cfac3ce4a411493bb31828ca8fee4cb7f83e7ce7`  
**Review status:** independent, read-only calibration; not an acceptance decision  
**Repository state:** frozen commit is clean and the local `origin/codex/iclr2027-drive-opd-controller` tracking ref resolves to the same SHA; no files were edited.

## Five-sentence claim summary

The submission proposes Drive-OPD, a continual-learning adaptation of DiffusionDrive that distils perception and planning while querying the teacher on states visited by the student. It introduces Failure-Patch-CL and Chronological-CL protocols intended to preserve complete timestamp–vehicle sessions across train, audit, and test cells. The claimed mechanism is that student-support distillation better protects deployed behavior than LwF on ground-truth-noised states, with perception/planner context swaps intended to identify perception-induced planning drift. The frozen evidence establishes code paths, approximate compute costs, a modified-model Stage-1 checkpoint, protocol stratification, and several smoke-level numerical invariants, but contains no pristine official DiffusionDrive continual-learning outcome or multi-seed method comparison. The AutoResearch process is disciplined and unusually explicit about evidence boundaries, but its required paper packages, figures, blind review, author response, and user-audit loop have not yet been executed in this commit.

## Overall assessment

**Current answer: no, the frozen evidence does not yet make an ICLR-level paper credible.** It provides a promising research plan and useful engineering groundwork, but the central method comparison is both empirically absent on the official model and confounded in the current implementation. A credible paper remains possible by the deadlines only if the P0 issues below are resolved immediately and official Seed-0 evidence arrives early enough to abandon the method cleanly if the effect is absent.

The schedule is high risk. On August 17, the official environment, caches, T0, real-batch adapter gate, continual runs, statistics, paper package, and review-response loop are all pending, while results are scheduled to freeze September 7, four days before the abstract deadline and nine days before the paper deadline.

## Strengths

- The evidence boundary is commendably honest. The documents explicitly state that modified RAP results are not pristine DiffusionDrive evidence and that Drive-OPD has not been shown to beat LwF, replay, ALER, or any continual-learning baseline.
- The process separates engineering gates from scientific iterations and requires claims, counterfactuals, failure criteria, negative controls, artifact hashes, and claim-to-evidence mappings.
- Session-level allocation is conceptually preferable to token-level random splitting, and the validator checks duplicate sessions, logs, and tokens across cells.
- The plan correctly distinguishes step-matched replay from full-exposure replay and recognizes that additional optimization exposure is a confound.
- The code contains concrete implementations of perception distillation, student-support planning distillation, EWC, A-GEM, replay, ALER-style search, continual matrices, and clustered bootstrap summaries.
- The negative-result policy is appropriate: a failed round is supposed to narrow the claim or eliminate a method rather than trigger an unstructured sweep.
- Commit hygiene is good for the reviewed planning commit: the branch is clean and its local remote-tracking ref matches the reviewed SHA.

## Fatal weaknesses if unchanged

1. **There is no official result supporting the paper’s central claim.** The official 06G plan says the training/metric caches, compatible environment, and T0 were unfinished. The only continual results come from a modified RAP model; even the Stage-2 new-domain evaluation of that pilot was pending. A paper about stability–plasticity cannot be evaluated from old-domain changes alone.

2. **The claimed OPD-versus-LwF controlled comparison is not currently controlled.** The default OPD state is created by adding noise at timestep 8, then queried and stepped at timestep 10, followed by a query at timestep 0 without code establishing that the intermediate state is a valid timestep-0 state ([drive_opd.py](/home/khwang/domain-selector/selector_bench/selector_bench/continual/drive_opd.py:67), [drive_opd.py](/home/khwang/domain-selector/selector_bench/selector_bench/continual/drive_opd.py:466)). LwF instead constructs separate GT-noised states directly at timesteps 10 and 0. Thus the arms differ in scheduler semantics as well as support, contradicting the assertion that query-support distribution is the only controlled difference.

3. **“Real log incremental protocol” is not yet established.** Failure-Patch-CL retrospectively ranks a static dataset using evaluator-derived failure metrics; this is a constructed curriculum over real logs, not demonstrated deployment arrival order. Moreover, the local protocol contains 14,951 tokens while the official inventory reports 103,288 tokens over the same 1,192-log scale, with no committed reconciliation showing that all tokens from assigned logs are represented. Session assignment may be atomic while session contents remain sparsely sampled.

4. **The evidence is not independently inspectable.** Reported checkpoints, manifests, CSVs, losses, and remote test reports are referenced by prose paths and hashes but are absent from the frozen tree. A hash without the corresponding artifact or a committed signed manifest cannot validate the reported number.

## Major weaknesses

- The four-context swap is an algebraic intervention diagnostic, not yet a causal identification strategy. Passing a student representation into a teacher planner assumes cross-model latent alignment; after training, representation rotations or scale changes can make the hybrid context off-manifold.
- The ALER “nearest manifold” diagnostic compares each searched point against the same batch of states from which it was initialized ([training runner](/home/khwang/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:551)). The nearest distance can therefore reduce to the bounded search displacement and does not validate membership in a driving-state manifold.
- The EWC accumulator squares a gradient already aggregated over a batch and multiplies it by batch size ([baselines.py](/home/khwang/domain-selector/selector_bench/selector_bench/continual/baselines.py:61)). Unless the official loss exposes per-example gradients, this contains cross-example terms and is not the empirical Fisher claimed in the report.
- A-GEM performs an additional train-mode replay forward before stepping, which can update BN buffers using old data. Its difference from sequential learning is therefore not limited to gradient projection.
- The relevant unit tests are toy contracts. The OPD loss test uses an identity scheduler and never exercises the student rollout, timestep consistency, official adapter, or an end-to-end OPD/LwF gradient-equivalence control ([test_drive_opd_losses.py](/home/khwang/domain-selector/selector_bench/tests/test_drive_opd_losses.py:19)).
- The official 06G adapter and its reported 18 CPU tests are not present in the reviewed commit. The committed runner constructs the acknowledged RAP-modified agent with `distill_feature=True` and `latent=True` ([training runner](/home/khwang/domain-selector/selector_bench/scripts/41_train_drive_cl_diffusiondrive.py:116)).
- The prescribed AutoResearch artifacts are plans rather than completed research-loop outputs.

## Mathematical and theoretical gaps

- The Wasserstein statement requires \(h(x)=\ell(d(x,c))\) to be Lipschitz on the relevant joint state/context domain. Lipschitzness of \(d\) alone is insufficient without assumptions on the loss, bounded outputs, conditioning, and compatible supports.
- The bound only limits the difference between two distillation expectations. It does not imply that OPD produces smaller forgetting, higher PDMS, safer trajectories, or a better stability–plasticity frontier than LwF.
- Student-support rollout uses detached transitions, making the implementation a semi-gradient objective. The paper must distinguish optimizing responses on samples from the current policy from differentiating through the policy-induced state distribution.
- No formal relationship connects response MSE or mode KL to NC, DAC, TTC, comfort, progress, BWT, or closed-loop failure.
- The perception decomposition is an identity, not a causal mediation result. Causal language requires assumptions about intervention validity, representation compatibility, and absence of off-manifold hybrid effects.
- EMA momentum `0.99`, distillation weights, confidence masks, and timestep choices lack a hypothesis-derived scale or stability analysis.
- The “high-\(W_1\)” subgroup is not operationally defined. Selecting it after seeing method outcomes would create subgroup-selection bias.

## Experimental, baseline, statistical, physical-validity, and reproducibility gaps

**Experimental design**

- No official sequential, replay, LwF, or Drive-OPD result exists.
- No Stage-2 new-domain outcome exists even for the modified-model four-arm pilot.
- No Stage-3 result, chronological result, final-test result, three-seed result, or failure analysis exists.
- The source model/policy that generated the Failure-Patch ranking metrics is not documented in a committed artifact.
- The discrepancy between 14,951 protocol tokens and 103,288 official tokens must be resolved at token and log level.
- “Usable T0” is asserted from one audit PDMS without a frozen acceptance threshold, comparator, or seed variability.

**Baselines and fairness**

- Required core baselines should be sequential, step-matched replay, full-exposure replay, LwF, DER++, and Drive-OPD on identical official code. EWC and A-GEM are useful secondary baselines only after their estimators and BN behavior are corrected.
- Report actual unique current-token exposure, old-token exposure, forward/backward count, wall time, peak memory, stored bytes, and optimizer steps. “Same number of steps” alone is not an equal-information budget.
- Include a joint/all-seen-data upper bound and the unchanged T0 reference for every domain.
- The batch-48 OOM override currently names only four arms; every main-table method needs the same frozen batch and exposure contract.

**Statistics**

- The statistical atom is inconsistent: data splitting treats timestamp–vehicle session as atomic, while inference bootstraps segmented logs ([statistics.py](/home/khwang/domain-selector/selector_bench/selector_bench/continual/statistics.py:51)). Logs from the same session remain dependent.
- Equal weighting of log means and token-weighted means estimate different targets; the primary estimand is not preregistered.
- The plan does not specify how seeds and sessions are combined. Seeds are repeated training runs over the same evaluation sessions, not independent evaluation observations.
- There is no multiplicity policy across methods, stages, domains, five component metrics, and high-shift subgroups.
- “At least two significant transitions” lacks a minimum effect size, family definition, and decision threshold.
- Invalid evaluation rows are excluded from summaries, which can create optimistic complete-case estimates; failure handling and worst-case sensitivity are missing.
- FWT requires pre-exposure evaluation of future domains, but the selection protocol says only already-seen audit cells are evaluated. This must be reconciled before results are generated.

**Physical validity**

- A component below 1.0 is treated as a “failure,” causing 160 of 162 sessions to contain at least one failure. These are evaluator-defined surrogate failures, not demonstrated collisions or deployment incidents.
- Cross-model context swaps may be physically and representationally invalid.
- The ALER distance uses normalized latent Euclidean distance without kinematic, lane, acceleration, or interaction constraints.
- Clamp saturation, denormalized trajectory feasibility, and scheduler-consistent deployment states are not reported.
- NAVSIM PDM evidence alone cannot support real-world safety claims; the paper must consistently call the outcomes benchmark safety proxies.

**Reproducibility**

- No round manifest, result CSV, plotting script, environment lock, or official adapter is committed.
- `pyproject.toml` declares only NumPy, while the continual code requires PyTorch, SciPy, NAVSIM, and external model code.
- Absolute host paths appear throughout configurations and reports.
- The actual protocol manifest is ignored/out-of-tree, preventing review of the reported split and hash.
- Of 42 checked-in test methods, 16 target the continual package; none is a committed official-model integration test. I could not execute the suite on this read-only host because pytest, NumPy/PyTorch, and a usable temporary directory are unavailable, so this is an environment limitation rather than evidence that the tests themselves fail.

## Supported versus unsupported claims

| Claim | Assessment |
|---|---|
| The repository contains session-aware protocol construction and duplicate/leakage checks | **Supported by code design**, but the reported real manifest is absent |
| The runner calls one AdamW `step()` per current batch | **Supported by code inspection** |
| OPD and LwF nominally issue two student and two teacher denoiser queries | **Supported at control-flow level** |
| Distillation, EWC, A-GEM, and ALER utilities exist | **Supported at implementation/smoke level** |
| Runtime, memory, T0 PDMS, Fisher counts, and remote CPU gates have the reported values | **Attested in prose only; not independently verifiable from this commit** |
| Failure-Patch stages are statistically separated | **Unsupported**; only means are supplied, despite “significant” wording |
| OPD and LwF differ only in query-support distribution | **Contradicted by the timestep/state construction** |
| Drive-OPD preserves old ability while learning new ability | **Unsupported** |
| Drive-OPD beats LwF, replay, ALER, or sequential training | **Unsupported** |
| Perception drift causally changes planning behavior | **Unsupported** |
| EMA is preferable to a fixed teacher | **Unsupported** |
| Failure-Patch-CL creates reproducible catastrophic forgetting | **Unsupported** |
| Results generalize to pristine DiffusionDrive, GoalFlow, or real driving safety | **Unsupported** |

## Prioritized requirements

### P0 — required before the next claim-bearing iteration

1. Fix and freeze scheduler-consistent OPD/LwF state construction. Add an official-model test that records each state’s generating timestep, queried timestep, transition timestep, query count, loss, and gradient; include a negative control where both arms receive identical states and must produce identical losses and gradients.
2. Reconcile the 14,951-versus-103,288 token inventories. Commit a portable manifest proving per-session and per-log official-token coverage, exclusions, source-policy provenance, and a truly unopened evaluation split.
3. Produce one pristine official Seed-0 table containing T0, sequential, step-matched replay, full-exposure replay, LwF, and Drive-OPD on both old and new audit domains. If there is no conflict or no new-domain plasticity, stop claiming a stability–plasticity solution.
4. Replace log-level inference with a preregistered session-clustered, method-paired analysis, including invalid-run handling, minimum effect sizes, and a seed/session hierarchical aggregation rule.
5. Commit the complete paper iteration package with machine-readable results, artifact hashes, table source, confidence-interval figure, and plotting script. Prose-only hashes are insufficient.
6. Correct or explicitly downgrade the EWC estimator and eliminate the A-GEM BN-state confound before placing either baseline in a main comparison.

### P1 — required for a competitive full paper

- Run three frozen seeds across both stage transitions, with per-session PDMS and NC/DAC/TTC/comfort/progress results.
- Complete DER++ and compute/memory-matched replay comparisons; report unique data exposure as well as optimization steps.
- Validate context interventions using representation-alignment controls and a no-drift negative control before using causal terminology.
- Replace the ALER self-referential distance with a held-out reference bank excluding the source sample and add trajectory feasibility and real-log predictive-validity checks.
- Define the high-support-shift subgroup before method outcomes are opened.
- Archive the exact official adapter commit, environment, commands, data hashes, checkpoint hashes, failures, and final-test access record.

### P2 — desirable after the main claim survives

- Use Chronological-CL to test whether preservation suppresses positive transfer.
- Add robustness to an independently constructed Failure-Patch cohort or alternate construction policy.
- Include the mechanism figure and stability–plasticity frontier only after the underlying controlled effect exists.
- Keep GoalFlow, O-LoRA, and TALR out of the critical path unless the official DiffusionDrive result is already frozen.

## Deadline judgment

A credible submission is still possible, but not on the current evidence trajectory without an immediate P0 correction. The official adapter, scheduler invariant, inventory reconciliation, and first complete Seed-0 old/new-domain table should exist no later than roughly August 24; otherwise there is insufficient time to diagnose a null or confounded result, freeze three-seed evidence by September 4, and complete a defensible paper before September 16. If the Seed-0 controlled comparison is null, the appropriate response is to narrow the paper to a rigorously validated benchmark or negative mechanism result, not to expand method searches.

## Process-compliance check at the frozen commit

| Required item | Status |
|---|---|
| Paper-style Method–Experiment–Results package | **Partial/fail:** one monolithic round report exists, but no prescribed `paper_iterations/P000_*` package |
| Result tables | **Partial:** inline prose tables exist, but no provenance-linked round table files |
| Confidence-interval figures and generation scripts | **Fail:** no round-specific figure or plotting script |
| New context-free GPT-5.6-sol review | **Fail/not demonstrated:** none is committed; this review cannot certify itself as that exact model SKU |
| Author Response | **Fail:** absent |
| Candidate commit and push | **Pass for the planning commit only:** clean SHA and matching local remote-tracking ref; the full review-response package has not been committed |
| Round decision and claims-to-evidence files | **Fail:** absent |
| Separate non-blocking user-feedback audit | **Fail:** no `USER_AUDIT_REQUEST.md` or feedback record exists |

## Separate non-blocking user-feedback audit

This audit channel is absent from the frozen commit and should remain separate from scientific promotion gates. The useful user-facing questions are: what policy produced the deployment metrics, whether the intended benchmark covers all 103,288 official tokens or a declared subset, and whether the primary contribution should be the method or the constructed protocol. Any response should be recorded for traceability but should not silently alter the frozen protocol, estimator, or main-table configuration.

## Rating

- **Overall score:** **3/10**
- **Confidence:** **4/5**
- **Leaning:** **Reject**
- **Calibration:** Strong research governance and a plausible hypothesis, but no official method evidence and a current confound in the central comparison prevent an ICLR-level claim at this commit.


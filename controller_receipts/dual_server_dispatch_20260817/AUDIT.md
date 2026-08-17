# Dual-server Duty/Goal Contract Audit

## Decision

**PASS, limited to server responsibility and work-goal documentation.**

This audit does not pass or authorize a source handoff, NAVSIM or nuScenes cache, continual-learning
stage manifest, T0, real-model preflight, GPU training, evaluation, or paper-performance claim.

## Controller dispatch

- controller branch: `codex/iclr2027-dual-server-controller`
- dispatch commit: `3013da9bd0189b599c558b25fd3f7a804de28f37`
- dispatch parent: `3b6523de2f06f912e6e7834a0a5ce4166864f544`
- publication status: local only; HTTPS push failed because GitHub credentials were unavailable

## Jump-host publication

- publication repository: `J1xiaoFu/DiffusionDrive-nuScenes-based-domain-specific-selector-design`
- publication branch: `codex/iclr2027-dual-server-dispatch-publication`
- first publication commit: `d3be2e8e7eda17f8c9afade2de5da4aa26f559f9`
- parent: `90e5314d92fc127e65d2d8207569ee5c08a899b3`
- tree: `bf57f039fbece5f49e067c185cecfd185874e3db`
- transport: authenticated GitHub application acting as the local jump host; no `gh` installation
  or GitHub credential was placed on 07G or 06G
- content check: all 14 published Git blob SHAs matched the selected files from local controller
  commit `0fd2b0aae9ab26d3a2bc99c965539f67666f784e`
- 06G reachability: explicit HTTPS fetch succeeded; commit/parent/tree matched, the five declared
  receipt payload hashes replayed, and both published 06G snapshots were byte-identical to the 06G
  local duty/goal files
- current P0 acceptance publication: commit
  `4e71ea3a2adb5c455b1f82090e2ec166c8f23f40`, parent `e1ec56924f9a7a21bf2073b0dabc9fe30b4ade2d`,
  tree `89c53d3e32947f7c83faeb234473eb66b73533ca`
- 06G independently fetched that commit to `FETCH_HEAD`, read the acceptance contract completely,
  and matched its SHA256 `374d81fd57e4bd40bed396b4af26f86f1fa52f8f38e712de2b880fe45d295b7e`;
  branch, checkout, origin and official NAVSIM source stayed unchanged
- current R1 session-integrity publication: commit
  `4c42746002fe05cae78525e1e2ab5c8496504de4`, parent `37448d7bac7721857148d49d6899d515c5dd0438`,
  tree `35bb05a80290b2de5df0b79523fb34a1583cf710`; all three Git blob IDs matched the local
  controller files from `2274ec96f1a04b7b9d5f160d4cc541239c8f4927`, and the published manifest replayed
  the audit and intake SHA256 values
- final publication-receipt commit: `1a5ddf47acfb8150ab28247ac7cbe26a30bb9ae1`, parent
  `4c42746002fe05cae78525e1e2ab5c8496504de4`, tree
  `afaba2d606a80694a71d393b74568b30b0546458`; 06G independently fetched it over ordinary
  HTTPS, matched its exact three-path diff and Git blobs, then replayed the published `SHA256SUMS`
  set 9/9 in isolation
- 06G preserved its source/origin and recorded that replay in one audit-only commit
  `eb3cc6d0fbe8fa46daab5c9e1aed41abd49f7414`, parent
  `af9213b216b0260b184d3d3ae109a4ed1bed70b8`, tree
  `9484997b5f185d4586a70f6d2d454b075633ad3e`; its sole receipt JSON has SHA256
  `31c356dd13305952b3d34be16263bfcf7c716ef6e629d9d8f03e141b6fdb8cbb`
- controller publication-replay receipt commit: `1f6e850ec7832f0680ebf5472bafa2ea15289ca9`,
  parent `1a5ddf47acfb8150ab28247ac7cbe26a30bb9ae1`, tree
  `dd945bbff8632409795b47c3d0f2907f5777d5d8`; the publication branch ref, raw commit
  parent/tree, exact three-path diff and all three Git blobs were independently replayed, and each
  fetched blob was byte-identical to the local controller file from `ae223719d82ae31c1a6bddea04ba3f11f85e091a`
- 07G P0 RC source publication: `02e2784cb39cd636af6f582c2ff3f72d9274c58f`, parent
  `94a30b148e3a7c6fe1ecdd2f8ce493ab8440946b`, tree
  `b95c40170c29c8408cc258ea22a173d2167d8a66`; the controller replayed the branch ref,
  raw commit and recursive tree, matching 37/37 frozen source blobs and 4/4 controller receipt
  blobs. The non-forced fast-forward has 35 actual diff paths because six source paths were already
  byte-identical in its parent.
- final 07G P0 RC publication receipt: `d1665712535e101c948bec2bb7f7c21e64fe1f36`,
  parent `02e2784cb39cd636af6f582c2ff3f72d9274c58f`, tree
  `7ba0facb01a1dc7f4354d32964127a252f5dc2c2`; 06G fetched it to `FETCH_HEAD`, matched the exact
  three-path diff and all blobs, replayed the published `SHA256SUMS` 14/14, and recomputed the
  37-file source bundle with zero missing or mismatched files.

This is a dispatch/receipt publication snapshot. It deliberately excludes the rejected/unreleased
`9dabf423` claim-protocol source and cannot serve as a source/transfer execution authorization.

The published acceptance contract itself records `FAIL / OPEN P0`. Its reachability PASS only proves
that both servers can audit the same gate bytes; it does not prove the rejected protocol has been fixed.

## 07G P0 RC controller acceptance

- isolated RC branch: `codex/iclr2027-drive-opd-07g-p0-rc`
- commit: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`, parent
  `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`, tree
  `0e6e9a0adda280765ce41ab377de9bbbaea0cff1`; exactly one commit, 34 changed paths and a
  clean worktree
- controller independently matched the exact name-status inventory SHA256
  `f8adebf5a5287c79d5626c8dee2b9ba9c2b0236039e5c2174fcb1cce95df4537`, path/content
  inventory SHA256 `b07949de23b0361c8a362d000d0d0cbe824f9649689e865f855a723f078cb44d`
  and every advertised path hash
- independent five-suite replay: 49 passed and 9 subtests in 91.50 s; output SHA256
  `a27d1444495b2756f7584fbda45791916bbda9eae8a356d16b6566e77973fae3`
- independent P0 replay: 12/12 attack families and 19/19 mutation receipts passed; every mutated
  CLI returned nonzero before a claim output existed, and the positive fixture passed
- frozen migration manifest: 37 source files, logical bundle SHA256
  `ee681bf3cdb0cc2dfd198f860ce927971d738904250daacb2d8cfbf47349de93`, manifest SHA256
  `70b9f99bc96633a6be9a9437b106e48dc1d8effdac55fbc8439869c3dbc2ed78`

Decision: **PASS only for exact RC identity, static/CPU/P0-fixture checks, source-only manifest
freeze, publication byte identity and independent 06G destination replay.** Fresh context-free review
remains pending. Dataset provenance/adapter, session-atomic CL manifest, cache, real-model preflight,
T0, GPU, training, evaluation and performance gates remain stopped.

## 06G P0 RC destination replay

- 06G audit-only commit: `7bcfbf5f173cc567228904bc76413c508194dfe7`, parent
  `eb3cc6d0fbe8fa46daab5c9e1aed41abd49f7414`, tree
  `2717af0aa87c8abf82eecd713244eb160be8ff1f`; its sole receipt JSON has SHA256
  `89503b3763e31a8383fe02bb6746783344b474a2cd8e11108730466fdef045e5` and Git blob
  `37e8df8901e6ab5f706b7f8eb2b83b956be9f521`
- controller reconstructed the receipt and deterministic XZ from the structured task log; receipt,
  XZ and decompressed full-index diff byte counts and SHA256 values all matched the read-only export
- public replay: final receipt commit/parent/tree and exact path set matched; published checksum set
  passed 14/14; the 37-file logical bundle recomputed to
  `ee681bf3cdb0cc2dfd198f860ce927971d738904250daacb2d8cfbf47349de93`
- isolated destination tests: 49 passed in 151.28 s; P0 harness passed 12/12 attack families and
  19/19 mutation receipts, with every mutated CLI rejected before claim output and the positive
  fixture accepted
- no cache, bytecode, pytest cache or checkpoint was created; no real model was imported and no
  GPU/CUDA action occurred; 06G branch, checkout, origin and NAVSIM source stayed unchanged

Decision: **PASS only for public reachability, byte identity, CPU source replay and P0 fixture
replay.** Fresh context-free exact-RC blind review remains pending. Dataset provenance/adapter,
session-atomic CL manifest, cache, real-model preflight, T0, model, GPU, training, evaluation and
performance claims remain stopped.

## Exact-RC context-free review freeze

- review ID: `P001-DRIVE-OPD-P0-RC-EXACT`
- target: `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`, parent
  `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`, tree
  `0e6e9a0adda280765ce41ab377de9bbbaea0cff1`
- required reviewer: a newly created context-free task using `gpt-5.6-sol` at `xhigh`, with no
  parent conversation or preloaded prior-review conclusion
- immutable inputs: the public source commit `02e2784cb39cd636af6f582c2ff3f72d9274c58f`,
  final receipt commit `d1665712535e101c948bec2bb7f7c21e64fe1f36`, 37-file bundle SHA256
  `ee681bf3cdb0cc2dfd198f860ce927971d738904250daacb2d8cfbf47349de93` and 06G destination
  receipt SHA256 `89503b3763e31a8383fe02bb6746783344b474a2cd8e11108730466fdef045e5`
- required review: independent code-first audit, all twelve historical required-fix dispositions,
  isolated five-suite and P0-harness replay, new adversarial bypass attempts, exact source/receipt
  hash replay and explicit P0/P1/P2 findings
- pass boundary: no open P0 and independent closure of every source/CPU-testable historical P0;
  all P1/P2 items must be dispositioned, and the unopened real-adapter/data/model gates must remain
  explicit

Decision: **FROZEN, NOT STARTED.** No review task was created by this freeze. Even a future PASS can
clear only the fresh context-free exact-RC review gate; dataset provenance/adapter, session-atomic CL
manifest, cache, real-model preflight, T0, model, GPU, training, evaluation and performance claims
remain stopped pending separate controller decisions and receipts.

## RC1 fresh blind-review intake

- reviewer task: `01a01004-4a38-70f2-9981-4e26d1e834cb`; exact RC, parent, tree, 34-path count and
  clean-before/after status matched
- structured-log provenance: model `gpt-5.6-sol`, reasoning effort `medium`; the task was fresh and
  context-free, but it began before the controller's later xhigh review freeze and therefore did not
  satisfy that strict xhigh contract
- exact final review: 10,831 message bytes, SHA256
  `a3f37e50be8a4e3b177fa1fc40210e68a64cab6393d37f15483c50d72d375a6b`; preserved with a
  conventional terminal newline in `07g_p0_rc_blind_review_rc1.md`
- reproduced positives: 49 tests plus 9 subtests passed in 96.38 s; all 12 named attack families and
  19 mutation instances rejected before claim output; filename/order-invariant positive fixture passed
- verdict: **4/10 REJECT, confidence 5/5**
- open P0: final-test chronology accepts self-declared/timestamp-shaped authorization without an
  authoritative ledger or time ordering; production observed presentation/identity budgets partly
  copy target values; seed power/calibration claims are not bound to or replayed from script 55
- open P1: ALER hardcodes query counts outside `capture_query_audit`; shared-cell family validation
  may reject valid common-baseline designs; transient storage is declared zero; chronology and seed
  positive fixtures need adversarial negative coverage
- controller source inspection confirmed the three P0 code paths. The reviewer could not resolve the
  external publication objects on its configured origin, while the controller's separate immutable
  fetch and 37-file bundle replay had already passed; that transport distinction does not weaken the
  local exact-RC P0 findings.

Decision: **RC1 REJECTED / P0 OPEN.** The lower-than-frozen reasoning effort cannot promote RC1, and
its executable P0 rejection is sufficient to terminate RC1. A future RC2 requires separate explicit
authorization, forward-only repair and a new exact-RC xhigh review. Dataset provenance/adapter,
session-atomic CL manifest, cache, real-model preflight, T0, model, GPU, training, evaluation and
performance claims remain stopped.

## RC2 forward-only source/CPU authorization

- immutable base: exact RC1 `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`, parent
  `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`, tree
  `0e6e9a0adda280765ce41ab377de9bbbaea0cff1`
- authorized destination: new branch `codex/iclr2027-drive-opd-07g-p0-rc2` in new worktree
  `/home/khwang/domain-selector-p0-rc2`, with exactly one non-merge commit whose parent is RC1
- P0 closure: authoritative ledger-backed final-test chronology and real Git time ordering; measured
  per-batch production and A-GEM budgets with stable identities; executable deterministic seed-design
  replay bound to pilot inputs, program source, invocation and exact normalized results
- included P1: measured ALER query counts, shared-cell/common-baseline family handling, measured
  transient storage and adversarial chronology/seed harness coverage
- authorized activity: source/config/test/paper-response edits, static checks, synthetic temporary-Git
  fixtures, CPU tests with CUDA hidden, one auditable RC2 commit and a complete identity/hash receipt
- immutable boundary: RC1, research, recovery and controller refs may not move; no merge/rebase/amend,
  reset, force-push, publication, review dispatch or 06G dispatch is authorized

Decision: **AUTHORIZED FOR RC2 SOURCE/CPU REPAIR ONLY.** RC2 publication, exact-RC2 xhigh review and
06G isolated replay remain controller-held future decisions. Dataset/raw-data access, session-atomic
CL manifests, cache, real-model preflight, T0, model, CUDA/GPU, training, evaluation and performance
claims remain stopped.

## RC2 exact-commit controller intake

- candidate: `codex/iclr2027-drive-opd-07g-p0-rc2@d527274d368b6d6def5809b002324b31e88ae1bd`
- identity: sole parent is immutable RC1 `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`, tree
  `dce36d261b8a0f4bde4519a3a393471c54b05356`, exactly one non-merge commit, 22 changed paths and a
  clean source worktree
- independent controller replay: exact detached commit, CUDA hidden, bytecode and pytest cache
  disabled; 62 tests plus 21 subtests passed in 134.69 seconds
- adversarial replay: 15/15 named families and 32/32 mutation instances rejected fail-closed with no
  claim artifact; the positive filename/order-invariant fixture passed
- static and artifact boundary: seven JSON files parsed, 114 Python files compiled in memory,
  `git show --check` passed, and the detached tree remained clean with no pytest cache, bytecode,
  model or checkpoint paths
- intake receipt: `07g_p0_rc2_acceptance.json`, SHA256
  `ba1146436af28089ad155885ad95b75a8eb440ad61239ff06bc7b66088247fe4`

Decision: **ACCEPTED FOR SOURCE PUBLICATION AND A NEW EXACT-RC2 CONTEXT-FREE REVIEW ONLY.** The three
RC1 P0 findings and associated P1 hardening are candidate-closed, not review-closed. Isolated 06G
source/CPU replay remains stopped until the publication and review boundary is frozen. Dataset and
adapter execution, session-atomic CL manifests, cache, real-model preflight, T0, model, CUDA/GPU,
training, evaluation and performance claims remain stopped.

## RC2 source-transfer freeze

- source: `codex/iclr2027-drive-opd-07g-p0-rc2@d527274d368b6d6def5809b002324b31e88ae1bd`
- manifest: 41 source/config/test files, SHA256
  `9eadb5064f5bebd8234e910fb3a6e6aa8dc16aeb7d077fbd30803cda0af66932`
- logical source bundle: `5bc6f3339f1af5ccde9c21a8068b94053ae00ea920eed9f3ade7613d70a5cd88`
- frozen generator: `selector_bench/scripts/51_freeze_drive_opd_06g_transfer.py`, SHA256
  `958d537d6707f130f72834790ba8bdfcd3192431d214d5fdb4241ea61e023a78`
- freeze receipt: `07g_p0_rc2_transfer_freeze.txt`, SHA256
  `b7f285b527174e0c089152feb7330c9505b9a19a57366887278a802b40678475`

Decision: **SOURCE BUNDLE FROZEN LOCALLY.** Public reachability is not yet established. Exact-RC2
review may use this immutable controller freeze, but isolated 06G replay remains stopped until review
and public fetchability both pass. All execution gates remain stopped.

## 06G R1 session-integrity intake

- remote audit commit: `af9213b216b0260b184d3d3ae109a4ed1bed70b8`, parent
  `0c4fb7cbc4b4ecc4bc056027a011fa3bd1d29d27`, advertised tree
  `875f78ee1f56e0d23b6d54f21afcf792b4b9c0db`
- transport: eight exact base64 chunks reconstructed from the controller host's structured task log;
  every chunk SHA256, the 1,157,132-character aggregate SHA256, the 867,848-byte XZ SHA256 and the
  2,116,560-byte patch SHA256 matched the remote manifest; `xz -t` passed
- patch boundary: five new audit/receipt files and zero modifications to existing source; isolated
  apply and reverse-apply checks passed, and every advertised payload hash replayed
- controller-derived table replay: canonical 162-row JSONL; 101/61 train/validation sessions,
  978/214 logs and 85,109/18,179 tokens; all session, log and token identifiers are globally unique,
  train/validation overlap is zero, and missing/duplicate/invalid chronology counts are all zero
- fail-closed replay: one positive and three adversarial synthetic cases passed against the received
  analyzer without reading raw NAVSIM, importing a model or using a GPU
- unresolved boundary: the remote commit/tree objects, sealed R1 input CSV/token payloads and exact
  parent helper files were not independently fetched on the controller; the full producer command and
  helper source/function hashes were therefore not replayed here

Decision: **PASS only for remote-patch intake, derived-table integrity and prospective session-atomic
representability.** It is not a frozen CL stage manifest or an independently replayed cache-builder
receipt, and it does not unlock cache construction, T0, GPU work, training, evaluation or a paper claim.

## 07G receipt

- branch: `codex/iclr2027-drive-opd-07g-research`
- parent: `9dabf42371e8dde72e1b2061bac4fa9d11b34230`
- commit: `d1cee1916f8f2329b4e3f3a1140e79e5d22c124e`
- tree: `ea171b20870fc59f4678b0b3a52f5945b9029fb8`
- changed paths: `AGENTS.md`, `ICLR2027_07G_NUSCENES_RESEARCH_GOAL_20260817.md`
- controller verification: commit/parent/tree, exact path set and both advertised file SHA256 values
  independently matched from the shared local Git object database
- worktree boundary: the 07G worktree still contains pre-existing P0 source/review changes not
  included in the duty/goal commit

07G is assigned Drive-OPD method research and the independent nuScenes mechanism/external-validity
lane. It remains source/CPU/N0-only pending P0 repair, a new frozen blind-review pass, data receipts,
an idle-GPU receipt and the disk safety floor.

## 06G receipt

- branch: `codex/iclr2027-diffusiondrive-06g`
- parent: `1bc9cc2957eeccc6927f3dc826135ce85bfae790`
- commit: `0c4fb7cbc4b4ecc4bc056027a011fa3bd1d29d27`
- tree: `b457366d0953d7a2370d7491430be52ad5ca0a89`
- changed paths: `AGENTS.md`, `ICLR2027_06G_EXECUTION_OBJECTIVE.md`
- remote attestation: clean worktree, exactly two documentation paths, no tracked `navsim/` diff,
  no cache/T0/GPU action
- portable evidence: 06G supplied the raw commit header and complete ordinary full-index diff with
  advertised diff SHA256 `0ac5ce3f4e8593ada59ad8d0f0f728868accbea4acedeb7dfbf613431bccc4aa`
- controller verification: both reconstructed file snapshots independently match the advertised
  file SHA256 values; the current controller host still cannot resolve SSH alias `06G`, so the raw
  remote commit/tree object and advertised full-diff hash have not been independently fetched

06G is assigned the primary pristine DiffusionDrive/NAVSIM evidence lane. R1 remains limited to
source population and official train/validation partition reconstruction. Cache, CL stages, T0 and
GPU remain stopped.

## Baseline decision

- required core: T0 reference, sequential, step-matched replay, LwF and Drive-OPD
- NAVSIM main extensions: full-exposure replay, trajectory-defined DER++, joint/all-seen oracle
- gated ablations: fixed/EMA OPD, planner-only, perception-only and identical-state/no-drift controls
- secondary only after real-adapter receipts: EWC and A-GEM
- off critical path: ALER-Drive, TALR, O-LoRA and GoalFlow/Flow Matching

nuScenes and NAVSIM maintain distinct manifests, metrics, result tables, statistical families and
promotion gates. Neither lane may unlock or substitute for the other.

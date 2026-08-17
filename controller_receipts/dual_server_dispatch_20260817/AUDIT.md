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

## RC2 source publication and context-free review freeze

- public source commit: `d262e3c14ba2bea31b48bd85ab1d721987b826b3`, sole parent
  `fda179f007e64614467791f1ab9ed3c229c8f0bb`, tree
  `c125b76a2f579016c2d4bcdafd9eb0db996f0a15`
- publication used exact Git blobs and an exact tree, then one non-force ref update; an independent
  HTTPS fetch reproduced the commit, parent and tree
- local deterministic publication commit `59b1c93bbf7e81a481d8882603af9e8b4d8e910e` has the same parent
  and tree; its content diff against the public commit is empty
- public full-index diff SHA256:
  `25e031f4ef7b78cd99b2b8e7ee3e0ecdf027a79560310366a7adb0d41ee37362`
- publication receipt: `07g_p0_rc2_publication.txt`
- frozen review: `P001-DRIVE-OPD-P0-RC2-EXACT`, new context-free `gpt-5.6-sol` at `xhigh`, read-only
  exact-source/CPU audit with all RC1 failures and twelve historical requirements mandatory
- review request: `07g_p0_rc2_context_free_review_request.md`
- review manifest: `07g_p0_rc2_context_free_review_manifest.json`

Decision: **PUBLIC SOURCE IDENTITY PASSED; EXACT-RC2 REVIEW AUTHORIZED BUT NOT YET PASSED.** Public
fetchability and byte identity do not close the RC1 P0 findings. Isolated 06G source replay remains
stopped pending the fresh review. Raw data, dataset adapters, session-atomic CL manifests, cache,
real-model preflight, T0, model, CUDA/GPU, training, evaluation and performance claims remain stopped.

## RC2 fresh context-free xhigh blind-review intake

- authoritative reviewer task: `01a0106d-4a96-72d2-aa90-da0acd5f72c3`, fresh zero-parent
  `gpt-5.6-sol` at `xhigh`; the earlier task `01a0105f-c503-7773-b2fa-757ea09e859f`
  infrastructure-failed without a final verdict and is not authoritative
- reviewed identity: public freeze `81973da8c2768e71f5f79cefc17888e19d2f6cbd`, logical source
  `d527274d368b6d6def5809b002324b31e88ae1bd`, public source
  `d262e3c14ba2bea31b48bd85ab1d721987b826b3`; 28/28 receipt rows and 41/41 logical plus
  41/41 public source blobs matched, with zero logical/public source-blob differences
- six isolated CPU suites: 62 tests plus 21 subtests passed; expanded shipped harness passed
  15/15 attack nodes and rejected 32/32 included mutations before claim output
- executable adversarial review found four uncovered P0 bypasses: non-monotonic Git and non-strict
  family/authorization/access chronology; cross-family sealed-access receipt transplantation;
  optional rather than mandatory multi-candidate common-baseline cells; and non-finite seed-design
  inputs producing and replaying estimated power `1.0`
- verdict: **3/10 REJECT / FAIL, confidence 5/5**; source/CPU P0 contract `NO`, non-promoting
  source-only publication or isolated CPU replay `YES` only when explicitly labeled rejected, and
  every execution gate `NO`
- exact task-final message: 17,688 bytes, SHA256
  `3dfa8d89b7222eaba0238566ccf38abcb2e801a0014485b008a171b0f729a431`; stored report
  `07g_p0_rc2_blind_review.md` removes only two Markdown hard-break trailing spaces and appends one
  terminal newline, yielding 17,685 bytes and SHA256
  `78664e96f07209691f633330bad51e6e642c6b3c33ec1954fa928ad0087c66bd`
- structured controller intake: `07g_p0_rc2_blind_review_intake.json`

Decision: **RC2 REJECTED / FOUR P0 FINDINGS OPEN.** No 06G dispatch is authorized. A future RC3 must
be forward-only, add the four reproductions to its frozen harness, and pass a new context-free exact-RC
review. Dataset/raw-data access, provenance/adapter work, session-atomic CL manifests, cache,
real-model preflight, T0, model, CUDA/GPU, training, evaluation and performance claims remain stopped.

### Rejected-result publication

- public commit `e36eb38da827290c277a0f5ffc615456046f5524`, parent
  `81973da8c2768e71f5f79cefc17888e19d2f6cbd`, tree
  `2cce21835436150ad65fe03805f294c6196d010d`; one non-force update and exactly four changed paths
- preserved local publication commit `3a4ca9ee5edacc8fe40e859595a025d5329ef44a`, parent
  `389e07b54fd0c220d9637b24d17788cddd6a16be`, has the exact same tree; public/local tree diff is empty
- ordinary HTTPS fetch to `FETCH_HEAD` reproduced the public commit, parent, tree, exact path set and
  `git show --check`; receipt: `07g_p0_rc2_blind_review_publication.txt`

Decision: **PUBLICATION PASS, REJECTED/NON-PROMOTING ONLY.** No isolated 06G replay is dispatched
because gate A failed. The public receipt cannot authorize data, cache, model or execution activity.

## RC3 forward-only source/CPU authorization

- authoritative basis: fresh context-free exact-RC2 `gpt-5.6-sol` `xhigh` review task
  `01a0106d-4a96-72d2-aa90-da0acd5f72c3`, verdict **3/10 REJECT / FAIL**, confidence 5/5
- immutable base: exact RC2 `d527274d368b6d6def5809b002324b31e88ae1bd`, parent
  `7edaaa05c199174e5b7d5f0cf46e45f50cd7e9f4`, tree
  `dce36d261b8a0f4bde4519a3a393471c54b05356`
- authorized destination: existing clean branch `codex/iclr2027-drive-opd-07g-p0-rc3` at exact RC2 in
  `/home/khwang/domain-selector-p0-rc3`, with exactly one new non-merge commit whose sole parent is RC2
- scoped P0 closure: monotonic raw-Git committer chronology plus strict four-time inequality; sealed
  final-test evidence bound to the active comparison family; mandatory two-or-more-candidate shared
  cells against one exact common baseline; finite-only seed-design inputs, replay and outputs
- frozen attacks: add all four executable reviewer reproductions after the 15 existing families;
  require at least 19 families and 36 mutation receipts, all nonzero/no-claim, plus one simultaneously
  valid strict/family-bound/multi-candidate/finite/order-invariant positive fixture
- non-regression: measured main/A-GEM budgets, ALER query capture, measured transient storage and all
  historical P0 closures remain mandatory
- authorized activity: scoped source/config/test/paper-response edits, synthetic temporary-Git fixtures,
  CUDA-hidden source-only CPU tests, temporary receipts and one auditable commit

Decision: **AUTHORIZED FOR RC3 SOURCE/CPU REPAIR ONLY.** RC3 may not publish, push, create a review
task or dispatch 06G work. Dataset/raw-data access, provenance/adapter work, session-atomic CL manifests,
cache, real-model preflight, T0, model, CUDA/GPU, training, evaluation, final-test unsealing and
performance claims remain stopped.

### Authorization publication

- public commit `7892cbcd06b99f44dc593813459b12e5eb84f902`, sole parent
  `e36eb38da827290c277a0f5ffc615456046f5524`, tree
  `559f03f519b5c9e01515708b105813172632a68c`; exactly four changed paths and one non-force update
- preserved local publication commit `0bc3e4521534a55a370293a6ff34aaca2f752ca5`, parent
  `3a4ca9ee5edacc8fe40e859595a025d5329ef44a`, has the exact same tree
- all four blob identities matched; ordinary HTTPS fetch reproduced the public commit, parent, tree,
  path set and `git show --check`; public full-index diff SHA256
  `d47417076f11cdca13ed1ebc0faa00ef3c005c4115cbf906c3d1ea27797721f4`
- receipt: `07g_p0_rc3_forward_authorization_publication.txt`

Decision: **PUBLICATION PASS FOR RC3 SOURCE/CPU AUTHORIZATION ONLY.** The publication conveys no
data, cache, model or execution permission and does not dispatch 06G work.

## RC3 exact-commit controller intake

- candidate: `codex/iclr2027-drive-opd-07g-p0-rc3@79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`
- identity: sole parent is immutable RC2 `d527274d368b6d6def5809b002324b31e88ae1bd`, tree
  `3ec47127a6d7c85326bbcf1d72ef76e2971f100d`, exactly one non-merge commit, 29 changed paths and a
  clean source worktree with no remote-containing ref
- independent controller replay: exact detached commit with CUDA hidden, bytecode and pytest cache
  disabled; 67 tests plus 73 subtests passed in 266.54 seconds; output SHA256
  `628db07ce5baf3dfe2e0eb31834d77b82d2d207a4efc697ac7628ab679971fb8`
- adversarial replay: 19/19 named families and 87/87 mutation instances rejected with nonzero CLI
  status before any claim artifact existed; the strict, family-bound, multi-candidate, finite and
  filename/order-invariant positive fixture passed
- targeted attack-output SHA256: P0-16 strict chronology
  `1e00b1739e0759f021b04825f70624f3743014d44879cd8b952379757d332dfa`; P0-17 active-family binding
  `64429b357291df1799c837728fb357e54375a73f6af3797f95aad7e15466bf1f`; P0-18 shared-cell
  multiplicity `af2f664907a6404136a4e2f6863f89b9979103aba65fc9c70765ac9bce986778`; P0-19 finite seed design
  `3753a059bea5a4b16221e139356e350303f4b44ed9c69efd608e33ab02966c0d`
- static and artifact boundary: `git show --check` passed; both changed JSON files parsed; 22 modified
  Python files parsed as AST; all 19 protocol JSON-emitting scripts use fail-closed
  `allow_nan=False`; the detached tree stayed clean with no pytest cache, bytecode, model or
  checkpoint artifacts
- structured intake: `07g_p0_rc3_acceptance.json`

Decision: **ACCEPTED FOR SOURCE PUBLICATION AND A NEW EXACT-RC3 CONTEXT-FREE REVIEW ONLY.** The four
RC2 P0 findings and historical source/CPU P0 protections are candidate-closed, not review-closed.
Isolated 06G source/CPU replay remains stopped until publication and review both pass. Dataset/raw-data
access, provenance/adapter work, session-atomic CL manifests, cache, real-model preflight, T0, model,
CUDA/GPU, training, evaluation, final-test unsealing and performance claims remain stopped.

## RC3 source-transfer freeze

- accepted source: `codex/iclr2027-drive-opd-07g-p0-rc3@79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`,
  parent `d527274d368b6d6def5809b002324b31e88ae1bd`, tree
  `3ec47127a6d7c85326bbcf1d72ef76e2971f100d`; controller acceptance commit
  `d026fd762b2472d5f3d01981dfbfd5d2da6bc28c`
- generator source: `selector_bench/scripts/51_freeze_drive_opd_06g_transfer.py`, SHA256
  `49ba27be6e553d1d8d4b670a2b16ec5ec0d356b1b8019462b71f86f1c43fa7f8`
- manifest: `07g_p0_rc3_transfer_manifest.json`, 7,434 bytes, 41 source files, SHA256
  `ff8ca94ebd3dd39abb048dabe8c8a87eaa383465b5cdb899d74cc6fbe2bcd437`
- independently recomputed logical source bundle SHA256:
  `68259f07dc97db312c00c71dcd18a899502588b4eb303ee12cce585e1cc894b1`, exactly matching the
  manifest, with zero missing or mismatched source files
- source worktree was clean before and after generation; no data, cache, model or execution artifact
  was read or created
- freeze receipt: `07g_p0_rc3_transfer_freeze.txt`

Decision: **SOURCE BUNDLE FROZEN LOCALLY FOR PUBLICATION AND EXACT-RC3 REVIEW ONLY.** Public
reachability is not yet established. Isolated 06G replay remains stopped until publication and fresh
review both pass. Every dataset, adapter, cache, model, T0, GPU, training, evaluation, final-test and
performance gate remains stopped.

### Exact source publication

- public commit `b1e1991afe7b9b14fd83f7510394c0ffb77ad700`, sole parent
  `7892cbcd06b99f44dc593813459b12e5eb84f902`, tree
  `1090673c5e8f148f2ec1b127044fa306b51b19cd`; exactly 28 changed paths and one non-force update
- preserved local publication commit `1c532e43fcb4b3c071c9fb55433cc0769dbf9032`, sole parent
  `0bc3e4521534a55a370293a6ff34aaca2f752ca5`, has the exact same tree
- ordinary anonymous HTTPS replay reproduced commit, parent, tree, path set and `git show --check`;
  the 7,434-byte manifest SHA256 is
  `ff8ca94ebd3dd39abb048dabe8c8a87eaa383465b5cdb899d74cc6fbe2bcd437`
- all 41 source blob SHA256 values matched with no missing files; independently recomputed logical
  bundle `68259f07dc97db312c00c71dcd18a899502588b4eb303ee12cce585e1cc894b1`
  exactly matched the manifest
- the publication tree differs by design from the whole RC3 worktree tree because the publication
  branch has a different parent/controller-receipt layout; exactness is bound at source blobs, bundle
  and source commit identity
- receipt: `07g_p0_rc3_publication.txt`

Decision: **PUBLICATION PASS FOR EXACT RC3 SOURCE AND RECEIPTS ONLY.** A fresh context-free exact-RC3
source/CPU blind review is now authorized. Isolated 06G replay and every data, adapter, cache, model,
T0, GPU, training, evaluation, final-test and performance gate remain stopped.

## RC3 fresh context-free xhigh blind review

- authoritative review task: `01a010f6-4606-7a51-a482-be5a01fb6d1e`; actual session metadata was
  independently checked as `gpt-5.6-sol` with `xhigh` reasoning effort; the earlier task
  `01a010ef-c27d-76d0-a3da-fb367e6ddf99` ran at medium effort, self-invalidated, returned no verdict
  and is not used as evidence
- exact reviewed source: `codex/iclr2027-drive-opd-07g-p0-rc3@79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`,
  sole parent `d527274d368b6d6def5809b002324b31e88ae1bd`, tree
  `3ec47127a6d7c85326bbcf1d72ef76e2971f100d`; reviewer worktree clean before and after with no
  pytest cache or Python bytecode
- public transfer independently reproduced: public commit `b1e1991afe7b9b14fd83f7510394c0ffb77ad700`,
  sole parent `7892cbcd06b99f44dc593813459b12e5eb84f902`, tree
  `1090673c5e8f148f2ec1b127044fa306b51b19cd`; all 41 local and public source hashes matched transfer
  manifest SHA256 `ff8ca94ebd3dd39abb048dabe8c8a87eaa383465b5cdb899d74cc6fbe2bcd437`
  and bundle `68259f07dc97db312c00c71dcd18a899502588b4eb303ee12cce585e1cc894b1`
- safe CPU reproduction: all six suites passed, totaling 67 tests plus 73 subtests; the standalone
  harness passed its shipped 19/19 families and rejected 87/87 mutations nonzero before any claim
  output; the higher-level audit also exited zero
- verdict: **3/10 REJECT / FAIL**, confidence **5/5**; the green source suite and attack harness do
  not close two newly reproduced P0s
- P0-1: seed design calibrates one anonymous seven-metric comparison while the frozen confirmatory
  family contains two comparisons and 14 hypotheses; the implementation chooses nine seeds although
  the stated full-family sign-resolution rule requires ten, and the seed calculation is not aligned
  with downstream bootstrap p-values; reproduction stdout SHA256
  `b28bc3fde2cfeef0848d5133ed6e110df20ea21316ac224ecc4941f1fd3b306f`
- P0-2: coercive resource validation accepted Boolean, fractional, string and non-finite values,
  including `transient_bytes=true`, `transient_bytes=1.5`, `wall_seconds=NaN`,
  `wall_seconds="nan"` and `persistent_bytes=true`; malformed evidence reached a final 14-hypothesis
  Holm output; direct bypass SHA256 `d3e3633854f9a9758ea6baeef4e699264e8ba2f76fcb5a2a89ec077e87b0713e`,
  end-to-end stdout SHA256 `ed39777ac687dc5da674bf39c84623e54b26eac3513089b21d0eee92a5cd46c8`,
  emitted global-result SHA256 `4ba3e605804fa11b9696c793aef9caf7d1bccb01fdf23f19bdfc695ccf08c15d`
- complete report: `07g_p0_rc3_blind_review.md`; structured intake:
  `07g_p0_rc3_blind_review_intake.json`

Decision: **RC3 REJECTED WITH TWO OPEN P0 FINDINGS.** Gate A is NO. Gate B is YES only for immutable,
explicitly rejected and non-promoting source bytes/CPU fixtures. Gate C is NO, so no 06G source replay
is dispatched. Gate D is NO. Dataset/raw-data access, provenance/adapter work, session-atomic CL
manifests, cache, real-model preflight, T0, model, CUDA/GPU, training, evaluation, final-test unsealing
and performance claims remain stopped pending a new forward-only candidate and fresh exact review.

## RC4 forward-only source/CPU authorization

- immutable rejected base: exact RC3 `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`, parent
  `d527274d368b6d6def5809b002324b31e88ae1bd`, tree
  `3ec47127a6d7c85326bbcf1d72ef76e2971f100d`
- authorized destination: new branch `codex/iclr2027-drive-opd-07g-p0-rc4`, suggested isolated
  worktree `/home/khwang/domain-selector-p0-rc4`, with exactly one non-merge commit whose sole parent
  is exact RC3; all existing refs remain immutable
- P0-A: bind seed design to the full ordered two-comparison/fourteen-hypothesis family and complete
  pilot structure; replay the downstream familywise procedure or a checked exact surrogate; derive
  resolution from the bootstrap p-value floor and Holm threshold instead of the rejected anonymous
  seven-metric `2^(1-n)` proxy
- P0-B: require exact positive integer byte fields, finite positive real wall time and non-empty typed
  host/device identities without Boolean/string/fractional/non-finite coercion at every producer,
  load, inventory, comparison and global-Holm boundary
- verification: preserve all nineteen RC3 attack families, add complete end-to-end adversarial cases
  for both P0s, retain ALER audit equivalence coverage, and run only static/synthetic/CUDA-hidden CPU
  checks before one exact identity/hash receipt
- authorization files: `07g_p0_rc4_forward_authorization.md` and
  `07g_p0_rc4_forward_authorization.json`

Decision: **AUTHORIZED FOR RC4 SOURCE/CPU REPAIR ONLY.** RC4 publication, transfer freeze, review and
06G dispatch are not authorized here. Before any 06G source replay, both a newly created exact-RC4
`gpt-5.6-sol`/`xhigh` review with no open P0 and controller-frozen dataset-specific session-atomic
manifests for NAVSIM and nuScenes must pass. Cache, real-model preflight, T0, CUDA/GPU, training,
evaluation, final-test unsealing and performance claims remain stopped behind later explicit gates.

## RC4 source/CPU candidate acceptance

- exact candidate: `codex/iclr2027-drive-opd-07g-p0-rc4@653e797a2381cdf017466ef95aefcc41961f6715`,
  sole parent exact RC3 `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`, tree
  `53943615137849d3338cc037914b302d6e3db363`; exactly one non-merge forward commit and 22 changed
  paths; research worktree clean and no remote-containing ref
- exact committed-source detached replay: six suites passed 76 tests plus 132 subtests in 311.85
  seconds; output SHA256 `9dbeb7dac11464e54739a41a0e753119dd595c760494b33a55bf5b2b6d9fa7d3`
- expanded schema-v3 attack harness: 21/21 families passed; 146/146 mutated invocations exited
  nonzero and produced no claim artifact; positive complete-family fixture exited zero; receipt
  SHA256 `c869a4e02a6563fbbfeb0842384d5d0ae296eb185a21332106ae45ab5e0f40fa`
- controller verification independently reproduced the exact commit/parent/tree, one-commit ancestry,
  22-path inventory SHA256 `4ec72aa5e835643746a6f2619d93cdce59cb4e84336ec4f75131393454834b32`,
  path/content inventory SHA256 `c083471c5514be3b027a5f6a6feef66acf85cbc34d1f6a5e43dc1381619744ba`,
  and binary-safe full-index diff SHA256
  `6d1b82b0e0871ac4506987bf56e74ad5505141f17f12c2b6cea47e57f2d53193`
- static boundary: `git show --check` passed, 11 changed Python files parsed, three changed JSON files
  parsed with non-finite constants rejected, and no bytecode, pytest cache, data, model or checkpoint
  artifacts were found
- RC3 P0-20 is closed only as a candidate: seed design is bound to the ordered two-comparison,
  fourteen-hypothesis family and replays the production conditional crossed-bootstrap plus Holm
  procedure; RC3 P0-21 is closed only as a candidate through strict non-coercive finite resource
  validation at all claim boundaries
- complete intake: `07g_p0_rc4_acceptance.json`

Decision: **ACCEPTED FOR SOURCE PUBLICATION AND A FRESH EXACT-RC4 REVIEW ONLY.** This is not a P0
promotion. Before any 06G source replay, a newly created context-free `gpt-5.6-sol`/`xhigh` review
must return zero open P0 and the controller must separately freeze session-atomic dataset manifests
for both NAVSIM and nuScenes, followed by an explicit source-replay gate. Cache, real-model preflight,
T0, CUDA/GPU, training, evaluation, final-test unsealing and performance claims remain stopped.

### RC4 deterministic source-transfer freeze

- the clean accepted RC4 worktree produced a 7,434-byte manifest containing exactly 41 required
  source/config/test files; manifest SHA256
  `7ee2fc57e8f3ea9420b0a18fafa7f8c7e9deae34fa45799c3367a1f3f97ad4e9`
- every declared source hash matched the exact `653e797a2381cdf017466ef95aefcc41961f6715`
  bytes with no missing or mismatched files; the independently recomputed canonical logical bundle
  SHA256 `e9aa55e56dabe55064f8a06523bf08aea468a7315020190926668b35cc18913b`
  matched the manifest
- generator `selector_bench/scripts/51_freeze_drive_opd_06g_transfer.py` SHA256
  `49ba27be6e553d1d8d4b670a2b16ec5ec0d356b1b8019462b71f86f1c43fa7f8`; source worktree clean
  before and after generation
- freeze receipt: `07g_p0_rc4_transfer_freeze.txt`; manifest:
  `07g_p0_rc4_transfer_manifest.json`

Decision: **SOURCE BUNDLE FROZEN FOR PUBLICATION AND EXACT-RC4 REVIEW ONLY.** No 06G replay is
dispatched. A zero-P0 exact-RC4 xhigh review, separately frozen session-atomic NAVSIM and nuScenes
manifests, and a later explicit controller gate are all mandatory before source replay. All later
execution gates remain stopped.

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

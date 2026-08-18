# Dual-server Duty/Goal Contract Audit

## Decision

**PASS, limited to server responsibility and work-goal documentation.**

This audit does not pass or authorize a source handoff, NAVSIM or nuScenes cache, continual-learning
stage manifest, T0, real-model preflight, GPU training, evaluation, or paper-performance claim.

## nuScenes session-manifest inventory intake

- source task `01a00dab-5dc6-7413-8ca7-930353db0e0c` inspected exact RC4
  `653e797a2381cdf017466ef95aefcc41961f6715` read-only and reported the committed prototype and
  dataset-evidence gap surface
- the exact 10,562-character delegated inventory hashes to
  `de6204c1b729872b006b7796aac27cf6bf820a4beae73970e0678574989ab385`; the controller independently
  matched all 18 advertised RC4 file hashes from the exact commit
- atomicity is not ambiguous in the controller freeze: nuScenes uses one complete `log_token`, with
  every child scene/sample/sample-data record inheriting one partition, stage and cell; scene-level
  assignment and fallback sorted-scene 80/20 splitting remain forbidden
- existing scene-level selection, converter, loader and NAVSIM claim scaffolds are prototype substrate
  only; no all-table provenance, complete-log ledger, exact memberships, info-PKL lineage,
  nuScenes evaluator/registries/family or independent replay exists
- detailed intake and program evidence matrix:
  `07g_nuscenes_session_manifest_intake_inventory.{json,md}`

Decision: **ACCEPTED AS A READ-ONLY GAP INVENTORY ONLY.** No dataset evidence or producer is accepted,
no manifest may be generated, and no source-replay or execution gate is promoted. Exact-RC4 xhigh
review, complete NAVSIM membership and complete nuScenes membership remain independent blockers.

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

### Exact RC4 source publication

- public commit `f93e53868b322b391b8a43ce4ff96ffae823f3cb`, sole parent
  `b1e1991afe7b9b14fd83f7510394c0ffb77ad700`, tree
  `13c5e731c9f6b2dd2ff4df2f31aaa2acf509b1c4`; exactly 18 changed paths and one non-force
  fast-forward update
- preserved local publication object `d23b0943f19925202b005752857a5d14a15eda63` has the same sole
  parent and exact same tree; no local publication ref was moved
- ordinary anonymous HTTPS replay reproduced commit, parent, tree, exact path inventory and
  `git show --check`; the 7,434-byte manifest SHA256 is
  `7ee2fc57e8f3ea9420b0a18fafa7f8c7e9deae34fa45799c3367a1f3f97ad4e9`
- all 41 source blob SHA256 values matched with no missing or mismatched files; independently
  recomputed logical bundle `e9aa55e56dabe55064f8a06523bf08aea468a7315020190926668b35cc18913b`
  exactly matched the manifest
- the publication tree differs by design from the whole RC4 worktree tree because the publication
  branch has a different parent/controller-receipt layout; exactness is bound at source blobs,
  logical bundle and source commit identity
- receipt: `07g_p0_rc4_publication.txt`

Decision: **PUBLICATION PASS FOR EXACT RC4 SOURCE AND RECEIPTS ONLY.** A fresh context-free exact-RC4
`gpt-5.6-sol`/`xhigh` source/CPU blind review may now be created. No 06G replay is dispatched. Before
any 06G source replay, the fresh review must return zero open P0, distinct session-atomic NAVSIM and
nuScenes manifests must be controller-frozen, and a later explicit source-replay gate must be issued.
Every data, adapter, cache, model, T0, GPU, training, evaluation, final-test and performance gate
remains stopped.

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

## Dataset-specific session-atomic manifest contract freeze

- contract directory: `controller_protocols/drive_opd_session_atomic_v1`; the Markdown contract,
  common Draft 2020-12 shape schema and separate NAVSIM/nuScenes requirements are bound by
  `dataset_session_atomic_contract_freeze.json`, SHA256
  `736823243bbc798bc050dfa4d5a251a316106ca859916b8ea6131f35282c4e68`
- NAVSIM binds the population receipt SHA256
  `fa2f8c417957c22d3fce83d36544c373353a533302deeb5b9a618d9db4aedb1a` separately from the
  session-integrity receipt SHA256
  `7156370a742df6328b03350f48c7b55a1608e9d0498dbd131f17ac075a2e63bd` and canonical session
  table SHA256 `cf12df3e215874362c67a6ce404a8bb8ae054fdc5c393f7a58385abd2453fa08`
- nuScenes freezes `complete_nuscenes_log_token` as the assignment unit; every child scene, sample
  and sample-data record must inherit one partition and stage. Scene-level selection and the legacy
  sorted 80/20 fallback are forbidden
- both datasets require separate canonical population ledgers and separate Chronological-CL and
  Failure-Patch-CL assignment artifacts, exact-set and descendant-closure validation, development-
  only selection inputs, producer/validator source hashes and an independent destination replay
- all three JSON files parse and `git diff --check` passes. The optional `jsonschema` package is not
  installed in this controller environment, so no library replay is claimed; later acceptance must
  run both schema and dataset-specific semantic validators independently

Decision: **CONTRACT SHAPE FROZEN; BOTH MEMBERSHIP MANIFESTS REMAIN BLOCKED.** NAVSIM still lacks
controller intake of the exact session ledger and frozen stage assignments. nuScenes has no accepted
population/provenance ledger and additionally lacks all-table closure, sensor/map/CAN inventory and
DiffusionDrive materialization lineage. The pending exact-RC4 context-free xhigh review must return
zero open P0, both complete manifests must later be accepted separately, and a distinct controller
gate is still mandatory before any 06G source replay. Dataset access, cache, model, T0, CUDA/GPU,
training, evaluation, final-test and performance gates remain stopped.

## Dataset manifest row semantics and fail-closed validator freeze

- this forward-only layer closes an ambiguity in the initial shape freeze: each protocol must assign
  exactly the official **development** atomic-unit set across three stages and exactly one
  development `train`, `audit` or `test` cell, while every official sealed-evaluation unit remains
  absent. The development `test` cell is not the official final partition
- population, assignment and selection-input row schemas now bind canonical JSONL bytes. NAVSIM
  atomic rows include complete session descendants at log, scene-token and token granularity;
  nuScenes retains complete `log_token` ownership of all scene/sample/sample-data descendants
- the standard-library semantic validator rejects duplicate keys, non-finite values, noncanonical
  JSONL, unsafe paths, duplicate descendants, incomplete or sealed assignments, handwritten stage or
  cell substitutions, selection-payload substitution and inconsistent aggregate counts/hashes
- official-population expectations are no longer self-consistency checks: the validator requires
  exact development/sealed/full atomic-unit counts and per-kind descendant counts, then independently
  replays controller-approved per-partition descendant-ID hashes. NAVSIM binds the accepted sorted
  token hashes for 103,288 total, 85,109 development and 18,179 sealed tokens
- neither the population receipt nor executable source may self-authorize: receipt bytes must equal
  `accepted_prior_evidence.population_receipt_sha256`; producer/validator bytes are re-hashed from an
  explicit source root and must exactly match controller-approved commit/tree/path/hash identities
- validation receipts are created only after all checks, with exclusive creation, `fsync` and a
  post-write byte replay; a failed validation leaves no output. Sixteen synthetic CPU tests cover the
  positive contract and principal adversarial substitutions; output SHA256
  `40fc3201f5294162d3b16cba85b2db408a2f1e55616d0773478aad830a88fd18`
- receipt: `dataset_session_atomic_validator_freeze.json`

Decision: **SEMANTIC VALIDATOR FROZEN; BOTH DATASET MANIFESTS STILL DELIBERATELY BLOCKED.** The
current requirements mark both dataset producers unapproved. nuScenes additionally has no accepted
population receipt, exact official expectations, or frozen failure-patch statistics. Those blockers
may only be replaced by later reviewed forward commits. No 06G replay is dispatched, and dataset/raw
access, cache, model, T0, CUDA/GPU, training, evaluation, final-test and performance gates remain
stopped.

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

## 07G nuScenes manifest-producer source-only authorization

- authorization ID: `P001-DRIVE-OPD-07G-NUSCENES-MANIFEST-SOURCE-V1`
- immutable base: RC4 `653e797a2381cdf017466ef95aefcc41961f6715`, parent
  `79a2c529f35fb18049b4f30b46a2fe9926bf7e5a`, tree
  `53943615137849d3338cc037914b302d6e3db363`
- authorized branch: `codex/iclr2027-drive-opd-07g-nuscenes-manifest-source-v1`
- required history: one non-merge child of exact RC4
- normative semantic contract: controller commit
  `04b2290979cb711ada763967ad55a62586f7d7be`, tree
  `131f601bccf9c7846346c341323a471d3189a624`; seven named contract files must
  be copied byte-identically and the production requirements must remain blocked
- required source surface: nuScenes protocol module, thin build CLI and synthetic adversarial tests
- frozen atomicity: complete `log_token`; every child scene/sample/sample-data record inherits one
  official partition, stage and development cell
- required source checks: official split identity with fallback disabled, all-table foreign-key and
  sample-chain closure, metadata/sensor/sweep/map/CAN provenance inventory, canonical ledgers,
  exact chronological/cell replay, blocked Failure-Patch behavior, source/input/output lineage and
  validate-before-atomic-publish output semantics
- boundary: synthetic source/CPU only; no real metadata or sensor access, membership generation,
  info PKL, anchor, cache, model, final test, CUDA/GPU, training, evaluation, publication, 06G
  dispatch or performance claim
- authorization files: `07g_nuscenes_manifest_source_v1_authorization.md` SHA256
  `3d57ed553aabefaf195806dea6f12b38429b65062aa732614e295e4b57484061` and
  `07g_nuscenes_manifest_source_v1_authorization.json` SHA256
  `2e666b9276c17763672203caf6a9909eae8029d98774b9ebf73bf4b6520fd121`

The candidate cannot self-approve. Controller review, an approved-producer freeze, a separately
authorized metadata-only N0 run, authoritative population expectations and real manifest generation
remain later gates. All dataset and execution gates remain stopped.

## nuScenes session-manifest source v1 controller audit

- exact candidate: `37234dc194242a31459b0210bfbbd220c4d1a1ee`, sole parent
  `653e797a2381cdf017466ef95aefcc41961f6715`, tree
  `c1d83c19ef2e08167b0014ac48c4828a95bb5651`; one non-merge commit, 11 paths, clean worktree
- exact controller-contract copy: PASS 7/7
- independent focused replay: 44 passed in 7.85 seconds, output SHA256
  `c4c40cd882547a6c315a144b7821915653427abce9f2c5714257af2518c5bfd1`
- adjacent regressions: 10 passed in 0.12 seconds, output SHA256
  `6df23f5e7aa2f42be8c82e63634e296d396d4d919f2ef69394b6bb2129841712`
- executable bypass: arbitrary hand-authored Failure-Patch statistics plus an internally matching
  `pre_access=true` receipt reached a controller-validator-passing published assignment; output
  SHA256 `9a4da0777d7f3063af348d1eee37d814484bd3b50289de9e15494d516f501810`
- open P0: neither the requirements nor accepted-prior-evidence surface freezes the Failure-Patch
  stream/receipt identity or authoritative access chronology; the receipt therefore self-attests the
  property it is meant to prove
- P1: the paper formula describes selected camera/LiDAR records while the ledger includes all
  sample-data descendants; authoritative split and CAN provenance must be bound in the later N0
  freeze

Decision: **SOURCE V1 REJECTED / P0 OPEN.** No producer approval, N0 dataset access, real manifest,
publication, 06G replay, cache, model, T0, CUDA/GPU, training, evaluation, final-test access or claim
is authorized. A repair requires a separate forward-only authorization after the controller freezes
the missing evidence-binding semantics.

## nuScenes manifest source v2 forward-only repair authorization

- rejected base: source v1 `37234dc194242a31459b0210bfbbd220c4d1a1ee`, tree
  `c1d83c19ef2e08167b0014ac48c4828a95bb5651`
- hardened controller contract: `fd169e509af64da942ba9c63fc1e0e8ff5e584e9`, tree
  `3ebe4edbc1072e19c1d666cadd89c426357f343e`
- new semantics: exact controller-frozen canonical selection-input hash; raw stream/receipt,
  family/access-ledger/evaluator/registry identities; strict monotone UTC chronology; full evidence
  object in the selection-parameter digest
- controller contract tests: 19 passed; output SHA256
  `2e2e53f3dec87894284f664428a206c63548bf595b01198702e91eb260201273`
- authorized destination: one non-merge child on
  `codex/iclr2027-drive-opd-07g-nuscenes-manifest-source-v2`
- boundary: synthetic source/CPU repair only; production requirements remain blocked

Decision: **AUTHORIZED FOR SOURCE V2 SYNTHETIC SOURCE/CPU REPAIR ONLY.** No producer approval,
dataset/raw metadata access, real manifest, publication, 06G replay, cache, model, T0, CUDA/GPU,
training, evaluation, final-test access or claim is authorized.

## nuScenes manifest source v2 controller audit

- exact candidate: `495bac670073fc3f46f0f773ca51033331078509`, sole parent
  `37234dc194242a31459b0210bfbbd220c4d1a1ee`, tree
  `a7c86fbafea77ec647197c6d24d65edff83c95a8`; one non-merge commit, seven paths, clean worktree
- exact hardened controller-contract copy: PASS 7/7
- independent v1-bypass regression: 1 passed in 0.72 seconds, output SHA256
  `6262dd4ae1e1610248aa7a1d90c86e6b7861110a1a434aeede186690d3652770`
- detached source-v2 suite: 71 passed in 12.12 seconds, output SHA256
  `9092d793d1fab5629e5e0b4de4ec9a9e0580f8675a9fc8a2f47a1c47e744e957`
- controller-validator suite: 19 passed in 0.50 seconds, output SHA256
  `815a1a0dc1c9272190561f6057a08651e12ab643edcc13279562a4a4390bfbc9`
- the rejected self-attestation path is closed on the synthetic source surface: raw evidence is
  content-bound, canonical selection input is independently frozen before stage calculation, and
  complete-log descendants include all sample-data records including radar
- external authority remains separate: the controller must authenticate the real family/access/
  evaluator objects and chronology, then freeze real evidence in a later reviewed commit

Decision: **PASS FOR A LATER APPROVED-PRODUCER IDENTITY FREEZE ONLY.** This audit does not itself
approve the producer or unlock N0. Authoritative Failure-Patch evidence, official nuScenes
provenance/expectations, real population and membership generation, publication/transfer, 06G
replay, cache, model, T0, CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## nuScenes producer-v2 identity freeze

- source-v2 audit parent: controller commit `145d6504ed09457e0cd094fad41341bd4a07306c`
- approved source identity: commit `495bac670073fc3f46f0f773ca51033331078509`, tree
  `a7c86fbafea77ec647197c6d24d65edff83c95a8`
- producer source SHA256: `a5e509d9fb33e5bbc00bdea71ecc51521eeeee39335f18e8667bea124ac2d773`
- validator source SHA256: `70b53893c3bc0704fbf09945a677412effdf4a2b0258ac56c2f8084798805ba3`
- updated nuScenes requirements SHA256:
  `08968b1af352210fb57bfdc7db0946f4f75191ba865b2889e476fb5fc8409507`
- controller contract replay after the freeze: 19 passed in 0.45 seconds, output SHA256
  `82f2132e84555fb9e8a16514fde4f5f090399de8c5b7f6812c1edf54709f30e6`

Decision: **PRODUCER IDENTITY FROZEN; MEMBERSHIP STILL BLOCKED.** No accepted population receipt,
official population counts/hashes, Failure-Patch parameters or authoritative Failure-Patch evidence
exists. Metadata-only N0 access itself requires a separate authorization. RC4 review, both complete
manifests, source publication/destination replay and the explicit 06G source-replay gate remain
pending. Raw data, real manifests, cache, model, T0, CUDA/GPU, training, evaluation, final test and
claims remain stopped.

## NAVSIM manifest source inventory and forward-only source authorization

- accepted evidence boundary: population receipt `fa2f8c417957c22d3fce83d36544c373353a533302deeb5b9a618d9db4aedb1a`,
  session-integrity receipt `7156370a742df6328b03350f48c7b55a1608e9d0498dbd131f17ac075a2e63bd`
  and 162-row session ledger `cf12df3e215874362c67a6ce404a8bb8ae054fdc5c393f7a58385abd2453fa08`
- exact accepted counts: 162 full sessions / 1,192 logs / 103,288 tokens; 101 development sessions /
  978 logs / 85,109 tokens; 61 sealed sessions / 214 logs / 18,179 tokens
- legacy exact-RC4 protocol source is rejected as a producer: it derives membership from a pickle
  cache index, retains a historical cache default, loads Failure-Patch scores from a generic CSV and
  does not emit the common controller-frozen artifacts or bind controller-authorized evidence
- R1 conversion guard: `session_key` is the atomic unit; logs/tokens must be exact descendants and
  chronology must be recomputed. The R1 ledger lacks an independent `scene_token` field, so a
  candidate must prove segmented-log/scene identity from accepted dataset evidence or consume an
  independent exact mapping; silent aliasing is forbidden
- authorization ID: `P001-DRIVE-OPD-07G-NAVSIM-MANIFEST-SOURCE-V1`
- immutable base: RC4 `653e797a2381cdf017466ef95aefcc41961f6715`; destination
  `codex/iclr2027-drive-opd-07g-navsim-manifest-source-v1`; exactly one non-merge child
- controller contract: commit `2c0694213b1db0289dd3fec9094d3f1531ee9a23`, tree
  `f9a929e5530b342dd3779ac1bea63b8c0f4e39b2`; seven files copied byte-identically while production
  producer and Failure-Patch evidence stay blocked
- audit JSON SHA256 `f0ffa2a220b3fbbace2aeb544bc53c4564e64414984b77cd1aee21ccbccf820e`;
  audit Markdown SHA256 `35387263785b701eea7ddd0bc225bd392c67ab65bf348073da71028dd33bdc28`
- authorization JSON SHA256 `b422dbb0800562d01426cc6d483d1adfca7de0f4dd619aa939a69a365cec1321`;
  authorization Markdown SHA256 `95d464505d7b414f3f38e5242f9e15cd9d714e337ceb21c20811bff231031b05`

Decision: **AUTHORIZED FOR ONE SYNTHETIC SOURCE/CPU CANDIDATE ONLY.** No real R1 evidence or dataset
read, manifest generation, producer approval, publication, transfer or 06G dispatch is authorized.
A valid exact-RC4 xhigh zero-P0 review, both independently accepted dataset manifests and a later
explicit controller gate are still required before any 06G source replay. Cache, model, T0,
CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## NAVSIM manifest source v1 candidate controller audit

- exact candidate: `64447d3a9cc60f61b8596aed41086b5b4311c889`, sole parent
  `653e797a2381cdf017466ef95aefcc41961f6715`, tree
  `5ce81ef06d71189d1ff46676ecc1b4d01e415728`; one non-merge commit, 12 paths, clean worktree
- exact controller-contract copy: PASS 7/7
- independent producer suite: 44 passed in 4.64 seconds, stdout SHA256
  `ab022635f1110d3ff024e7942a0a37f47789bfc432a943d9d2571c327175c9cf`
- controller-validator suite: 19 passed in 0.56 seconds, stdout SHA256
  `d0654e6b420aef5ffb33b81c839cd73cd87307576aa967d1be275e3ca8fe8bce`
- independent full repository replay: 143 tests and 132 subtests passed in 304.57 seconds
- historical cache/pickle/CSV surfaces reject; session descendants, sealed partition, exact scene
  identity authority, Failure-Patch authority and staged output sealing all fail closed

Decision: **PASS FOR A SEPARATE APPROVED-PRODUCER IDENTITY FREEZE ONLY.** This audit does not approve
the producer or generate a real manifest. Scene identity, authoritative Failure-Patch evidence, real
R1 input transfer, complete manifest validation/destination replay and exact-RC4 xhigh zero-P0 review
remain pending. 06G source replay, cache, model, T0, CUDA/GPU, training, evaluation, final-test
access and claims remain stopped.

## NAVSIM producer-v1 identity freeze

- source audit parent: controller commit `00d3853e11105e0b5c4653e62482ae5c29f3c2c6`
- approved source identity: commit `64447d3a9cc60f61b8596aed41086b5b4311c889`, tree
  `5ce81ef06d71189d1ff46676ecc1b4d01e415728`
- producer source SHA256: `75f683cb8a7931783f1342a9b479d37470bc215b297efc421b003e3a2ab6cfae`
- validator source SHA256: `70b53893c3bc0704fbf09945a677412effdf4a2b0258ac56c2f8084798805ba3`
- updated NAVSIM requirements SHA256:
  `54d3f5f203a3db568c5cc5d18305c3dc4b00b81863e353c9dbd30bdefc833d64`
- controller-validator replay after freeze: 19 passed in 0.40 seconds, stdout SHA256
  `b84a5f56832006ce4e2bf9f759666949900308668a781b362e5aab41c6c2b8e8`

Decision: **PRODUCER IDENTITY FROZEN; REAL MANIFEST STILL BLOCKED.** Controller-authorized scene
identity and Failure-Patch evidence are absent; real R1 inputs are not transferred and no real
membership exists. The exact-RC4 xhigh zero-P0 review, both complete manifests and explicit 06G gate
remain pending. Cache, model, T0, CUDA/GPU, training, evaluation, final-test access and claims remain
stopped.

## NAVSIM token-level scene-mapping source-v2 authorization

The official-source semantic audit closes the alias option and also exposes an overconstraint in the
approved v1 producer: one row per segmented log plus unique scene tokens imposes an unsupported
log-to-scene bijection. The controller authorizes one forward-only source/CPU child of exact
`64447d3a9cc60f61b8596aed41086b5b4311c889` to replace that surface with one canonical row per
accepted R1 token. Exact ledger coverage and log/session/partition membership are mandatory;
multiple empirical log/scene relationships are allowed only within one complete session, and scene
tokens may never cross sessions or official partitions.

This is source authorization only. It does not authorize real metadata access, mapping or manifest
generation, count updates, publication, 06G dispatch, cache/model/T0/CUDA/GPU/training/evaluation,
final-test access or claims. A separate source audit/freeze and later real-evidence authorization are
required.

## nuScenes N0 provenance source-v1 authorization

- controller parent: `702d50280a69c98951c985b9935f460a4aea4e87`, tree
  `bebfd54dece7a87caf01d494ca8e22c8c09335da`
- exact forward base: nuScenes manifest producer-v2 commit
  `495bac670073fc3f46f0f773ca51033331078509`, tree
  `a7c86fbafea77ec647197c6d24d65edff83c95a8`
- authorized branch: `codex/iclr2027-drive-opd-07g-nuscenes-n0-provenance-source-v1`
- changed-path budget: four exact source/test/paper paths
- phase boundary: source and synthetic CPU fixtures only

Decision: **FORWARD-ONLY N0 PROVENANCE SOURCE AUTHORIZED; REAL DATA ACCESS STOPPED.** The candidate
must provide explicit content-addressed release/devkit/table/split/map/CAN inputs, all-table
foreign-key and reciprocal-chain closure, one complete-log population ledger, streaming registered
sensor/map/CAN inventories, deterministic semantic replay and atomic no-replace publication. It must
not inspect the real nuScenes or NAVSIM installations in this phase. A later controller audit and a
separate authorization are required before any real N0 scan. Real manifest generation, 06G source
replay, cache, model, T0, CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## nuScenes N0 provenance source-v1 candidate controller audit

- exact candidate: `ac38c9ffca9d4a428cee97606a92f93fb5ff8d82`, sole parent
  `495bac670073fc3f46f0f773ca51033331078509`, tree
  `07900590456acb6e523f7d815f969dccc11ae226`; one non-merge commit, four authorized paths, clean
- source executor: focused provenance 54/54, adjacent nuScenes manifest 71/71, controller contract
  19/19, and full repository 227 tests plus 132 subtests passed
- independent detached-Git controller replay: 132/132 passed in 71.65 seconds, output SHA256
  `559b07f6481d9b859cbbb1fd329f62d7608809dad2947977fd586056526588a5`
- source identity binds isolated Git state, one origin, exact tracked HEAD blobs, frozen manifest
  producer ancestry/blob, descriptor-safe input reads, strict CLI inputs and private staged output
  semantic replay with atomic no-replace publication

Decision: **SOURCE IDENTITY FROZEN; REAL N0 SCAN STILL BLOCKED.** A later separate controller
authorization must freeze a content-addressed real input specification before any nuScenes access.
Real N0 evidence, session-atomic manifest generation, publication/transfer, 06G source replay,
cache, model, T0, CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## nuScenes N0 real-root identity intake

- exact accepted source inspected read-only: `ac38c9ffca9d4a428cee97606a92f93fb5ff8d82`, tree
  `07900590456acb6e523f7d815f969dccc11ae226`; source worktree remained clean
- canonical dataset root: `/home/khwang/datasets/nuscenes/data`; metadata, samples, sweeps, maps and
  CAN roots plus their parent chains are not symlinks
- the DiffusionDrive data path is a symlink alias to the canonical root and is forbidden as an
  evidence root
- installed nuScenes devkit package, `1.1.10` distribution metadata and official split-source
  candidate were identified but not read or accepted as upstream provenance
- neither expected trainval metadata archive exists; the extracted metadata directory has no
  accepted upstream-origin receipt
- samples and sweeps may contain sealed-test files; future inventory must be token-reachable from
  explicitly addressed trainval tables, never directory-wide

Decision: **PASS FOR ROOT IDENTITY INVENTORY ONLY; DATA REMAINS UNREAD.** The canonical roots may be
named in a later controller authorization, but no real input specification or N0 scan is authorized.
A deterministic synthetic/source-only input builder must first be reviewed. Real N0 evidence,
session-atomic manifest generation, publication/transfer, 06G source replay, cache, model, T0,
CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## nuScenes N0 input-builder source-v1 authorization

- controller parent: `9564d3043b3c7aef7d096d375ebaf390bc788698`, tree
  `71f2fd1a641b7db5739237a6d395564ebbd83d3c`
- exact forward base: accepted N0 provenance source commit
  `ac38c9ffca9d4a428cee97606a92f93fb5ff8d82`, tree
  `07900590456acb6e523f7d815f969dccc11ae226`
- authorized branch: `codex/iclr2027-drive-opd-07g-nuscenes-n0-input-builder-source-v1`
- changed-path budget: four exact source/test/paper paths
- phase boundary: source and synthetic CPU fixtures only

Decision: **FORWARD-ONLY INPUT-BUILDER SOURCE AUTHORIZED; REAL DATA ACCESS STOPPED.** The builder
must create the exact explicit input specification consumed by frozen script 60, require independent
upstream release provenance, bind devkit and official splits, derive registered paths only from
trainval-reachable tokens, and forbid root discovery or shared-tree inventory. This phase must not
read or hash real nuScenes/NAVSIM contents or generate a real input specification, N0 output or
manifest. Publication/transfer, 06G source replay, cache, model, T0, CUDA/GPU, training, evaluation,
final-test access and claims remain stopped.

## nuScenes N0 input-builder source-v1 candidate controller audit

- exact candidate: `2734b602e6ed92b6c1111bb36b4a9c02eacd471a`, sole parent
  `ac38c9ffca9d4a428cee97606a92f93fb5ff8d82`, tree
  `cc8780eabb763ee8d05f355e96df55d71ecf4d82`; one non-merge commit, four authorized paths, clean
- source executor: 185/185 tests passed in 126.44 seconds; output SHA256
  `9a6232f44a5aa8307e1fdaf042ac4c74b1f2c4ae9faf601724ef76921ee7ea0f`
- independent controller detached replay: 185/185 passed in 127.01 seconds; output SHA256
  `86ad1b018c4c3370475dce65b5f254e5a1b1fa01c5a74bd967c9ffd604c3f264`
- addressed real-mode authorization binds the canonical request subject; reuse after any other
  request/root/source/address mutation rejects
- no shared-root discovery, recursive walk, model/torch/CUDA import, bytecode/cache residue or
  worktree drift was observed

Decision: **SOURCE IDENTITY FROZEN WITH DOCUMENTATION-SCOPE CORRECTIONS REQUIRED.** The accepted
source supports an extracted metadata directory, not an archive. In real mode it derives explicit
registered-file expectations but does not replace script 60's later complete all-table-closure gate.
Those two paper statements must be narrowed before paper/publication promotion. Independent upstream
origin remains absent and no real input-specification construction is authorized. Real nuScenes/N0,
session-atomic manifest generation, publication/transfer, 06G source replay, cache, model, T0,
CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## nuScenes N0 input-builder documentation-scope v2 authorization

- controller parent: `6fbda507b92815fa0009049f2d6ad1be02df85af`, tree
  `d2017bc7a844394e55839547f561173c1ebd3f1d`
- exact forward base: accepted input-builder source-v1 commit
  `2734b602e6ed92b6c1111bb36b4a9c02eacd471a`, tree
  `cc8780eabb763ee8d05f355e96df55d71ecf4d82`
- authorized branch: `codex/iclr2027-drive-opd-07g-nuscenes-n0-input-builder-doc-scope-v2`
- changed-path budget: the input-builder paper document only

Decision: **FORWARD-ONLY DOCUMENTATION CORRECTION AUTHORIZED; ALL EVIDENCE AND EXECUTION GATES
UNCHANGED.** The child must state extracted-directory-only input and leave complete all-table closure
to frozen script 60. Source/config/test changes, archive implementation, real-data access, real input
construction, paper promotion, publication/transfer and 06G dispatch are not authorized. Cache,
model, T0, CUDA/GPU, training, evaluation, final-test access and claims remain stopped.

## nuScenes upstream-origin human audit request

- canonical extracted metadata root:
  `/home/khwang/datasets/nuscenes/data/v1.0-trainval`
- original metadata archive: absent
- accepted public immutable archive digest: absent
- official distribution path: account- and terms-gated nuScenes download page
- frozen safe route: authenticated human acquisition into a fresh isolated path, credential/signed
  URL redaction, streaming archive hash, safe fresh extraction, then exact thirteen-table comparison
  only under a later controller authorization

Decision: **HUMAN ACQUISITION CHOICE REQUIRED; NO DATA ACTION AUTHORIZED.** The human must approve an
isolated official re-download, provide the original official archive/receipt, or decline real
nuScenes evidence. This request itself permits no login, download, metadata/table read, N0 or
manifest construction. Publication/transfer, 06G source replay, cache, model, T0, CUDA/GPU,
training, evaluation, final-test access and claims remain stopped.

## nuScenes N0 input-builder documentation-scope v2 candidate audit

- exact candidate: `b4c3ffd8b197c6fdd4a7ef7416cc8852714a5717`, sole parent
  `2734b602e6ed92b6c1111bb36b4a9c02eacd471a`, tree
  `068b319424359eb43bb35fc24948ea52df3c5c51`
- exact change: one authorized paper document only; CLI/library/test Git blobs match the base
- independent controller detached replay: 185/185 passed in 127.15 seconds; output SHA256
  `f138d3d94dac3e51e779c3b8da1ac88c0ec91e4e89f68bb3c90032b8857a437e`
- corrected boundary: extracted metadata directory only; real-mode graph derivation is not complete
  all-table closure; synthetic script-60 replay is interface compatibility only

Decision: **DOCUMENTATION CORRECTION PASSED; NO EVIDENCE OR EXECUTION PROMOTION.** Source identity is
unchanged and the two recorded wording discrepancies are closed. Independent upstream origin,
exact-RC4 zero-P0 review, complete frozen session-atomic manifests and a later controller gate remain
mandatory. Real data/N0, publication/transfer, 06G source replay, cache, model, T0, CUDA/GPU,
training, evaluation, final-test access and claims remain stopped.

## NAVSIM scene/log semantic source audit

- official `v1.1` tag: `0811876c274e8b058ab2be9b3dcd4d37bd23f177`
- official `v1.1` branch head: `3e8291bfa89ff247231e0227778840cd0a036896`
- both identities contain the same dataloader and dataclass blobs
- the loader keys filtered scene windows by frame token and separately groups them by `log_name`
- `SceneMetadata` stores `log_name`, `scene_token` and `initial_token` as distinct source fields

Decision: **SILENT LOG-TO-SCENE ALIAS REJECTED; EXACT R1 MAPPING REQUIRED.** The current equal
scene/log counts are unproven because the accepted R1 session ledger contains no independent
`scene_token`. A future intake must bind every accepted token to its exact segmented log and source
scene token, then independently replay unique-set counts, hashes, partitions and session closure.
No frozen producer or requirements byte is changed here. NAVSIM/nuScenes manifests, 06G source
replay and all dataset/cache/model/T0/CUDA/GPU/training/evaluation/final-test/claim gates remain
stopped.

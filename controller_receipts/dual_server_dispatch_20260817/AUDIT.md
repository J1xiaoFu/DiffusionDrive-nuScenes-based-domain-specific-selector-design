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

This is a dispatch/receipt publication snapshot. It deliberately excludes the rejected/unreleased
`9dabf423` claim-protocol source and cannot serve as a source/transfer execution authorization.

The published acceptance contract itself records `FAIL / OPEN P0`. Its reachability PASS only proves
that both servers can audit the same gate bytes; it does not prove the rejected protocol has been fixed.

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

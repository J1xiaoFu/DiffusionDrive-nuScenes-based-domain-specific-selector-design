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

This is a dispatch/receipt publication snapshot. It deliberately excludes the rejected/unreleased
`9dabf423` claim-protocol source and cannot serve as a source/transfer execution authorization.

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

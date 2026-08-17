# 06G R1 Official NAVSIM Navtrain Audit

## Decision

**PASS, limited to official source-population and official train/val partition reconstruction.**

This receipt does not authorize cache generation, T0 training, GPU use, or construction of the
Failure-Patch-CL/Chronological-CL stages. Those gates remain closed.

## Bound evidence

- 06G branch: `codex/iclr2027-diffusiondrive-06g`
- 06G atomic evidence commit: `1bc9cc2957eeccc6927f3dc826135ce85bfae790`
- parent official DiffusionDrive commit: `9b52ed0ec06b073d82d6f392ab084c7b301c8681`
- evidence tree: `2a6cffd88ec048730a7d97a437c1d69ac3a8412a`
- remote receipt SHA256: `fa2f8c417957c22d3fce83d36544c373353a533302deeb5b9a618d9db4aedb1a`
- atomic format-patch SHA256: `8ce8d78ed1d0d32953c7c9ca396caaa57f3a24db4c3a4f16826d7f62ed50c542`
- source-only patch SHA256: `11fc55883aa1db640ebaa5db082bd90c3a67b658cd91c797be4bd8d9aa37b66e`
- remote branch status after commit: clean
- changes below upstream `navsim/`: none
- remote CPU validation: five tests passed (`5 passed in 0.89s`; sealed run `5 passed in 0.98s`)

The controller copy of `receipt.json` is accepted only if its local SHA256 exactly equals the remote
receipt SHA above.

## Reconstructed population

| Quantity | Value |
|---|---:|
| Whitelisted logs | 1,192 |
| Sessions | 162 |
| Official navtrain tokens | 103,288 |
| Official train tokens | 85,109 |
| Official validation tokens | 18,179 |
| Cross-log duplicate tokens | 0 |
| Unassigned selected tokens | 0 |

The full token SHA256 is
`5614a8a32a7030bda7cc71696e085f59116df442ba6c0b2604bb0649fea7f083`, identical to the
independent P001 reconstruction on the controller host. Train and validation token hashes are
`6e625045dc4f0b453c634c1165071b7dd008ac2df0445b8f1cd37c95db400207` and
`2ba748b9f4bcff790aee60d7da889a7503515246797d51e7045aab9eb108024a`.

## Audit interpretation

The agreement on count and full-token hash closes the ambiguity about the official NAVSIM navtrain
source population. It does **not** repair the historical 14,951-token local cache: that cache remains
ineligible because its selection rule is unreproducible.

The 85,109/18,179 split is NAVSIM's official train/validation partition. It must not be described as
the paper's three-stage continual-learning split. Stage membership must be generated later from
session-atomic, prospectively frozen Failure-Patch-CL and Chronological-CL rules.

## Remaining hard gates

1. Publish the controller source/transfer ref accessible to 06G.
2. Freeze session-atomic CL stage manifests without reading final-test outcomes.
3. Execute and independently replay the official cache builder from the frozen source/config.
4. Bind cache index, token-to-log map, raw-tree metadata, environment and builder hashes.
5. Run the real-adapter OPD/LwF query-budget gate.

Until all five pass, P002 remains screening-only and computation remains stopped.

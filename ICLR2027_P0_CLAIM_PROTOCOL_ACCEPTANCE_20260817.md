# Drive-OPD P0 claim-protocol 可执行验收合同

状态：**FROZEN ACCEPTANCE CRITERIA；当前 FAIL；不授权 cache/T0/GPU/training**

冻结日期：2026-08-17

适用范围：07G nuScenes 方法/外部有效性线与 06G NAVSIM 主证据线的共同 claim-protocol。
数据 provenance、cache、adapter、训练和性能门仍由各数据线分别通过。

## 1. 目的与当前边界

本合同把 `9dabf42371e8dde72e1b2061bac4fa9d11b34230` 盲审发现的绕过转成可执行、
fail-closed 的验收条件。通过本合同只证明 run/evaluation/claim 身份链具备进入后续数据门
审计的资格，不证明算法正确、数据可用、cache 合格或性能有效。

公开 dispatch 分支上的职责快照与 receipt 仅用于跨服务器传输；它不是修复后的 source
release。任何执行晋级都必须引用新的完整 source commit、tree、transfer manifest 和独立
盲审 receipt。

## 2. 不可混配的身份图

每条 claim-bearing 证据必须能沿下列有向边逐字节重放：

```text
registered method_id -> training_arm
dataset/version -> protocol -> domain/method/metric registries
source/config/stage-start -> run_id/seed/RNG -> checkpoint
checkpoint -> frozen evaluator run -> CSV/result
result cell -> registered comparison/hypothesis -> claim receipt
claim receipt -> immutable family inventory -> global Holm decision
```

任一边缺失、重复、未知、内容哈希不符或跨 seed/run/arm/domain 混配即拒绝。路径相同不代表
内容相同，路径不同也不代表独立证据；两者都必须检查。

## 3. 最小不可变 receipt schema

### 3.1 Run completion receipt

必须绑定：

- dataset、dataset version、dataset-root metadata SHA256；
- source commit/tree、dirty status、environment lock SHA256；
- config/protocol 路径与内容 SHA256；
- method registry SHA256、registered `method_id`、唯一 `training_arm`；
- run ID、seed、RNG state/record SHA256、checkpoint stage、stage-start state SHA256；
- target 与 observed epochs/steps/optimizer updates；
- current/old unique identities、presentations、forward/backward、student/teacher query counts；
- endpoint checkpoint 与 optimizer-state SHA256、parent checkpoint identity；
- wall time、peak VRAM、persistent/transient stored bytes、host 和 GPU UUID（如使用 GPU）。

### 3.2 Evaluator-run receipt

必须绑定：

- 上游 run receipt 与加载 checkpoint SHA256；
- evaluator source commit/tree、config 和 environment SHA256；
- dataset/protocol/domain registry SHA256；
- distinct checkpoint stage、evaluation stage、registered domain 与 split；
- exact token/sample/scene-set manifests 与内容 SHA256；
- CSV/result/summary 的内容 SHA256、row count、invalid/failure count 与 metric schema SHA256；
- completion status。该 receipt 是 checkpoint 产生该 CSV 的唯一合法语义边。

### 3.3 Claim inventory

Git 中冻结的 family inventory 必须显式列出所有 expected comparisons、semantic coordinates、
receipts、run/results、checkpoints 和 CSV 的内容哈希。发现过程不得依赖文件名 suffix；注册根
下的 unexpected、missing、重复内容、重复 semantic coordinate、未注册 family 一律拒绝。

## 4. 必须通过的攻击性 CLI 回归

修复候选必须提供可独立运行的命令、fixtures、期望 error code 与终端输出 SHA256。以下每项
都必须在 mutation 后稳定拒绝，并证明拒绝发生在 claim decision 之前：

1. `P0-01 seed-reuse`：两个 nominal seeds 共用 run/result/CSV/checkpoint/RNG 证据。
2. `P0-02 csv-swap`：candidate checkpoint 配 baseline CSV/result/summary。
3. `P0-03 arm-mismatch`：`method_id` 与 registry 中的 `training_arm` 不一致。
4. `P0-04 unknown-domain`：任意未注册 evaluation domain/stage/split。
5. `P0-05 suffix-hide`：合法 schema receipt 改名或换 suffix 后逃出 inventory。
6. `P0-06 duplicate-receipt`：byte-identical receipt 以不同路径计为多份证据。
7. `P0-07 duplicate-raw-evidence`：不同 receipt 重用相同 run/result/checkpoint/CSV 内容。
8. `P0-08 missing-cross-cell`：缺少非对角 checkpoint-stage × evaluation-domain 单元。
9. `P0-09 row-integrity`：CSV 出现 duplicate/extra/missing/invalid rows 或 token set 不符。
10. `P0-10 mutable-protocol`：config/protocol/registry 内容改变但路径或声明 SHA 未同步。
11. `P0-11 incomplete-run`：completed steps/updates/query counts 未达到冻结 target。
12. `P0-12 hidden-extra-claim`：注册根内存在 family inventory 未列出的 claim-eligible object。

另需一个完整 family 的 positive fixture：任意调整文件名和遍历顺序仍得到同一 PASS 与 Holm
决定，以证明 validator 检查的是内容和语义而非偶然路径。

## 5. 运行时与统计门

- query count 必须来自真实 adapter/query path，并在 optimizer step 前与预算核对；bookkeeping
  不得在热路径逐 query 调用 CUDA `.cpu()`/`.item()` 同步。
- 提交 audit on/off 的相同 batch loss、gradient 与 optimizer-update 等价性测试；LwF 与 OPD
  identical-state negative control 必须在相同数据和预算下通过。
- 在允许 GPU 前，先以 CPU fixture 验证语义；真实 batch 的计数开销、loss/gradient、resume
  receipt 属于后续 dataset-specific adapter gate，不由本合同提前授权。
- confirmatory seed 数必须由前瞻 power/precision receipt 决定，不能仅写死一个整数；所有
  comparisons 必须在结果产生前进入 immutable family，并统一执行 global Holm。
- chronology 必须由可执行 scene/session ordering、commit/time receipt 与 final-test access
  ledger 证明。final test 保持封存，session × seed 使用 crossed paired analysis。

## 6. 数据线隔离

NAVSIM 与 nuScenes 使用不同 dataset manifests、caches、checkpoints、metric registries、family
inventories 和结果表。任一线通过或失败都不能替另一线解锁；nuScenes 结果不能改变 NAVSIM
split/hypothesis，NAVSIM 结果也不能事后选择 nuScenes split 或方法。

## 7. 冻结、转移与独立复审 receipt

候选只有同时提供以下完整字段，controller 才接受复审：

- source branch、完整 commit、parent、tree、clean status 与精确 changed-path set；
- test command、pass count、完整输出 SHA256、12 项 attack manifest 及每项 fixture/error code；
- method/domain/metric/family registry 路径与 SHA256；
- 一份 valid receipt chain 与至少 12 份 rejected mutation receipts；
- fresh context-free blind-review task ID、reviewed full SHA、decision、review bytes SHA256；
- transfer manifest path/SHA256、bundle SHA256、文件数、每文件 path/content SHA256；
- 可由 06G 普通 HTTPS fetch 的 repository、branch 与完整 publication commit。

仅 source 测试自报 PASS、prefix SHA、task status、review prose 或 transfer bundle 单独存在均不
足以晋级。06G fetch 后还必须独立重算 Git object、manifest、bundle 与文件内容哈希。

## 8. 当前决定

当前决定为 **FAIL / OPEN P0**。07G 可继续源码修复、CPU 攻击测试、nuScenes 只读 provenance
与 scene-atomic split 设计；06G 可继续只读官方 source/manifest 设计与 receipt 审计。两台机器
的 claim-bearing cache、真实模型 preflight、T0、CL training/evaluation 和 GPU gate 均保持
STOPPED，直至新的完整 SHA/receipt 被 controller 独立验证并逐数据线明确晋级。

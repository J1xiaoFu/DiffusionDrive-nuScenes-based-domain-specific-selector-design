# ICLR 2027 Drive-OPD：当前服务器目标与执行规划

状态：ACTIVE

责任服务器：当前服务器（`/home/khwang/domain-selector`）

责任任务：`ICLR 2027｜Drive-OPD 方法与论文总控`

工作分支：`codex/iclr2027-drive-opd-controller`

计划冻结日：2026-08-17

## 1. 截止日与唯一总目标

ICLR 2027 官方摘要截止为 **2026-09-11 AOE**，全文截止为
**2026-09-16 AOE**。用户此前估计的 2026-09-25 不再作为提交截止日。

唯一总目标是在官方截止日前形成一篇证据边界清楚、可复现并达到投稿质量的
智驾原生持续学习论文：定义真实日志增量协议，验证 Drive-OPD 是否在学习新能力的
同时保持旧能力，并用因果审计说明学生分布指导与感知漂移处理何时必要。

当前服务器不承担 pristine 官方 DiffusionDrive 的大规模训练；它负责方法、协议、
统计、证据审核、论文和双机总控。06G 权重不得复制到本机，只接收配置、commit、
SHA256、JSON、CSV 和曲线。

## 2. 当前已知证据与待完成点

- 本机 RAP-modified NAVSIM 可行性训练已完成 Stage-2 的四臂：顺序训练、LwF、固定
  教师 OPD、`m=0.99` EMA-OPD；每臂 30 epoch、1470 step，均无 NaN。
- Stage-1 旧域精确审计已完成。相对 T0 的日志聚类配对均值变化为：顺序训练
  `+0.0343`、LwF `+0.0994`、固定 OPD `+0.0777`、EMA-OPD `+0.0486`。
  当前证据没有显示总分遗忘；EMA 的 NC/TTC 有负向趋势。
- Stage-2 新域 343-token 精确指标缓存已完成并通过完整性检查；四臂的新域统一评测
  尚未启动。
- 上述结果只能作为机制/工程可行性证据，不能冒充 pristine 官方 DiffusionDrive
  主结果。官方复现和主实验由 06G 负责。

## 3. 当前服务器工作包

### C0：治理与证据合同（8月17日）

- 冻结双机职责、命名、目录、manifest 和结果 schema。
- 建立独立 `codex/` 分支；源码、配置、测试和报告进入 Git，checkpoint、cache、日志
  留在忽略目录。
- 所有跨机结果必须带代码 commit、配置 SHA、数据 manifest SHA、checkpoint SHA、
  样本数、失败数、随机种子和硬件/环境摘要。缺任一项只记为诊断证据。

交付：本文件、06G 规划、共享结果合同、清洁或已解释的 `git status --short`。

### C1：完成本机 Round001 审计（8月18日前）

- 在冻结的 343-token Stage-2 审计集上统一评测 T0 和四个 Stage-2 终点。
- 报告旧能力保持、新能力增益、逐指标 NC/DAC/TTC/comfort/progress、日志聚类
  paired bootstrap 95% CI；不得只报告平均分。
- 将“无旧域遗忘但有正迁移”与“新域收益是否被蒸馏抑制”分开解释。
- 完成当轮数学审计、强制网络文献检索、证据边界和下一轮门控。

晋级门：只有真实新域收益和旧域保持同时可比，才允许用本轮选择方法；否则只把它
作为执行链路验证。

### C2：智驾原生因果与 Seed-0 漏斗（8月19日至8月25日）

- 完成四种感知/规划上下文互换，判断感知漂移是否因果改变规划响应。
- 统一一次 forward/backward/AdamW 更新，保证 LwF 与 OPD 仅在查询状态分布上不同。
- 先完成必要组件消融：顺序训练、planner-only、LwF、固定 OPD、EMA-OPD、仅感知
  保持、完整 Drive-OPD。
- 再按公平预算接入 EWC、A-GEM、step-matched replay、full-exposure replay、DER++、
  ALER-Drive；非关键 TALR/O-LoRA 只在主证据稳定后执行。

晋级门：至少两个真实冲突转换显著优于顺序训练；在高师生支持偏移子集上优于
LwF/ALER；Chronological-CL 不压制正迁移。

### C3：06G 主实验审核与统计（8月26日至9月4日）

- 对 06G 每个阶段做配置和样本级证据审核，不接受仅有最终均值的结果。
- Seed-0 漏斗冻结后，只将固定主表方法扩展为三种子。
- 建立阶段×数据域×指标矩阵，计算 BWT、FWT、阶段曲线面积和日志聚类置信区间。
- 分离优化步数收益：step-matched replay 与 full-exposure replay 必须同时存在。
- 9月4日前决定 Flow Matching 是否可进入正文；若 GoalFlow 环境/官方模型未过真实
  batch gate，则降为未来工作，不牺牲 DiffusionDrive 主表。

### C4：结果与论文冻结（9月5日至9月16日）

- **9月7日**：主结果、方法定义、统计协议和图表冻结；之后只修阻塞性错误。
- **9月8日至10日**：完成九页正文、附录、匿名化、复现清单和限制讨论；人工审计。
- **9月11日 AOE**：提交真实标题、作者列表和摘要；作者/OpenReview 信息此前完成。
- **9月12日至14日**：只做审计指出的必要复算、文字与补充材料修订。
- **9月15日**：最终人类审计、代码与结果 SHA 快照、提交候选版本冻结。
- **9月16日 AOE**：全文提交。

## 4. 论文最小可交付与降级策略

必须具备：

1. Failure-Patch-CL 与 Chronological-CL 两种完整日志协议；
2. pristine 官方 DiffusionDrive 的顺序训练、replay、LwF、Drive-OPD 及关键 CL 基线；
3. 至少三个种子、日志聚类统计、分指标安全分析；
4. 感知漂移与学生分布指导的机制消融；
5. 可审计代码、配置、manifest 和结果哈希。

若算力或时间不足，按以下顺序降级：Flow Matching → O-LoRA/TALR → 非关键教师策略。
不得删减三种子主表、LwF/replay、公平预算或最终测试封存。

## 5. 工作树与人工审计纪律

- 每轮开始和结束记录 `git branch --show-current`、`git status --short`、HEAD SHA。
- 禁止在 `main` 上实验性提交；所有变更进入命名清楚的 `codex/` 分支。
- 大产物只写入预登记 artifact 目录；提交 manifest、配置、摘要和 SHA，不提交权重、
  cache、pickle 或原始数据。
- 一个提交只覆盖一个可审计目的；提交前运行相关 CPU 测试和真实一批 GPU gate。
- 以下情况必须主动请求人工审计：数据切分/最终测试访问变化、损失语义或公平预算变化、
  主表配置冻结、大规模删除、合并/历史改写、提交候选版本冻结。
- 审计通过后立即提交并推送；推送失败必须记录分支、commit、错误和所需人工动作，不能
  把已完成工作继续留为未追踪文件。
- 不执行破坏性 reset/checkout，不覆盖用户已有修改；无法归属的脏文件先隔离并请求审计。

## 6. 双机接口

06G 每个里程碑只同步：

```text
code_commit
config_path + config_sha256
data_manifest_path + manifest_sha256
checkpoint_path_on_06G + checkpoint_sha256
seed / stage / arm / optimizer / lr / batch / steps
metric_json / per_log_csv / curve_csv
sample_count / failure_count / environment_summary
```

当前服务器给 06G 的唯一可执行输入是已冻结配置与 manifest；口头超参数或未提交补丁
不得进入主实验。

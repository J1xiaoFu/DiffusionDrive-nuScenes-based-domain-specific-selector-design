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

## 2. AutoResearch-Drive 论文交付闭环

从本计划修订起，研究轮编号使用 `P###`。一轮不再以“验证能否运行”或“可行性通过”
作为科研目标；import、单 batch、显存和短跑只属于工程 gate。每个 `P###` 必须对应
论文中的一个明确 claim、主表单元、消融问题或机制图，并完成以下闭环：

1. **论文问题与数学预注册**：写明主张、反事实、优化目标、假设、可辨识量、理论预期、
   主要/次要终点、统计单位、失败判据和会影响正文哪张表/图。
2. **强制相关工作检索**：实验前检索最相关的原始论文，确认是否已有相同现象与解决
   思路；遇到异常结果后再次检索，区分已知机制、可迁移方案和智驾特有差异。
3. **论文级实验**：冻结配置和公平预算，运行主比较、负对照、关键消融与统计检验；
   工程失败不算一轮，修复后仍需交付完整实验。
4. **论文级章节包**：产出可直接进入论文的 Method、Experimental Setup、Results、
   Analysis、Related Work、Limitations、Claims-to-Evidence；数值实验至少含一张结果表、
   一张带置信区间的图及其生成脚本，禁止只留日志或会话总结。
5. **数学与机制审计**：用梯度、分布偏移、条件生成与优化器动力学解释结果，列出替代
   解释，并用数据或额外对照排除；观察不符合推导时必须更新假设，不能只调超参数。
6. **无上下文内部盲审**：每轮新建一个独立 GPT-5.6-sol 任务，只提供该轮冻结 commit
   和章节包，以 ICLR 审稿标准给出评分、置信度、致命问题、缺失基线、理论漏洞、统计
   与复现问题。当前作者任务生成逐条 Author Response，并把可执行意见纳入下一轮。
7. **用户审计旁路**：向用户提供可读摘要、表图和内部 review，明确请求 feedback；该
   feedback 不阻塞自动迭代，也不自动进入下一轮。只有用户明确要求调整时才更新目标。
8. **版本闭环**：章节包、内部 review、Author Response、配置、图表脚本和结果哈希通过
   测试后提交并推送；工作树清洁后才可把 `P###` 标为完成。

每轮建议目录固定为：

```text
paper_iterations/P###_<claim>/
  ROUND_MANIFEST.json
  PAPER_SECTION.md
  METHOD_AND_THEORY.md
  EXPERIMENTS.md
  RESULTS_AND_ANALYSIS.md
  RELATED_WORK.md
  CLAIMS_EVIDENCE.md
  figures/  tables/  plot_scripts/
  INTERNAL_REVIEW.md
  AUTHOR_RESPONSE.md
  USER_AUDIT_REQUEST.md
  DECISION.md
```

内部 review 是自动研究闭环的一部分；用户 feedback 是额外审计通道。负结果可以完成
一轮，但必须改变论文主张、淘汰方法或产生有证据支持的新实验，而不能以“再扫一轮”
作为结论。

## 3. 当前已知证据与待完成点

- 本机 RAP-modified NAVSIM 历史先导训练已完成 Stage-2 的四臂：顺序训练、LwF、固定
  教师 OPD、`m=0.99` EMA-OPD；每臂 30 epoch、1470 step，均无 NaN。
- Stage-1 旧域精确审计已完成。相对 T0 的日志聚类配对均值变化为：顺序训练
  `+0.0343`、LwF `+0.0994`、固定 OPD `+0.0777`、EMA-OPD `+0.0486`。
  当前证据没有显示总分遗忘；EMA 的 NC/TTC 有负向趋势。
- Stage-2 新域 343-token 精确指标缓存已完成并通过完整性检查；四臂的新域统一评测
  尚未启动。
- 上述结果只能作为历史机制诊断证据，不能冒充 pristine 官方 DiffusionDrive
  主结果。官方复现和主实验由 06G 负责。

## 4. 当前服务器工作包

### C0：治理与证据合同（8月17日）

- 冻结双机职责、命名、目录、manifest 和结果 schema。
- 建立独立 `codex/` 分支；源码、配置、测试和报告进入 Git，checkpoint、cache、日志
  留在忽略目录。
- 所有跨机结果必须带代码 commit、配置 SHA、数据 manifest SHA、checkpoint SHA、
  样本数、失败数、随机种子和硬件/环境摘要。缺任一项只记为诊断证据。

交付：本文件、06G 规划、共享结果合同、清洁或已解释的 `git status --short`。

### C1 / P001：调度一致的支持蒸馏与会话原子协议（8月20日前）

- 推导并实现 scheduler-consistent 的学生 rollout；生成、查询和转移 timestep 必须一致，
  并用“OPD/LwF 输入同一状态时 loss/gradient 相同”的官方模型负对照冻结公平性。
- 对齐本机 14,951-token cache inventory 与 06G 官方 103,288-token raw inventory，逐
  session/log/token 解释纳入与排除；在完成前禁止用 10,444-token portable manifest 定义
  pristine 官方主实验。
- 把主要统计单位从 segmented log 改为时间戳×车辆 session；预注册主 estimand、
  seed/session 层级汇总、无效行、最小效应量和多重比较规则。
- 修复或降级 EWC batch-gradient Fisher，并阻止 A-GEM replay forward 更新 BN buffer。
- 形成完整 `P001` 章节包、调度机制图、token 覆盖表、数学审计、两阶段文献检索、
  证据边界和下一轮决策。旧四臂性能评测不再占用论文关键路径。
- 冻结并推送章节包后，新建无上下文 GPT-5.6-sol ICLR reviewer 任务；保存 review 和
  Author Response，再向用户展示同一版本以获得额外 feedback。

论文决策：只有状态/时间、公平预算、数据覆盖与统计单位四项均可识别，P002 才能产生
性能主张；否则继续收缩协议，不能运行或解释新的 OPD/LwF 排名。

2026-08-17 P001 实测更新：本机 `navtrain.yaml` 的 103,288 个 allowlist token 全部是
unit-stride 下合法场景；其声明的 `frame_interval=14` 只产生 7,457 个场景，而现有
`valid_cache_train.pkl` 含 14,951 个场景（allowlist 的 14.48%）。14,951 与 CL 完整
manifest 完全一致，10,444 是其中三个 train split 的并集；但 14,951 子集没有生成
规则、随机种子或 receipt，因此数据覆盖门当前为 **FAIL**。不得事后把现有 token 列表
包装成可复现抽样。06G 必须从冻结的 pristine 官方 config 与 loader 重新生成场景清单
和 cache receipt，再由本机重建 CL manifest；在此之前 P002 性能排名保持阻塞。

### C2 / P002：pristine 官方 Seed-0 稳定—塑性主表（8月21日至8月25日）

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

## 5. 论文最小可交付与降级策略

必须具备：

1. Failure-Patch-CL 与 Chronological-CL 两种完整日志协议；
2. pristine 官方 DiffusionDrive 的顺序训练、replay、LwF、Drive-OPD 及关键 CL 基线；
3. 至少三个种子、日志聚类统计、分指标安全分析；
4. 感知漂移与学生分布指导的机制消融；
5. 可审计代码、配置、manifest 和结果哈希。

若算力或时间不足，按以下顺序降级：Flow Matching → O-LoRA/TALR → 非关键教师策略。
不得删减三种子主表、LwF/replay、公平预算或最终测试封存。

## 6. 工作树与人工审计纪律

- 每轮开始和结束记录 `git branch --show-current`、`git status --short`、HEAD SHA。
- 禁止在 `main` 上实验性提交；所有变更进入命名清楚的 `codex/` 分支。
- 大产物只写入预登记 artifact 目录；提交 manifest、配置、摘要和 SHA，不提交权重、
  cache、pickle 或原始数据。
- 一个提交只覆盖一个可审计目的；提交前运行相关 CPU 测试和真实一批 GPU gate。
- 每轮先提交“待盲审候选”commit；内部 reviewer 只能审这一冻结版本。Author Response
  和修改另作提交，避免审稿对象随审计变化。
- 以下情况必须主动请求人工审计：数据切分/最终测试访问变化、损失语义或公平预算变化、
  主表配置冻结、大规模删除、合并/历史改写、提交候选版本冻结。
- 审计通过后立即提交并推送；推送失败必须记录分支、commit、错误和所需人工动作，不能
  把已完成工作继续留为未追踪文件。
- 用户默认授权常规研究、训练、评测、文件写入、建审计任务、提交与推送；大批量文件
  删除不在默认授权内，必须先列出精确对象、空间收益和恢复性并请求用户批准。
- 不执行破坏性 reset/checkout，不覆盖用户已有修改；无法归属的脏文件先隔离并请求审计。

## 7. 双机接口

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

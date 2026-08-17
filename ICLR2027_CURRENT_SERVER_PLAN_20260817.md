# ICLR 2027 Drive-OPD：当前服务器目标与执行规划

状态：ACTIVE — 07G NUSCENES METHOD / EXTERNAL-VALIDITY LANE；CLAIM GPU BLOCKED

责任服务器：07G（方法研究与 nuScenes 独立证据线）

责任任务：`ICLR 2027｜Drive-OPD 方法与论文总控`

研究工作分支：`codex/iclr2027-drive-opd-07g-research`

总控工作树：`/home/khwang/domain-selector-controller`

本机研究分支/工作树：`codex/iclr2027-drive-opd-07g-research` / `/home/khwang/domain-selector`

计划冻结日：2026-08-17

双服务器修订：`ICLR2027_DUAL_SERVER_EXECUTION_PLAN_20260817.md` 是当前职责、基线与门禁
的上位合同；本文件中“07G 不承担计算”或“所有官方实验只在 06G”的旧表述由该合同取代。

## 0. 分支职责与最新远端门状态

- 总控任务 ID 为 `01a00dab-5dc6-7413-8ca7-930353db0e0c`；06G 执行任务 ID 为
  `01a00e84-1f10-7331-ac52-fbc0c5d75a5a`。`01a00f54-8adc-7372-a8c5-bb3efe33fbef`
  仅为已完成的 P001 独立盲审任务。根目录 `AGENTS.md` 固化了该身份与职责合同。
- `codex/iclr2027-drive-opd-controller` 只管理跨机计划、冻结 manifest、远端 receipt、审稿
  结论、晋级门和论文里程碑索引。
- `codex/iclr2027-drive-opd-07g-research` 管理本机 Drive-OPD 方法、因果/统计实现、论文轮次、
  测试与图表。
- `codex/iclr2027-diffusiondrive-06g` 管理 pristine 官方 DiffusionDrive/NAVSIM 的数据、
  cache 与 GPU 执行证据；它不是方法或协议定义的第二来源。

2026-08-17 用户调度更新：07G 与 06G 都作为计算服务器，但按数据集隔离证据。06G 继续
负责 pristine DiffusionDrive/NAVSIM 主表；07G 负责 DiffusionDrive/nuScenes 方法研究、
持续学习协议、机制消融与外部有效性。两条线独立过门、独立成表，不合并 NAVSIM PDMS
与 nuScenes planning L2/collision，也不允许一条线的 PASS 解锁另一条线。

07G 本机只读资源快照发现 8×RTX 3090 均被现有任务占用，`/home` 仅余约 51 GB；已有
`/home/khwang/datasets/nuscenes` 约 474 GB。不得终止他人任务、抢占 GPU 或复制第二份
完整数据。先做 PID/所有者、数据版本、scene/sample、metadata 与磁盘水位 receipt。

冻结候选 `9dabf42371e8dde72e1b2061bac4fa9d11b34230` 的新盲审为 3/10 reject；seed、
CSV/checkpoint、method/domain identity、receipt discovery/uniqueness 存在可执行绕过。该 P0
是数据集无关的共同门。修复、冻结并通过新盲审前，07G 只做源码、CPU、数据/切分审计，
不运行 claim-bearing nuScenes cache、T0、训练或 GPU 评测。

2026-08-17，06G R1 原子提交 `1bc9cc2957eeccc6927f3dc826135ce85bfae790`
通过官方 navtrain 总体与官方 train/val 划分核验：103,288 = 85,109 + 18,179，完整 token
SHA256 为 `5614a8a32a7030bda7cc71696e085f59116df442ba6c0b2604bb0649fea7f083`，与总控
独立重建一致。审计源见 `controller_receipts/06g/R1_official_navtrain_20260817/`。

该通过范围不包括训练 cache、三阶段持续学习 manifest、T0、GPU gate 或任何性能结果；
这些计算继续封存。

## 1. 截止日与唯一总目标

ICLR 2027 官方摘要截止为 **2026-09-11 AOE**，全文截止为
**2026-09-16 AOE**。用户此前估计的 2026-09-25 不再作为提交截止日。

唯一总目标是在官方截止日前形成一篇证据边界清楚、可复现并达到投稿质量的
智驾原生持续学习论文：定义真实日志增量协议，验证 Drive-OPD 是否在学习新能力的
同时保持旧能力，并用因果审计说明学生分布指导与感知漂移处理何时必要。

07G 承担 Drive-OPD 方法、协议、统计、证据审核和 nuScenes 计算；06G 承担 pristine 官方
DiffusionDrive/NAVSIM 主实验。两机只交换源码、配置、manifest、SHA256、JSON、CSV 与
曲线；因数据集不同，不跨机拼接 checkpoint 或把一台服务器的方法结果与另一台服务器
的基线结果直接比较。

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

## 4. 07G 工作包

以下 N0–N3 是当前执行目标；原 C0–C4 内容保留为治理/论文历史与 NAVSIM 接口背景，发生
冲突时以 N0–N3 和双服务器计划为准。

### N0：nuScenes 资源、数据与协议清点

- 记录实际 GPU UUID/PID/所有者、磁盘安全水位，不抢占现有任务。
- 对 `/home/khwang/datasets/nuscenes` 冻结版本、scene/sample counts、metadata hash、
  maps/CAN bus 完整性与 trainval/test 边界；final test 继续封存。
- 以 scene 为原子冻结 train/calibration/audit 与 Chronological-CL 规则；旧 40/10 scene
  partial 只有在生成规则和 membership 可重放时才可升级。

### N1：claim-protocol P0 与真实 nuScenes adapter

- 修复 seed/run/config/protocol 绑定、checkpoint→evaluator→CSV lineage、method registry、
  domain registry、显式 receipt inventory 和内容唯一性。
- 增加非对角 checkpoint-stage × evaluation-stage/domain、恶意换 CSV/checkpoint、重复
  receipt、未知 domain 等回归测试，并完成新一轮无上下文盲审。
- 新盲审通过且 GPU 确认空闲后，才做一个真实 batch 的 loss/query/gradient/resume gate。

### N2：nuScenes controlled-partial Seed-0

- 核心臂：T0、sequential、step-matched replay、LwF、Drive-OPD、joint/all-seen oracle。
- 使用官方 planning L2/collision horizons 与 scene-level retention/plasticity；不生成或借用
  NAVSIM PDMS。
- 没有同时出现新域塑性和非平凡旧域遗忘时停止扩臂，并按负结果收缩论文主张。

### N3：nuScenes 扩展与机制

- Seed-0 过门后才加入 full-exposure replay、DER++、fixed/EMA OPD 与三种子。
- planner-only、perception-only、identical-state、no-drift/context-swap 属于消融；EWC、
  A-GEM 只做 real-adapter 通过后的次要比较。
- ALER-Drive、TALR、O-LoRA 和 GoalFlow 不进入关键路径。

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
- 旧基线列表由双服务器计划取代。NAVSIM 主表核心为 sequential、step-matched replay、
  full-exposure replay、LwF、DER++、Drive-OPD 与 joint oracle；EWC/A-GEM 为验证通过后的
  次要臂，ALER-Drive/TALR/O-LoRA 不在关键路径。

晋级门：至少两个真实冲突转换达到预注册最小效应并优于顺序训练；在高师生支持偏移
子集上优于 LwF 与 step-matched replay；Chronological-CL 不压制正迁移。所有检验进入
同一冻结 Holm family，不以未校准的“显著”计数晋级。

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
2. 06G pristine NAVSIM 的顺序训练、两类 replay、LwF、DER++、Drive-OPD、joint oracle，
   以及 07G 独立 nuScenes Tier-A 机制表；
3. 至少三个种子、日志聚类统计、分指标安全分析；
4. 感知漂移与学生分布指导的机制消融；
5. 可审计代码、配置、manifest 和结果哈希。

若算力或时间不足，按以下顺序降级：ALER/TALR/O-LoRA/Flow Matching → EWC/A-GEM →
nuScenes 扩大规模 → nuScenes DER++。不得删减 06G NAVSIM 三种子主表、LwF/两类 replay、
DER++、joint oracle、公平预算或最终测试封存。

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

07G 与 06G 都只执行 controller 冻结的源码、配置和 manifest；口头超参数或未提交补丁
不得进入实验。两机结果必须额外带 dataset/version、method registry、evaluation-domain
registry、stage-start state SHA、result/checkpoint lineage、实际 query/forward/update/exposure
预算和 GPU UUID。nuScenes 与 NAVSIM 结果保持独立表格和独立统计 family。

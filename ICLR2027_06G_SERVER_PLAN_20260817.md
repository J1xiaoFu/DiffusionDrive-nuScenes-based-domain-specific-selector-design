# ICLR 2027 Drive-OPD：06G 官方实验目标与执行规划

状态：DISPATCHED，等待 06G 返回 `REMOTE_GOAL_SET`

责任服务器：06G

沿用任务：`ICLR 2027｜06G 官方 DiffusionDrive 主实验`

远端工作区：`/home/xlxia/drive_opd_remote_20260817T150443_06g`

目标分支：`codex/iclr2027-diffusiondrive-06g`

## 1. 唯一目标

在 ICLR 2027 官方摘要截止 **2026-09-11 AOE**、全文截止
**2026-09-16 AOE** 前，基于冻结的官方 `hustvl/DiffusionDrive` commit
`9b52ed0ec06b073d82d6f392ab084c7b301c8681` 和完整 NAVSIM navtrain，完成
pristine 官方 DiffusionDrive 的持续学习缓存、T0、Seed-0 方法漏斗、配置冻结后的
三种子主实验和结构化评测，为当前服务器提供可审计主表证据。

06G 是执行器而不是方法定义的第二来源：不得在远端静默改变数据协议、损失语义、
baseline 公平预算或最终测试边界。发现问题时先报告证据，由当前服务器更新冻结配置。

## 2. 06G AutoResearch-Drive 闭环

06G 同样运行自迭代 research，而不是只做无解释的训练队列。工程 gate（环境、import、
单 batch、OOM 恢复）不计为研究轮。每个远端轮次使用与当前服务器一致的 `P###` claim
编号，并必须完成：

1. 从论文 claim 反推实验问题，预注册数学预期、主/次终点、统计单位、负对照、失败
   判据和目标表图；不得以“先跑起来看看”定义一轮。
2. 实验前检索最相关原始工作；遇到异常损失、漂移或优化器现象时再次检索已有解释和
   解决方案，并记录哪些机制可迁移到智驾、哪些不满足物理/条件生成假设。
3. 在冻结官方代码和公平预算下执行训练与统一评测，保留逐日志结果、失败样本、loss
   曲线和资源成本，而不仅是最终均值。
4. 在远端产出论文级 `PAPER_SECTION.md`、方法/理论、实验设置、结果与机制分析、相关
   工作、局限、Claims-to-Evidence、至少一张表和一张带置信区间的图及绘图脚本。
5. 推送待审 commit 后，为每轮新建一个无既有对话上下文的 GPT-5.6-sol reviewer 任务，
   只给冻结 commit/章节包，要求按 ICLR 标准审稿；保存 review，逐条撰写 Author
   Response，并将内部 review 的可执行意见纳入下一轮。
6. 向用户单独展示章节包、表图和内部 review 并请求 feedback；用户 feedback 是额外
   审计，不阻塞远端自动迭代，也不自动修改配置，除非用户明确下达变更。
7. review、response、结果哈希、源码、配置与图表脚本提交推送且工作树清洁后，才能把
   该轮标记为完成。

负结果必须收缩论文主张、淘汰方案或支持下一项可检验机制；不能自动转化为更多无假设
的超参数扫描。06G 可以自主修复工程问题和补充已预注册对照，但协议/损失/最终测试的
变更仍需回传当前服务器确认。

## 3. 已知起点

- 数据环境脚本：`/home/xlxia/datasets/navsim-navtrain/env.sh`。
- 已核验 1192 个 whitelist log、103288 token，九种必需模态各 152495 文件，必需集合
  零缺失；另有 118 个非 whitelist annotation log，不得混入协议。
- training/metric cache、兼容训练环境和 T0 尚未完成。
- 本机已有 devkit 不是官方 DiffusionDrive；必须使用上述冻结官方 commit，不得把
  TransFuser 或本地修改版当作官方基线。
- 最近一次资源快照中 GPU1 有外部 VLLM；每次启动前必须重查，不能占用或终止他人任务。

## 4. 阶段门与交付

### R0：项目与环境冻结（8月18日前）

- 沿用现有隔离工作区和任务，不新建重复项目。
- 官方 clone 保留只读参考；需要适配时在独立 `codex/` 分支/工作树完成。
- 建立兼容环境；记录 Python、PyTorch、CUDA、关键包、DiffusionDrive commit 和完整
  安装命令。镜像/solver 失败也要保存原始错误，替代源必须最小且可复现。
- 通过 CPU import、配置解析和单 GPU 一批真实 forward/backward gate。

交付：环境摘要、commit、`git status --short`、真实 batch 日志和显存峰值。

### R1：精确缓存与协议映射（8月20日前）

- **阻塞门**：P001 已证明本机 14,951-token cache 是 103,288 个合法 allowlist 场景的
  14.48% 后验子集，不等于本机 `frame_interval=14` 所产生的 7,457 个场景，且缺失原始
  选择规则/seed/receipt。portable manifest 的 10,444 只是该不可复现 cache 子集的训练
  split。不得复制这两个集合或把它们当作 pristine 官方主实验数据。
- 在冻结的官方 commit 上直接运行官方 SceneLoader/config，输出 raw allowlist → 合法
  window → 实际训练 token 的逐 log/session reason counts。不得预设官方结果一定是
  103,288，也不得用扫描已有 cache 目录来替代源数据选择。
- 06G 存储充足时优先缓存官方规则产生的完整训练集合；若预计空间/时间无法接受，先向
  当前服务器回传完整集合计数和成本，由当前服务器预注册确定性子采样规则后再构建，
  禁止先看结果再选子集。
- cache receipt 必须记录官方 config/source SHA、原始数据树 SHA、选择规则、seed、逐
  session/token-set SHA、cache builder 列表、成功/失败 token 和最终 index SHA。只有
  receipt 能从源数据重放得到同一 token hash，当前服务器才重建 Failure-Patch/Chronological
  manifests 并解除 P002 阻塞。
- 只在对齐后使用冻结 manifest 指定的训练 token；审计/测试 token 不得进入训练 cache。
- 构建官方 training/metric cache，并对 token 集合、数量、重复和缺失做 SHA 审计。
- 将 Failure-Patch-CL 与 Chronological-CL 映射到完整 log 原子边界。

交付：cache manifest、数据 SHA、样本数、缺失/额外项、生成命令。缓存和原始数据不进 Git。

### R2：T0 与 Seed-0 漏斗（8月21日至8月28日）

- 在任何蒸馏训练前，完成 scheduler-consistent 状态/时间合同和官方模型负对照；OPD 与
  LwF 输入同一状态时，response/mode loss 与梯度必须在容差内相同。
- 使用同一官方训练脚本、初始化、AdamW、学习率、batch、增强和总步数训练 Stage-1 T0。
- 在 Stage-2 运行最小四臂：顺序训练、LwF、固定教师 OPD、`m=0.99` EMA-OPD。
- 所有方法保持一次联合 forward/backward 和一次 AdamW 更新；LwF 与 OPD 只允许查询
  状态分布不同。
- 先统一完成训练，再用同一 evaluator 评测，不在训练中混入会扰动状态的临时评测。
- 按当前服务器冻结配置补充 planner-only、replay、DER++、EWC/A-GEM 和完整 Drive-OPD。

交付：每臂 endpoint/rolling optimizer checkpoint、loss CSV、配置与 checkpoint SHA、
逐日志指标，以及对应该 claim 的论文级章节、表图、数学分析和相关工作检索。只永久
保留阶段终点和最近一个可恢复断点。

### R3：配置冻结与三种子主实验（8月29日至9月4日）

- Seed-0 只用于筛选，不反复针对最终测试调参。
- 当前服务器确认晋级方法和配置 SHA 后，扩展到三种子。
- 同时运行 step-matched replay 与 full-exposure replay，显式分离额外 exposure 收益。
- 每个阶段评测所有已见域；输出阶段×域×PDMS/NC/DAC/TTC/comfort/progress 矩阵。

交付：三种子 JSON/CSV、逐日志结果、训练与评测曲线、失败清单和完整运行 manifest。

### R4：冻结与复算窗口（9月5日至9月15日）

- 9月5日交付全部主实验结构化结果；9月7日配合当前服务器冻结主结果。
- 9月8日后禁止新增调参臂，只复算审计确认的错误或缺失结果。
- 9月15日生成最终代码 commit、配置 SHA、checkpoint SHA、结果包 SHA 和恢复说明。

## 5. 资源与运行纪律

- 每个 GPU/NVIDIA 命令遵守 06G 本地 `AGENTS.md`/启动器要求；若远端无同名规则，则至少
  在启动前后记录 GPU UUID、显存、占用 PID 和 CUDA 可见映射。
- 不根据序号假设设备空闲；GPU1 必须重新证明确实空闲后才能使用。
- 训练权重留在 06G；跨机只同步配置、SHA、JSON、CSV 和曲线。
- 大任务必须支持滚动恢复；磁盘和 checkpoint 保留策略在启动前写入 manifest。
- 任何 OOM/NaN/中断都保留首个失败证据；不得静默减 batch 或改变损失。

## 6. Git 与人工审计纪律

- 每轮开始/结束记录分支、HEAD 和 `git status --short`。
- 官方 frozen clone 不直接修改；所有适配进入 `codex/iclr2027-diffusiondrive-06g`。
- 源码、配置、测试、短报告入 Git；数据、cache、checkpoint、完整日志不得入 Git。
- 通过相关测试和一批真实 GPU gate 后按单一目的提交并推送。
- 每轮先冻结并推送待盲审 commit；内部 reviewer 不审未提交的移动目标。review 与
  Author Response 另作提交。
- 数据边界、损失语义、公平预算、主表配置、大规模删除、合并或提交候选冻结前，主动
  请求人工审计；审计通过后完成 commit/push，不长期遗留未追踪源码。
- 用户默认授权常规研究、训练、评测、文件写入、建审计任务、提交与推送；大批量文件
  删除必须先报告精确对象、预计释放空间和恢复方式并等待用户明确批准。
- 推送或权限失败时立即报告确切分支、commit、错误和所需人工动作，不另建无法追踪的
  影子代码树。

## 7. 向当前服务器回传的固定 schema

```text
server=06G
code_commit / dirty_status
official_upstream_commit
config_path / config_sha256
data_manifest_path / manifest_sha256
stage / arm / seed
optimizer / lr / batch / steps / teacher_policy
checkpoint_path / checkpoint_sha256
metric_json / per_log_csv / loss_curve_csv
sample_count / failure_count
gpu_uuid / environment_summary
evidence_boundary / known_deviations
```

缺少代码 commit、配置 SHA 或数据 manifest SHA 的运行不得进入论文主表。

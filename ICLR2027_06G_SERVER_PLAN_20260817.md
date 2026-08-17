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

## 2. 已知起点

- 数据环境脚本：`/home/xlxia/datasets/navsim-navtrain/env.sh`。
- 已核验 1192 个 whitelist log、103288 token，九种必需模态各 152495 文件，必需集合
  零缺失；另有 118 个非 whitelist annotation log，不得混入协议。
- training/metric cache、兼容训练环境和 T0 尚未完成。
- 本机已有 devkit 不是官方 DiffusionDrive；必须使用上述冻结官方 commit，不得把
  TransFuser 或本地修改版当作官方基线。
- 最近一次资源快照中 GPU1 有外部 VLLM；每次启动前必须重查，不能占用或终止他人任务。

## 3. 阶段门与交付

### R0：项目与环境冻结（8月18日前）

- 沿用现有隔离工作区和任务，不新建重复项目。
- 官方 clone 保留只读参考；需要适配时在独立 `codex/` 分支/工作树完成。
- 建立兼容环境；记录 Python、PyTorch、CUDA、关键包、DiffusionDrive commit 和完整
  安装命令。镜像/solver 失败也要保存原始错误，替代源必须最小且可复现。
- 通过 CPU import、配置解析和单 GPU 一批真实 forward/backward gate。

交付：环境摘要、commit、`git status --short`、真实 batch 日志和显存峰值。

### R1：精确缓存与协议映射（8月20日前）

- 只使用 portable manifest 指定的训练 token；审计/测试 token 不得进入训练 cache。
- 构建官方 training/metric cache，并对 token 集合、数量、重复和缺失做 SHA 审计。
- 将 Failure-Patch-CL 与 Chronological-CL 映射到完整 log 原子边界。

交付：cache manifest、数据 SHA、样本数、缺失/额外项、生成命令。缓存和原始数据不进 Git。

### R2：T0 与 Seed-0 漏斗（8月21日至8月28日）

- 使用同一官方训练脚本、初始化、AdamW、学习率、batch、增强和总步数训练 Stage-1 T0。
- 在 Stage-2 运行最小四臂：顺序训练、LwF、固定教师 OPD、`m=0.99` EMA-OPD。
- 所有方法保持一次联合 forward/backward 和一次 AdamW 更新；LwF 与 OPD 只允许查询
  状态分布不同。
- 先统一完成训练，再用同一 evaluator 评测，不在训练中混入会扰动状态的临时评测。
- 按当前服务器冻结配置补充 planner-only、replay、DER++、EWC/A-GEM 和完整 Drive-OPD。

交付：每臂 endpoint/rolling optimizer checkpoint、loss CSV、配置与 checkpoint SHA、
逐日志指标。只永久保留阶段终点和最近一个可恢复断点。

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

## 4. 资源与运行纪律

- 每个 GPU/NVIDIA 命令遵守 06G 本地 `AGENTS.md`/启动器要求；若远端无同名规则，则至少
  在启动前后记录 GPU UUID、显存、占用 PID 和 CUDA 可见映射。
- 不根据序号假设设备空闲；GPU1 必须重新证明确实空闲后才能使用。
- 训练权重留在 06G；跨机只同步配置、SHA、JSON、CSV 和曲线。
- 大任务必须支持滚动恢复；磁盘和 checkpoint 保留策略在启动前写入 manifest。
- 任何 OOM/NaN/中断都保留首个失败证据；不得静默减 batch 或改变损失。

## 5. Git 与人工审计纪律

- 每轮开始/结束记录分支、HEAD 和 `git status --short`。
- 官方 frozen clone 不直接修改；所有适配进入 `codex/iclr2027-diffusiondrive-06g`。
- 源码、配置、测试、短报告入 Git；数据、cache、checkpoint、完整日志不得入 Git。
- 通过相关测试和一批真实 GPU gate 后按单一目的提交并推送。
- 数据边界、损失语义、公平预算、主表配置、大规模删除、合并或提交候选冻结前，主动
  请求人工审计；审计通过后完成 commit/push，不长期遗留未追踪源码。
- 推送或权限失败时立即报告确切分支、commit、错误和所需人工动作，不另建无法追踪的
  影子代码树。

## 6. 向当前服务器回传的固定 schema

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

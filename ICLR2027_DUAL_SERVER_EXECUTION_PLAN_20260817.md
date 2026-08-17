# Drive-OPD ICLR 2027 双服务器并发执行计划

状态：DUTY CONTRACTS ACKNOWLEDGED；不授权 cache/T0/GPU/training

冻结日期：2026-08-17

控制分支：`codex/iclr2027-dual-server-controller`

控制起点：`3b6523de2f06f912e6e7834a0a5ce4166864f544`

共同 claim-protocol 的唯一可执行 P0 验收门冻结在
`ICLR2027_P0_CLAIM_PROTOCOL_ACCEPTANCE_20260817.md`。通过该门只允许进入各数据线后续
provenance/adapter 审计，不自动授权 cache、T0、GPU、training 或性能 claim。

## 1. 总目标与调度决定

论文主目标仍是验证 Drive-OPD 能否在真实驾驶分布的顺序变化中兼顾新域塑性与旧域
保持，并解释学生支持分布与感知漂移的作用。计算改为两条并发但证据隔离的数据线：

- **06G / NAVSIM 主线**：pristine 官方 DiffusionDrive、论文主表、Failure-Patch-CL 与
  Chronological-CL、三种子和 NAVSIM PDM 指标。
- **07G / nuScenes 副线**：Drive-OPD 方法研究、严格协议实现、nuScenes 场景原子持续学习、
  机制消融和跨数据集外部有效性。该线使用 nuScenes 官方规划指标，不输出 NAVSIM PDMS。
- **Controller**：冻结共同方法语义、公平预算、基线层级、数据/评测边界和 receipt schema；
  只依据可验证 commit、SHA 和独立 replay receipt 晋级。

两条线分别过门，互不解锁。NAVSIM 与 nuScenes 的数值不得合并平均或进入同一个显著性
检验；正文只能对同方向的机制结果做预注册后的定性/分层总结。

## 2. 当前事实与硬门

### 06G / NAVSIM

- R1 `1bc9cc2957eeccc6927f3dc826135ce85bfae790` 仅通过官方 103,288-token
  source population 与 85,109/18,179 train/validation 重建。
- claim-bearing cache、session-atomic CL stages、T0、GPU 和性能实验继续 STOPPED。
- 解锁顺序：controller source ref → session-atomic manifests → cache build+independent replay
  → cache/index/environment hashes → real-adapter OPD/LwF budget preflight。

### 07G / nuScenes

- 本机发现 `/home/khwang/datasets/nuscenes`，约 474 GB；历史文档还引用另一主机的
  `/root/autodl-*` 路径，这些路径不能作为本机证据。
- 2026-08-17 快照显示 8×RTX 3090 均被现有进程占用且利用率为 65–100%；不得抢占、
  终止或复用他人进程。`/home` 仅余约 51 GB，不得复制第二份完整 nuScenes。
- 历史 40 train scenes / 1,611 samples 与 10 val scenes / 401 samples 的 controlled partial
  只可作为既有诊断，必须重新绑定本机实际数据、scene/sample hashes、生成规则和 checkpoint。
- `9dabf42371e8dde72e1b2061bac4fa9d11b34230` 的新盲审为 3/10 reject；seed、结果文件、
  checkpoint、method/domain identity 与 receipt 去重存在 P0 绕过。新冻结审计通过前，
  nuScenes claim-bearing cache/T0/training/GPU evaluation 全部 STOPPED。

07G 立即获准做只读资源/PID/磁盘清点、数据 provenance、scene-atomic split 设计、源码修复、
CPU tests 和 blind review closure。任何单 batch GPU gate 也必须等待空闲 GPU 被证实且新 P0
审计通过；禁止通过换数据集规避共同的 claim-protocol gate。

## 3. 调整后的实验基线

基线按论文价值分层，避免双机并发扩大成无假设的臂扫描。

### Tier A：两条数据线的核心比较

1. **T0 / unchanged stage-start**：参考点，不算持续学习方法。
2. **Sequential fine-tuning**：遗忘下界。
3. **Step-matched replay**：与当前方法相同 optimizer updates/forward-backward 预算；同时
   报告 current/old unique-token exposure。
4. **LwF**：固定教师、教师数据支持上的蒸馏；与 Drive-OPD 使用相同 loss terms、teacher
   queries、forward/backward 和 optimizer updates。
5. **Drive-OPD**：scheduler-consistent student-support distillation；真实 adapter query
   counter 必须在 optimizer step 前通过预算门。

### Tier B：主线强基线与上界

6. **Full-exposure replay**：显式展示额外旧数据 exposure 的收益，不与 step-matched replay
   混称为公平预算。
7. **DER++**：依据 [Dark Experience Replay++](https://arxiv.org/abs/2004.07211) 的 replay +
   历史输出约束；必须冻结适用于连续轨迹/规划输出的明确回归定义，不能直接照搬分类
   logits 语义。
8. **Joint/all-seen oracle**：非持续学习上界，单独展示，不参加同预算优胜声明。

06G NAVSIM 主表要求 Tier A 全部、full-exposure replay、DER++ 和 joint oracle。07G nuScenes
先运行 Tier A 与 joint oracle；仅当 controlled-partial Seed-0 同时显示可学习的新域增益和
非平凡遗忘时，才扩展 DER++/full-exposure replay 与三种子。

### Tier C：消融与次要基线

- fixed-teacher OPD、EMA-OPD；
- planner-only、perception-only preservation；
- identical-state LwF/OPD negative control；
- no-drift/context-swap controls；
- EWC、A-GEM 只在 real-adapter estimator/buffer receipt 通过后做 Seed-0 次要比较。

### 从关键路径降级

- **ALER-Drive**：原方法来自
  [LLM latent-embedding repair](https://openreview.net/pdf/029da17b1af821f352ceb8e27573e1ae51a5e21f.pdf)；
  当前 driving adaptation 的 manifold/kinematic validity 未通过，只保留为探索性 appendix
  候选。
- **TALR、O-LoRA、GoalFlow/Flow Matching**：不进入 9 月 4 日前的关键路径。
- [H2C](https://arxiv.org/abs/2508.01158)/压缩 replay 等近期驾驶相关方案先进入
  related-work/设计审计；它们面向轨迹预测或其他任务，除非能在冻结预算内直接复现且
  不延误 Tier A/B，不新增实现臂。

## 4. 公平预算与结果合同

每个可比较 arm 必须绑定：

```text
dataset / dataset_version / dataset_root_metadata_sha256
code_commit / dirty_status / upstream_commit
config_path / config_sha256
stage_manifest_path / stage_manifest_sha256
stage_start_state_path / state_sha256
training_arm / method_id / seed / checkpoint_stage
evaluation_stage / evaluation_domain / registered_domain_manifest_sha256
optimizer / lr / batch / optimizer_updates
current_unique_tokens / old_unique_tokens / total_examples_seen
teacher_queries / student_queries / forward_backward_count
wall_time / peak_vram / stored_bytes / host / gpu_uuid
checkpoint_sha256 / result_csv_sha256 / receipt_sha256
```

- `method_id == training_arm` 必须由注册表验证；aliases 需预先冻结。
- distinct seeds 必须绑定 distinct run IDs、RNG records 和 checkpoint/result artifacts；
  byte-identical evidence 不得重复计数。
- candidate result/checkpoint/config/seed 必须由同一 immutable run manifest 绑定，禁止替换
  baseline CSV 或混搭 checkpoint。
- evaluation domain 必须来自冻结 registry；未知 domain fail closed。
- receipt discovery 基于 manifest 中的显式文件列表和 SHA，不依赖可绕过的文件名 suffix。
- 所有 expected checkpoint-stage × evaluation-stage/domain 单元必须完整；缺失、重复、非有限
  或身份不一致均 fail closed。

## 5. 07G：nuScenes 方法与外部有效性工作包

### N0：资源、数据与泄漏审计

- 提交实际 GPU UUID/PID/所有者、显存和磁盘水位；不终止现有进程。
- 绑定 `/home/khwang/datasets/nuscenes` 的版本表、scene/sample counts、metadata hashes、
  CAN bus/maps 完整性与 trainval/test 边界；final test 保持封存。
- 冻结 scene-atomic train/calibration/audit 切分。历史 partial 若无法由规则重建则弃用，
  不把旧列表事后包装成正式协议。
- 设置至少 100 GB 安全水位；当前约 51 GB 时只做元数据与源码工作，新的 cache/checkpoint
  必须先给出空间预算和非破坏性回收方案并请求审计。

### N1：共同协议 P0 与真实 adapter gate

- 修复盲审指出的六类身份/receipt 绕过并新增攻击性回归测试。
- 对 five-seed confirmatory threshold 做 power/calibration 审计；chronology 必须由时间戳和
  scene 顺序可执行重放，不只写在 prose 中。
- 消除热路径逐 query CUDA `.cpu()` 同步；计数器保持设备无关且在 step 前可审计。
- 新 context-free blind review 通过后，才允许空闲 GPU 上做一个真实 batch 的 loss、query、
  gradient、resume gate。

### N2：controlled-partial Seed-0 漏斗

- 仅运行 T0、sequential、step-matched replay、LwF、Drive-OPD、joint oracle。
- 使用官方 nuScenes planning L2/collision horizons，并按 scene 报告；感知指标与规划指标
  分开，不制造 NAVSIM PDMS。
- 若没有同时出现新域塑性和旧域遗忘，停止扩大基线，转为负结果/协议论文证据。

### N3：nuScenes 扩展

- 只有 N2 通过预注册最小效应门，才增加 full-exposure replay、DER++、fixed/EMA OPD 和
  三种子；扩大到更多 trainval scenes 需要新的 scene-manifest receipt。
- 输出独立 nuScenes 表与机制图，不与 06G NAVSIM 数值合并。

## 6. 06G：NAVSIM 主证据工作包

### R1：官方 cache 与 CL manifest

- 延续 source-population PASS，但完成 claim-bearing cache build、独立 replay 和
  session-atomic Failure-Patch/Chronological manifests。
- 所有缓存/manifest 由 06G 官方 pristine source 产生；07G nuScenes 不改变该数据门。

### R2：NAVSIM Seed-0 完整漏斗

- 先完成 T0、sequential、step-matched replay、LwF、Drive-OPD、full-exposure replay、
  DER++ 和 joint oracle；fixed/EMA OPD 是方法消融。
- 统一评测所有 checkpoint-stage × evaluation-stage/domain，输出七个冻结指标及 PDM
  聚合，禁止仅交最终均值。

### R3：三种子主表

- 只扩展 Seed-0 冻结后保留的 Tier A/B 方法；同一方法所有 seeds 使用同一 config SHA。
- 以 session×seed crossed paired analysis 和全局 Holm family 汇总；五种子只用于预注册的
  confirmatory 项且必须先有 power/calibration 依据。

## 7. 并发顺序与资源回退

当前并发：

- 07G：P0 修复、盲审闭环、nuScenes 元数据/切分/空间审计；等待 GPU 空闲。
- 06G：NAVSIM cache provenance/replay、CL manifests、真实 adapter budget preflight；GPU 停止。

各自数据门通过后：

- 06G 优先运行 NAVSIM Seed-0 Tier A/B 漏斗；这是论文主线，资源优先级最高。
- 07G 并行运行 nuScenes controlled-partial Tier A 漏斗；不等待 NAVSIM 结果，不使用 NAVSIM
  结果选择 nuScenes split。
- 两机都通过 Seed-0 决策门后并行三种子；任何一线失败只降级该线，不阻塞另一线的合格结果。

若资源不足，降级顺序为：ALER/TALR/O-LoRA/GoalFlow → EWC/A-GEM → nuScenes 扩大规模 →
nuScenes DER++。不得删除 06G NAVSIM 的 Tier A、full-exposure replay、DER++、joint oracle、
三种子、公平预算或最终测试封存。

## 8. 晋级与发布

每台服务器修改自己的根 `AGENTS.md` 和工作目标后，回传：branch、full commit SHA、父
commit、变更文件 SHA256、测试结果、`git status --short`、GPU/数据门状态。Controller
独立核验后发布 dispatch ref。

本计划提交本身不授权 cache/T0/GPU。执行授权必须逐数据线、逐阶段给出明确 commit、
manifest 和 receipt SHA；口头状态和 prefix SHA 不足以解锁。

2026-08-17 职责回执：07G 文档提交
`d1cee1916f8f2329b4e3f3a1140e79e5d22c124e` 已由 controller 本地独立核验；06G 文档
提交 `0c4fb7cbc4b4ecc4bc056027a011fa3bd1d29d27` 已由 06G 任务回传，但 controller 主机
无法解析 SSH host alias，因此使用本机已认证 GitHub 应用发布独立协调快照分支
`codex/iclr2027-dual-server-dispatch-publication`。首个 publication commit
`d3be2e8e7eda17f8c9afade2de5da4aa26f559f9` 已由 06G 通过普通 HTTPS fetch 获取；其
commit/parent/tree、receipt 5/5 哈希与两份 06G 文档逐字节比较全部通过。审计边界与原始
回执封存在 `controller_receipts/dual_server_dispatch_20260817/`。该通过只确认职责文件与
receipts 可达，不确认 9dab/P0 source release、数据门、训练代码或执行授权。

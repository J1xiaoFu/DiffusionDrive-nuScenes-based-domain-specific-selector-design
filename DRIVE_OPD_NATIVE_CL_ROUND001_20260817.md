# Drive-OPD Native CL — Round 001 (2026-08-17)

## 1. 本轮目标

把“OPD可能保留旧能力”的早期证据，迁移为可运行、可审计的智驾原生持续学习实验，而不是继续在已见过全量数据的 checkpoint 上调局部损失。具体目标是：

1. 建立会话原子化的 Failure-Patch-CL 与 Chronological-CL 数据协议；
2. 保持 DiffusionDrive 官方损失、感知与规划联合更新及单次 AdamW 更新，实现 Drive-OPD、同预算 LwF 和教师策略；
3. 接入可迁移的 EWC、A-GEM 与 ALER-Drive 基线；
4. 从零训练只见第一批通用日志的共享阶段起点 `T0`，最终测试集保持封存；
5. 完成数学审计、强制网络文献检索、成本核算和下一轮门控。

## 2. 预期结果与解释

- 数据协议应无 session、log、token 泄漏；安全修补批次应在 NC/TTC/DAC 风险上明显高于通用批次，效率修补批次应由“安全通过但 progress/comfort 较弱”的场景组成。
- 同一 checkpoint 初始化时，OPD/LwF 的蒸馏损失应接近零；一次更新后，两者只能因规划查询状态分布不同而分化。
- OPD 与 LwF 必须具有相同的学生/教师去噪查询次数，否则不能把效果差异归因于学生分布指导。
- ALER 在相同学生/教师上不应从数值噪声构造伪对抗 latent；若发生，说明语言 latent 搜索直接迁移到驾驶生成状态存在不稳定性。
- `T0` 必须只见 Stage-1 数据且具备可用规划能力，才能成为 Stage-2 的合法旧教师。

## 3. 算法与实验工作流

### 3.1 数据协议

输入缓存含 14,951 个 token、1,192 个 segmented logs、162 个时间戳×车辆会话。完整 session 为不可分割的最小单元。每个阶段先分 session，再形成 70% train、10% audit、20% test；最终 test 在配置冻结前禁止使用。

Failure-Patch-CL 不采用“是否至少一次失败”的二分类，因为 160/162 个 session 至少含一次失败，无法形成有信息量的分层。替代规则为：

- session 安全风险最高 30% → Stage-2 safety patch；
- 剩余 session 中较安全的一半内，安全场景 progress/comfort 失败较高者 → Stage-3 efficiency patch；
- 其余 → Stage-1 general base。

产生的 train/audit/test token 数分别为：

| Stage | Train | Audit | Test |
|---|---:|---:|---:|
| General base | 4,370 | 624 | 1,258 |
| Safety patch | 2,335 | 343 | 665 |
| Efficiency patch | 3,739 | 545 | 1,072 |

协议内容 SHA256：`60427d6d94d6d283c072d98b748eb2f5c40a31a61fbc69a86a7ab22b17db948b`。测试确认 session/log/token 三层均无泄漏。

### 3.2 Drive-OPD 与 LwF

每个训练批次只执行一次 AdamW 更新：

\[
L=L_{\mathrm{NAVSIM}}+\lambda_p L_{\mathrm{perc}}+\lambda_oL_{\mathrm{plan}}.
\]

`L_NAVSIM` 完整保留官方轨迹、扩散、agent 分类/框和 BEV 语义损失；感知和规划参数均可更新。`L_perc` 使用高置信 teacher agent 的 Hungarian 结构匹配及置信掩码 BEV KL，不约束低置信、新出现或不确定区域。

OPD 从共享噪声开始，在学生响应上推进两步 DDIM 状态，并在每个学生状态查询固定教师。LwF 使用 GT 轨迹加共享噪声得到的外生状态。两臂共享教师、感知损失、batch、microbatch、时间步和查询实现，唯一受控差异是规划查询支持分布。

学生 rollout 已改为“学生响应同时用于蒸馏并 detach 后推进下一状态”，避免过去先 rollout、再重复查询造成的额外学生 forward。现在两臂每步均为 2 次学生和 2 次教师去噪查询。

### 3.3 感知因果审计

已实现四上下文响应审计：`(Tperc,Tplan)`、`(Sperc,Tplan)`、`(Tperc,Splan)`、`(Sperc,Splan)`。先比较同一状态上的 response RMSE、mode KL 和 top-1 变化；只有响应门显著，再运行昂贵 PDMS 交换评测。辅助上下文在官方 train-mode forward 之前以 eval 行为计算，避免当前 batch 的 BN running-stat 更新制造伪感知漂移。

### 3.4 迁移基线

- EWC：从阶段起点模型和旧日志 buffer 估计官方驾驶联合损失的经验对角 Fisher；估计过程使用 eval 行为，不修改 BN；Fisher 预处理成本单列。
- A-GEM：当前批次梯度与 1,024 个旧日志 token 缓冲区的参考梯度比较；仅当内积为负时投影，最后仍只执行一次 AdamW 更新。
- ALER-Drive：在 GT 加噪的规划状态附近进行一次有界 latent 上升，再做固定教师 response MSE + reverse mode KL 修复；教师查询预算与 OPD 匹配为 2。记录 searched latent 到真实加噪轨迹状态集合的最近距离。
- replay：step-matched replay 保持与顺序训练相同 token/优化步数；full-exposure replay 使用所有已见训练 token，作为高成本上界。

DER++ stored-response cache、O-LoRA 独立 PEFT 表和 TALR-Scene 仍待接入，因此当前不能宣称完整主表已实现。

## 4. 真实结果

### 4.1 协议分离度

| Stage | session safety failure mean | safe-scene efficiency failure mean | safe-scene progress mean |
|---|---:|---:|---:|
| General base | 0.2916 | 0.2921 | 0.8397 |
| Safety patch | 0.4742 | 0.3996 | 0.7820 |
| Efficiency patch | 0.2189 | 0.3417 | 0.8255 |

Stage-2 的安全失败显著更高；Stage-3 在保持较低总体安全失败的同时，安全通过场景的效率失败高于 general base。它们是根据既有业务指标形成的真实回流子集，而不是人为修改标签制造冲突。是否产生可重复的模型冲突，仍需 Stage-2/3 真实训练与审计评测确认。

### 4.2 OPD/LwF 公平与成本门

实现门使用一个已见全量 NAVSIM 数据的旧 checkpoint，仅验证代码、计算和数值不变量，不作为持续学习效果证据。batch=64，蒸馏 microbatch=8；排除首步 CUDA/数据预热后：

| Arm | seconds/step | relative to sequential | peak memory | student/teacher denoiser queries |
|---|---:|---:|---:|---:|
| Sequential | 0.856 | 1.00x | 20.76 GiB | 0 / 0 |
| Fixed OPD | 1.299 | 1.52x | 23.23 GiB | 2 / 2 |
| LwF | 1.317 | 1.54x | 23.23 GiB | 2 / 2 |

相同权重首步的 OPD 蒸馏总损失约 `1.35e-15`；LwF 约 `2.42e-8`。更新后 OPD 与 LwF 的感知项接近，但规划 mode KL 分化，说明当前实现确实把“查询支持分布”隔离为核心变量。

期间发现 NAVSIM 缓存 GT 轨迹为 float64、规划头为 float32，导致 LwF 首次 GPU 运行报 dtype 错误；已在外生状态入口对齐 planner device/dtype，并补充回归测试。修复后运行通过。

### 4.3 EWC、A-GEM 与 ALER 真实梯度门

- A-GEM 在真实 batch=64、旧日志 replay batch=8 上完成一次更新。首批新/旧梯度点积为 `719,479.3 > 0`，因此未投影；这正确表示该批次没有一阶冲突，而不是强制修改更新。
- EWC 从 16 个旧样本的工程 smoke Fisher 建立 547 个参数项；相同起点首步 penalty=0，第二步 penalty=`9.07e-6 > 0`，证明锚点与 Fisher 正则进入真实梯度图。该 16 样本 Fisher 不用于效果实验；正式实验使用预注册 1,024 样本。
- Stage-1 audit 通过后，正式 Stage-2 Fisher 已使用合法 `T0` 和预注册的 1,024 个旧样本完成：64 个 batch×16，仍为 547 个参数项，Fisher 非零元素数 58,250,487、元素和 30,535.84；文件 SHA256=`22b0cfb8cdcf3578285837d640ff666ce9f10a6864b8b2f7cf30e7404010adbd`，内部绑定的源 checkpoint SHA 与 `T0` 完全一致。该预处理成本与阶段训练成本分开报告。
- ALER 初版在相同学生/教师首步产生 `0.0137` 伪修复损失。审计定位为学生 query 处于 train/dropout、教师处于 eval，且消耗了官方 forward 的随机数。改为匹配 eval 行为后，首步修复损失降至约 `4.8e-8`，官方首步损失重新与其他臂对齐。随后又发现近零分歧梯度经单位归一化仍会产生固定 0.05 的任意 latent 位移；加入 `1e-6` 梯度有效门后，首步 active fraction=0、latent 位移=0，第二步模型真实分化后 active fraction=1、位移=0.05。这一现象直接支持“语言 latent 搜索不能未经物理/数值审计直接搬到驾驶生成状态”。

### 4.4 共享 T0

Stage-1 `T0` 从 scratch 训练，只使用 4,370 个 general-base train token，batch=64、AdamW、LR=`1.5e-4`、5 epoch warmup、180 epoch cosine schedule；每 epoch 69 step，共完成 30 epoch、2,070 次更新。滚动断点含模型、优化器和 scheduler，阶段终点使用 hardlink，避免双份磁盘占用。官方训练损失 epoch mean 从 239.69（epoch 1）降至 76.69（epoch 30），无 NaN/Inf，单步约 0.82 秒，峰值显存约 20.1 GiB。阶段终点 SHA256 为 `b25ff2499579065f1c4fb921a91220d3681641047798fd162c66bf6e845ced78`。

原始官方 metric cache 归属 `rguo` 且权限为 `600`，第一次 624-token 评测因此在模型推理前全部失败；该失败不计为模型结果。随后使用官方 `MetricCacheProcessor` 在 `/home/khwang` 对精确的 28 个 audit 日志、624 个 audit token 重建缓存。元数据 SHA256 为 `c7f2ec2cc57521d16d0be5b0fe11a0286409f5bc333f0a8559e0597e4352a439`，缓存 token 集与协议 token 集严格相等。

合法本地缓存上的官方 PDM 评测为 624/624 成功、0 失败。token 平均 PDMS=`0.712595`；以完整日志为聚类单位的平均值为 `0.636988`，10,000 次配对 bootstrap 的 95% CI 为 `[0.572809, 0.699781]`。token 均值分项为 NC=`0.938301`、DAC=`0.841346`、TTC=`0.838141`、comfort=`0.899038`、progress=`0.688704`。该结果只来自 Stage-1 audit，最终 test 未开启。它证明只见 Stage-1 的 `T0` 已具备可用规划能力，可作为 Stage-2 合法固定教师；它尚不证明任何持续学习方法有效。

### 4.5 06G 最小迁移

已为 06G `AutoDrive` 工作区创建独立执行任务，并冻结只含 17 个源码、配置和测试文件的迁移分支 `codex/drive-opd-migration-20260817`。远端 commit 为 `49629a873aae5d186aad3f2514b93fc1f85791e0`，tree SHA 为 `1906286bc1a910c6292a9884bb243d7c7aeaefaf`；本机 `main` 和未提交工作树未切换、未覆盖。首批远端范围只包括 sequential、LwF、fixed OPD 与 EMA(0.99)-OPD，权重、优化器、数据和特征缓存留在生成服务器。

06G 初次盘点时观察到的 NAVSIM 缺失日志与模态计数异常，已由用户确认是 `navtrain` **仍在解压**的瞬态结果，不构成数据质量或覆盖率证据。解压完成后，远端已从 `/home/xlxia/datasets/navsim-navtrain/env.sh` 指向的最终路径重新审计：该脚本只有 5 个固定路径 `export`，无外部命令和文件/进程副作用；官方 navtrain 白名单含 1,192 个日志、103,288 个 token，官方四帧历史窗口所需的九种传感器集合零缺失。九模态各有 152,495 个文件，共 1,372,455 个文件；额外 118 个标注日志全部不在官方 navtrain 白名单中，不能再称为缺失数据。

远端轻量收尾已完成：迁移清单 16/16 个文件 SHA256 匹配；另行新增的 nuScenes V1 duck-typed adapter 在 `CUDA_VISIBLE_DEVICES=''` 下通过 18 个 CPU 合同测试，覆盖顺序损失等价、相同教师/学生的近零蒸馏、LwF/OPD 查询预算相等、单次 AdamW 更新和优化器恢复。该证据只验证接口合同，不是 V1SparseDrive/NAVSIM 的真实模型证据。远端 manifest、CPU 报告及 checkpoint 清单分别位于 `/home/xlxia/drive_opd_remote_20260817T150443_06g/experiment_manifest.json`、`reports/cpu_adapter_gate_report.json` 和 `checkpoint_sha256.json`。

最终原始数据门已通过，但 `NAVSIM_EXP_ROOT` 尚无 training/metric cache，环境路径中的 devkit 只有 TransFuser、无 DiffusionDrive agent，且没有兼容依赖环境或合法 Stage-1 教师，因此第一次真实 1-batch/GPU 门正确保持未运行。用户随后授权使用 06G 空闲资源，本轮已冻结官方 NAVSIM DiffusionDrive main commit `9b52ed0ec06b073d82d6f392ab084c7b301c8681` 作为 06G 干净复验源，而不是迁移 07G 带 RAP/Waymo/rendered-camera 改动的 agent。便携训练 manifest 已发布到迁移分支 commit `2cf57cfee30757f14ba5e363bbe9c51d309740ab`，文件 SHA256=`fada48c55b4c8b0234c44f82b16d718f1ae8e0b9d3ea8dcfb744b1bbe8c25b46`；它只包含三个 train split 的 4,370/2,335/3,739 个 token（并集 10,444），明确排除 audit/test。06G 正在创建独立官方环境与协议专用 cache，门通过后从 scratch 重训合法 `T0`。

## 5. 数学审计

### 5.1 OPD 与 LwF 的可辨识差异

令 teacher/student 差异为 `d(x,c)=fS(x,c)-fT(x,c)`。LwF 最小化外生支持 `q_ext` 上的 `E||d||²`，而 OPD 最小化学生生成支持 `qS` 上的同一量。若 `d` 对状态是 L-Lipschitz，则可写出支持转移误差界：

\[
|E_{q_S}\ell(d)-E_{q_{ext}}\ell(d)|\le L_\ell W_1(q_S,q_{ext}).
\]

因此学生—教师分布偏移较小时，LwF 可能与 OPD 相当；偏移较大时，外生状态保持并不直接约束部署时学生会访问的状态。主实验必须在高 `W1` 代理子集上比较，而不能只看全局平均 PDMS。

### 5.2 感知上下文偏差

规划差异可分解为：

\[
f_S(x,c_S)-f_T(x,c_T)=
[f_S(x,c_S)-f_T(x,c_S)] + [f_T(x,c_S)-f_T(x,c_T)].
\]

第一项是同上下文规划器漂移，第二项是感知上下文变化带来的 teacher response 偏差。只做规划 OPD 无法区分两者；四上下文互换用于估计第二项的因果贡献。若第二项不显著，不能把 OPD 失败归因于感知漂移。

### 5.3 A-GEM 与 EWC

A-GEM 在 `g_new·g_old<0` 时使用：

\[
g'=g_{new}-\frac{g_{new}^Tg_{old}}{\|g_{old}\|^2}g_{old},
\]

从而在一阶近似下使旧 buffer 损失不增加。它只保护 buffer 的当前局部方向，不能保证 NC/TTC 等闭环指标不下降。

EWC 的二次项是旧损失在阶段起点附近的局部对角曲率近似。首步 penalty 必为零，第二步才应非零；本轮真实梯度结果符合这一不变量。由于官方联合损失各项尺度不同，EWC lambda 必须在 audit split 上做预注册的对数尺度选择。

### 5.4 ALER 数值审计

当 teacher/student 相同时，理论分歧梯度为零。把浮点噪声梯度直接单位归一化会将任意微扰放大为固定半径搜索，这在自由语言 embedding 中已危险，在具有扩散状态几何的驾驶轨迹中更无法解释。加入梯度有效门只是消除数值伪搜索；它不能证明 adversarial state 位于真实驾驶流形，因此仍必须报告最近真实状态距离及其对真实日志退化的预测能力。

## 6. 强制网络文献检索闭环

- [CLAD](https://arxiv.org/abs/2210.03482) 支持自然 chronological stream 的必要性，但其任务集中在分类/检测，不能替代端到端规划、生成支持和闭环安全协议。
- [DER/DER++](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) 面向模糊任务边界的 General CL，并组合 replay 与历史 logits；其驾驶对应应是旧完整日志 + 当时结构化感知/轨迹响应，而不是只保存教师轨迹。
- [A-GEM](https://openreview.net/pdf/68b34388f383ea8919ffc158ee58a351da811add.pdf) 给出小型 episodic memory 的平均参考梯度约束；它能迁移到连续驾驶损失，但需要额外旧日志 forward，且一阶参数约束不等于闭环安全保证。
- [EWC](https://doi.org/10.1073/PNAS.1611835114) 通过参数重要性加权的局部二次约束保护旧能力，架构无关，因而是必要正则基线。
- [O-LoRA](https://aclanthology.org/2023.findings-emnlp.715/) 将不同任务置于正交低秩子空间；驾驶无推理 task ID，因此只能采用累加式、task-ID-free 版本并在独立 PEFT 参数预算表中比较。
- [LAMOL](https://iclr.cc/virtual_2020/poster_Skgxcn4YDS.html) 与 [SSR](https://aclanthology.org/2024.acl-long.77/) 的核心是假设模型能生成完整旧输入—输出训练实例。规划器只能生成轨迹，不能生成与之物理一致的多相机/LiDAR/参与者交互，因此不制造伪“生成回放”基线。
- [Progressive Prompts](https://arxiv.org/abs/2301.12314) 冻结主模型并按显式任务持续增加 prompt；这与无 task ID、感知—规划联合适应的单一驾驶策略不一致。
- [ALER](https://openreview.net/pdf?id=3CLOFiyWLU) 使用固定参考模型和 adversarial latent prompt；本轮真实门控已经显示 mode 匹配与近零梯度归一化可制造伪风险点，物理流形审计不可省略。
- [GoalFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_GoalFlow_Goal-Driven_Flow_Matching_for_Multimodal_Trajectories_Generation_in_End-to-End_CVPR_2025_paper.html) 是独立的端到端 Flow Matching 规划模型，并报告单步生成；因此后续应在其原生速度场上复验，而不是把 DiffusionDrive 强改成未经验证的 FM 架构。

## 7. 证据边界

当前已经证明：协议无泄漏；Drive-OPD/LwF 的核心损失进入真实 DiffusionDrive 梯度；查询预算匹配；成本约 1.52–1.54x；EWC/A-GEM/ALER 的关键运算能在真实 GPU batch 上执行；只见 Stage-1 的合法 `T0` 完成 30 epoch 训练并在 624-token audit 上得到有效 PDMS，因而可以进入 Stage-2。

当前尚未证明：Drive-OPD 优于 LwF、ALER 或 replay；感知漂移是失败因果；Failure-Patch 三阶段产生稳定遗忘；EMA 教师优于固定教师；方法可跨 DiffusionDrive/GoalFlow；任何最终测试或三种子统计显著性。已见全量数据的工程 checkpoint 不能作为上述结论的证据。07G 当前可行性 runner 来自 RAP 工作树，其中 DiffusionDrive agent 含 Waymo/rendered-camera、camera-only 与特征蒸馏改动，不是 pristine 官方 NAVSIM DiffusionDrive；因此其 Stage-2 结果只能进入可行性/机制表，论文官方基线必须由正在 06G 冻结的官方 commit 独立复验。

## 8. 下一轮门控

1. 用合法 `T0` 生成正式 EWC Fisher，并复核 ALER 的真实流形距离与退化预测；
2. 接入 DER++ stored-response cache，再冻结 Seed-0 funnel 的方法预算；
3. Stage-2 先运行 sequential、step-matched replay、full-exposure replay、LwF、fixed OPD、ALER 与完整 Drive-OPD，并评测 Stage-1/2 audit；
4. 只有真实 Stage-2 结果表明冲突和方法差异存在，才继续 Stage-3、三种子和最终 test；
5. 完成四上下文响应门，只有响应显著才支付完整 PDMS swap 成本；
6. 等待用户确认 06G `navtrain` 解压完成并提供最终路径；随后完成数据类型核验，仅启动 1-batch/1-step 门，门通过才运行四臂 Seed-0 pilot；
7. 配置冻结后运行 Chronological-CL，检验方法是否压制正迁移，再进入 GoalFlow 速度场蒸馏复验。

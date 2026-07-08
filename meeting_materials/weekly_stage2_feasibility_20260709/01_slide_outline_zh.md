# 建议 PPT 页纲

## Slide 1: 本周问题与结论
- 问题：Stage2 selector-vs-random feedback 是否有可学习信号？
- 结论：有。r40 aggregate selector 在 4 个 random aggregate 对照上全部胜出。
- 关键数：selector L2=6.3099，random mean L2=6.7095。

## Slide 2: 实验设计为什么要换基座
- DiffusionDrive stage2 checkpoint 已在 nuScenes 上训练，继续微调会引入过拟合/泄漏风险。
- 因此使用 stage1 checkpoint：`DiffusionDrive/ckpts/sparsedrive_stage1.pth`。
- 验证重点是 selector 能否通过数据选择改善小规模微调，而不是利用已有 stage2 能力。

## Slide 3: 数据与评估设置
- nuScenes Trainval mini-scale controlled partial。
- Train: 40 scenes / 1611 samples；Val: 10 scenes / 401 samples。
- DiffusionDrive 100-iter fine-tuning + standalone planning eval。
- 指标主看 L2，obj_box_col 作为安全参考。

## Slide 4: 全量微调学习率调优
- 放 `figures/fig01_full_lr_sweep_l2.png`。
- 结论：3e-5 是 full-data sweep 中最优 L2。
- 后续 selector/random 对比统一使用 planner lr=3e-5。

## Slide 5: Stage2 高频反馈机制
- 每轮 selector 选 8 条，random 选 8 条对照。
- DiffusionDrive 常驻 worker，两个 arm 都从 stage1 checkpoint reset。
- 40 轮后 selector 总共聚合 320 条样本。
- 这是高频 cheap proxy feedback；最终仍以真实 aggregate eval 定胜负。

## Slide 6: 阶段性表现
- 放 `figures/fig02_stage2_progression_l2.png`。
- r10/r20 还不稳定，r30 开始超过 random，r40 明显胜出。
- 说明信号需要积累，不能只看早期少量反馈。

## Slide 7: 最终四种 random seed 对比
- 放 `figures/fig03_r40_four_random_seeds.png`。
- Selector L2=6.3099。
- Random L2: 6.8750, 6.8181, 6.5795, 6.5652。
- Selector beat 4/4 random aggregate baselines。

## Slide 8: 消融与下一步
- `selector_lr=0.003` 消融失败：selector L2=7.2803，random L2=6.6052。
- 结论不是“任意 RL 更新都有效”，而是“存在可学习信号，更新方式需要调”。
- 下一步：reward shaping、候选退出/回放、更新强度、周期性真实验证、多 random seed。

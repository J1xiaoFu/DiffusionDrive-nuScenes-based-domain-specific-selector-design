# 周会 PPT 材料包：Stage2 Selector 可行性验证

生成时间：2026-07-08  
建议汇报时间：2026-07-09 下午周会

## 一句话结论

在 DiffusionDrive 架构 + stage1 checkpoint + nuScenes Trainval mini-scale controlled partial 上，
高频 selector-vs-random 反馈可以学习到有效选择策略：最终 r40 aggregate selector 的 L2 为 **6.3099**，
在 4 个 same-budget random aggregate 对照上全部胜出。

## 为什么这组实验是有效验证

- 使用 DiffusionDrive 原始架构，而不是替代模型。
- 使用 stage1 checkpoint 作为基座；避免使用已经在 nuScenes 上训练过的 stage2 checkpoint 继续微调造成过拟合/泄漏。
- 数据为 nuScenes Trainval 的 mini-scale controlled partial：40 个 train scenes / 1611 train samples，10 个 val scenes / 401 val samples。
- 先做 full-data 1000+ sample fine-tuning 学习率 sweep，确定 planner lr=3e-5 是合理工作点。
- 再做 Stage2 selector 强化反馈：40 次 8-sample selector-vs-random loss-duel，聚合 320 个 selector 样本做真实 100-iter DiffusionDrive 微调和验证。
- 最终在 4 个 random aggregate 对照上验证，不依赖单个随机种子。

## 最适合放进 PPT 的文件

- `tables/01_stage1_full_lr_sweep.csv`：全量数据学习率调优表。
- `tables/02_stage2_aggregate_progression.csv`：r10/r20/r30/r40 阶段进展。
- `tables/03_r40_four_random_seed_comparison.csv`：最终 4 个 random seed 对比。
- `figures/fig01_full_lr_sweep_l2.png`：学习率调优图。
- `figures/fig02_stage2_progression_l2.png`：Stage2 聚合进展图。
- `figures/fig03_r40_four_random_seeds.png`：最终四个随机种子对比图。
- `01_slide_outline_zh.md`：建议 PPT 页纲。

## 注意表述

建议说“nuScenes Trainval mini-scale controlled partial”，不要说已经验证了完整 nuScenes Trainval。
当前结论是可行性验证：Stage2 与 random 的比较确实存在可学习信号；后续还需要继续研究如何更稳定地利用该强化学习信号。

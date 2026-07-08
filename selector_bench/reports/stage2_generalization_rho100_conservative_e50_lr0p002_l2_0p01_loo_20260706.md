# Stage 2 Feedback Generalization Diagnostic

Diagnostic: `leave_one_episode_out`
Episode buffer: `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_feedback/partial_trainval_stage1_cluster_domain_vs_random_rho_box100_20260706.jsonl`
Init selector: `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selectors/stage1_cluster_domain_neural_v0_normfix_trainval_partial_teacher_100iter_20260706.json`
Fold work dir: `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706`

## Summary

| metric | value |
| --- | ---: |
| folds | 8 |
| preference accuracy | 0.375000 |
| positive reward accuracy | 0.750000 |
| negative reward accuracy | 0.000000 |
| wrong sign folds | 5 |
| mean signed margin | -0.424944 |
| min signed margin | -1.152354 |

## Folds

| fold | heldout | reward | selector | random | predicted gap | signed margin | correct | checkpoint |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| 0 | partial_trainval_stage1_cluster_domain_vs_random_100iter_20260706_rho_box100 | -1.204067 | 9889c8fa4af1 | 5d1145d00918 | 0.923480 | -0.923480 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_00/selector.json` |
| 1 | partial_trainval_stage2_feedback_vs_random_100iter_20260706_rho_box100 | 1.215359 | 7614af31cc86 | 5d1145d00918 | 0.197361 | 0.197361 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_01/selector.json` |
| 2 | partial_trainval_stage2_feedback_v1_vs_random_100iter_20260706_rho_box100 | 2.146896 | 7b5d7fa56539 | 5d1145d00918 | 0.068949 | 0.068949 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_02/selector.json` |
| 3 | partial_trainval_stage2_feedback_v2_vs_random_seed2026070602_100iter_20260706_rho_box100 | -1.258546 | b3d4e2cac19b | 507b31247cb2 | 0.923538 | -0.923538 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_03/selector.json` |
| 4 | partial_trainval_stage2_feedback_v3_rho_box100_vs_random_seed2026070603_100iter_20260706_rho_box100 | 1.176412 | c2758d5ba469 | f897d2bd9d2f | 0.428086 | 0.428086 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_04/selector.json` |
| 5 | partial_trainval_stage2_feedback_v4_rho_box100_vs_random_seed2026070604_100iter_20260706_rho_box100 | 1.342428 | 709895840892 | 1bba02582e94 | -0.271082 | -0.271082 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_05/selector.json` |
| 6 | partial_trainval_stage2_feedback_v5_rho_box100_vs_random_seed2026070605_100iter_20260706_rho_box100 | -3.018055 | d3cfca56a22b | ef49a4d982f2 | 0.823494 | -0.823494 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_06/selector.json` |
| 7 | partial_trainval_stage2_feedback_v6_rho_box100_vs_random_seed2026070606_100iter_20260706_rho_box100 | -1.804786 | d6306e617e92 | 8acd111dcdf3 | 1.152354 | -1.152354 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_rho100_conservative_e50_lr0p002_l2_0p01_loo_20260706/fold_07/selector.json` |

# Stage 2 Feedback Generalization Diagnostic

Diagnostic: `leave_one_episode_out`
Episode buffer: `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_feedback/partial_trainval_stage1_cluster_domain_vs_random_20260706.jsonl`
Init selector: `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selectors/stage1_cluster_domain_neural_v0_normfix_trainval_partial_teacher_100iter_20260706.json`
Fold work dir: `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706`

## Summary

| metric | value |
| --- | ---: |
| folds | 8 |
| preference accuracy | 0.500000 |
| positive reward accuracy | 1.000000 |
| negative reward accuracy | 0.000000 |
| wrong sign folds | 4 |
| mean signed margin | -0.259617 |
| min signed margin | -2.332154 |

## Folds

| fold | heldout | reward | selector | random | predicted gap | signed margin | correct | checkpoint |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| 0 | partial_trainval_stage1_cluster_domain_vs_random_100iter_20260706 | -0.731837 | 9889c8fa4af1 | 5d1145d00918 | 1.677718 | -1.677718 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_00/selector.json` |
| 1 | partial_trainval_stage2_feedback_vs_random_100iter_20260706 | 2.941442 | 7614af31cc86 | 5d1145d00918 | 1.030275 | 1.030275 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_01/selector.json` |
| 2 | partial_trainval_stage2_feedback_v1_vs_random_100iter_20260706 | 2.863383 | 7b5d7fa56539 | 5d1145d00918 | 0.172428 | 0.172428 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_02/selector.json` |
| 3 | partial_trainval_stage2_feedback_v2_vs_random_seed2026070602_100iter_20260706 | -0.305944 | b3d4e2cac19b | 507b31247cb2 | 1.550674 | -1.550674 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_03/selector.json` |
| 4 | partial_trainval_stage2_feedback_v3_rho_box100_vs_random_seed2026070603_100iter_20260706 | 1.404386 | c2758d5ba469 | f897d2bd9d2f | 2.724655 | 2.724655 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_04/selector.json` |
| 5 | partial_trainval_stage2_feedback_v4_rho_box100_vs_random_seed2026070604_100iter_20260706 | 1.114455 | 709895840892 | 1bba02582e94 | 0.944212 | 0.944212 | True | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_05/selector.json` |
| 6 | partial_trainval_stage2_feedback_v5_rho_box100_vs_random_seed2026070605_100iter_20260706 | -2.684237 | d3cfca56a22b | ef49a4d982f2 | 1.387957 | -1.387957 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_06/selector.json` |
| 7 | partial_trainval_stage2_feedback_v6_rho_box100_vs_random_seed2026070606_100iter_20260706 | -0.876609 | d6306e617e92 | 8acd111dcdf3 | 2.332154 | -2.332154 | False | `/root/autodl-tmp/domain_selector/selector_bench/work_dirs/stage2_generalization_primary_negcal_p1_loo_20260706/fold_07/selector.json` |

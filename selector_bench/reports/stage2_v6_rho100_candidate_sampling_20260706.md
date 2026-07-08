# Stage 2 v6 rho100 Candidate Sampling

Selector: `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selectors/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_20260706.json`

Ranking rule: minimize negative replay overlap, then v5-negative overlap, then maximize selected clean score mean.

| rank | seed | penalty | hash | neg_overlap | v5_overlap | v4_overlap | score_mean | score_min | pair_overlap_mean | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 47 | 1.0 | d6306e617e92 | 56 | 21 | 91 | 2.389571 | 0.521236 | 297.6 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed47_replay1p0_20260706.json` |
| 2 | 45 | 1.0 | 94a3bc788e24 | 57 | 21 | 90 | 2.394959 | 0.493702 | 299.0 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed45_replay1p0_20260706.json` |
| 3 | 49 | 1.0 | 06d9710c4d5a | 58 | 20 | 90 | 2.388411 | 0.535188 | 295.2 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed49_replay1p0_20260706.json` |
| 4 | 48 | 1.0 | 6f5c220fc2fe | 60 | 20 | 91 | 2.391491 | 0.187384 | 298.4 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed48_replay1p0_20260706.json` |
| 5 | 44 | 1.0 | 89eb8491e3de | 60 | 22 | 90 | 2.395323 | 0.285966 | 300.6 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed44_replay1p0_20260706.json` |
| 6 | 46 | 1.0 | 4494fb7044bd | 60 | 22 | 90 | 2.391965 | 0.177313 | 300.0 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed46_replay1p0_20260706.json` |
| 7 | 45 | 0.5 | 1ba251d97dac | 84 | 28 | 92 | 2.504571 | 0.798344 | 298.2 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed45_replay0p5_20260706.json` |
| 8 | 49 | 0.5 | 64fbc1d2155f | 84 | 30 | 91 | 2.501801 | 0.603840 | 296.6 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed49_replay0p5_20260706.json` |
| 9 | 48 | 0.5 | 7ed25b796b56 | 84 | 31 | 98 | 2.503746 | 0.187384 | 299.0 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed48_replay0p5_20260706.json` |
| 10 | 47 | 0.5 | 247de2679806 | 85 | 30 | 91 | 2.499558 | 0.239338 | 296.2 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed47_replay0p5_20260706.json` |
| 11 | 46 | 0.5 | 2610c3725dee | 87 | 30 | 92 | 2.499114 | 0.569292 | 297.4 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed46_replay0p5_20260706.json` |
| 12 | 44 | 0.5 | 95c6e34a2ece | 88 | 28 | 91 | 2.500341 | -0.342869 | 300.6 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed44_replay0p5_20260706.json` |
| 13 | 47 | 0.2 | 40a14597c177 | 101 | 36 | 94 | 2.584096 | 0.539338 | 296.8 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed47_replay0p2_20260706.json` |
| 14 | 45 | 0.2 | 11fa04a170d6 | 102 | 34 | 94 | 2.591316 | 0.738841 | 298.2 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed45_replay0p2_20260706.json` |
| 15 | 49 | 0.2 | 76b22c70c50e | 102 | 34 | 94 | 2.589137 | 0.671211 | 298.6 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed49_replay0p2_20260706.json` |
| 16 | 48 | 0.2 | f2a5a83172fd | 102 | 37 | 97 | 2.591425 | 0.187384 | 298.6 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed48_replay0p2_20260706.json` |
| 17 | 44 | 0.2 | 9f2ff2e05358 | 103 | 35 | 92 | 2.584052 | -0.042869 | 297.2 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed44_replay0p2_20260706.json` |
| 18 | 46 | 0.2 | e729e0235d57 | 104 | 35 | 97 | 2.584552 | 0.800633 | 297.8 | `/root/autodl-tmp/domain_selector/selector_bench/artifacts/selections/stage2_feedback_partial_trainval_neural_v6_rho_box100_normfix_gumbel20_seed46_replay0p2_20260706.json` |

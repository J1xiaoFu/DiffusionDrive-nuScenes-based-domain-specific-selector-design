# Stage 2 Collision-Weight Sensitivity

Source: `/root/autodl-tmp/domain_selector/selector_bench/artifacts/stage2_feedback/partial_trainval_stage1_cluster_domain_vs_random_20260706.jsonl`

Collision metrics are stored as fractions. A +1 percentage point box-collision gap changes the scalar cost by `0.01 * rho_box`.

Decision: keep the comparable feedback buffer on `rho_box=0.05` for direct history tracking. Use separate reweighted buffers or reports for collision-aware `rho_box` values so reward scales are not silently mixed.

| episode | selector | dL2 sel-rand | dBox pp | break-even rho_box | R@0 | R@0.05 | R@10 | R@50 | R@100 | R@250 | R@500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| partial_trainval_stage1_cluster_domain_vs_random_100iter_20260706 | 9889c8fa4af1 | 0.731601 | 0.472 | n/a | -0.731601 | -0.731837 | -0.778847 | -0.967834 | -1.204067 | -1.912767 | -3.093933 |
| partial_trainval_stage2_feedback_vs_random_100iter_20260706 | 7614af31cc86 | -2.942306 | 1.727 | 170.376 | 2.942306 | 2.941442 | 2.769611 | 2.078832 | 1.215359 | -1.375061 | -5.692428 |
| partial_trainval_stage2_feedback_v1_vs_random_100iter_20260706 | 7b5d7fa56539 | -2.863742 | 0.717 | 399.492 | 2.863742 | 2.863383 | 2.792057 | 2.505319 | 2.146896 | 1.071627 | -0.720487 |
| partial_trainval_stage2_feedback_v2_vs_random_seed2026070602_100iter_20260706 | b3d4e2cac19b | 0.305467 | 0.953 | n/a | -0.305467 | -0.305944 | -0.400775 | -0.782007 | -1.258546 | -2.688165 | -5.070863 |
| partial_trainval_stage2_feedback_v3_rho_box100_vs_random_seed2026070603_100iter_20260706 | c2758d5ba469 | -1.404500 | 0.228 | 615.773 | 1.404500 | 1.404386 | 1.381691 | 1.290456 | 1.176412 | 0.834282 | 0.264063 |

# Day 11 Multi-Seed Analysis

## Goal

Repeat the fair 3-epoch comparison across multiple random seeds.

Setting:

- Seeds: 1, 2, 3, 4, 5
- Memory per class: 20
- Memory examples: 3,000
- Epochs per task: 3
- Replay baseline: frozen DistilBERT encoder plus replay classifier head
- Tuned MTM: fast = 0.0, medium = 0.9, slow = 0.1

## Per-seed results

|   seed | method            |   final_average_accuracy |   avg_forgetting_including_final |   avg_forgetting_excluding_final |   memory_examples |   epochs |
|-------:|:------------------|-------------------------:|---------------------------------:|---------------------------------:|------------------:|---------:|
|      1 | replay_frozen     |                   0.7776 |                           0.152  |                           0.1689 |              3000 |        3 |
|      1 | mtm_no_fast_90_10 |                   0.7891 |                           0.134  |                           0.1489 |              3000 |        3 |
|      2 | replay_frozen     |                   0.7724 |                           0.1573 |                           0.1748 |              3000 |        3 |
|      2 | mtm_no_fast_90_10 |                   0.7791 |                           0.1462 |                           0.1625 |              3000 |        3 |
|      3 | replay_frozen     |                   0.7789 |                           0.1573 |                           0.1748 |              3000 |        3 |
|      3 | mtm_no_fast_90_10 |                   0.7864 |                           0.1387 |                           0.1541 |              3000 |        3 |
|      4 | replay_frozen     |                   0.7871 |                           0.146  |                           0.1622 |              3000 |        3 |
|      4 | mtm_no_fast_90_10 |                   0.7891 |                           0.1282 |                           0.1425 |              3000 |        3 |
|      5 | replay_frozen     |                   0.794  |                           0.1407 |                           0.1563 |              3000 |        3 |
|      5 | mtm_no_fast_90_10 |                   0.7816 |                           0.1436 |                           0.1595 |              3000 |        3 |

## Aggregate summary

| method            | metric                         |   mean |     sd |     se |   ci95_low |   ci95_high |   n_seeds |
|:------------------|:-------------------------------|-------:|-------:|-------:|-----------:|------------:|----------:|
| mtm_no_fast_90_10 | final_average_accuracy         | 0.7851 | 0.0045 | 0.002  |     0.7794 |      0.7907 |         5 |
| mtm_no_fast_90_10 | avg_forgetting_including_final | 0.1381 | 0.0073 | 0.0032 |     0.1291 |      0.1471 |         5 |
| mtm_no_fast_90_10 | avg_forgetting_excluding_final | 0.1535 | 0.0081 | 0.0036 |     0.1435 |      0.1635 |         5 |
| replay_frozen     | final_average_accuracy         | 0.782  | 0.0085 | 0.0038 |     0.7714 |      0.7926 |         5 |
| replay_frozen     | avg_forgetting_including_final | 0.1507 | 0.0073 | 0.0033 |     0.1416 |      0.1597 |         5 |
| replay_frozen     | avg_forgetting_excluding_final | 0.1674 | 0.0081 | 0.0036 |     0.1574 |      0.1775 |         5 |

## Paired tests

| metric                         |   mean_replay |   mean_mtm |   mean_difference_mtm_minus_replay |   sd_difference |   se_difference |   ci95_low |   ci95_high |   paired_t_stat |   paired_t_p_value |   cohens_dz |
|:-------------------------------|--------------:|-----------:|-----------------------------------:|----------------:|----------------:|-----------:|------------:|----------------:|-------------------:|------------:|
| final_average_accuracy         |        0.782  |     0.7851 |                             0.0031 |          0.0093 |          0.0042 |    -0.0085 |      0.0146 |          0.7363 |             0.5024 |      0.3293 |
| avg_forgetting_excluding_final |        0.1674 |     0.1535 |                            -0.0139 |          0.0102 |          0.0045 |    -0.0265 |     -0.0013 |         -3.063  |             0.0375 |     -1.3698 |

## Main interpretation

Across five random seeds, tuned MTM achieved slightly higher mean final average accuracy than replay:

- Replay mean final average accuracy: 0.7820
- MTM mean final average accuracy: 0.7851
- Mean difference, MTM minus replay: 0.0031
- 95% CI: [-0.0085, 0.0146]
- Paired t-test p-value: 0.5024

The accuracy difference favors MTM on average, but the confidence interval includes zero, so this accuracy improvement should be interpreted cautiously.

For forgetting, tuned MTM achieved lower mean average forgetting excluding the final task:

- Replay mean forgetting: 0.1674
- MTM mean forgetting: 0.1535
- Mean difference, MTM minus replay: -0.0139
- 95% CI: [-0.0265, -0.0013]
- Paired t-test p-value: 0.0375

A negative forgetting difference favors MTM. In this five-seed experiment, MTM reduced forgetting more consistently than it improved accuracy.

Because this analysis uses only five seeds, the results should still be interpreted cautiously. However, the multi-seed analysis supports the claim that tuned MTM is at least competitive with replay and may reduce forgetting more reliably.

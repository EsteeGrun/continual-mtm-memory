# Day 12 Multi-Seed Figures Report

## Goal

Visualize the Day 11 multi-seed replay vs tuned MTM comparison.

## Figures created

1. `mean_accuracy_ci.png`
2. `mean_forgetting_ci.png`
3. `paired_accuracy_by_seed.png`
4. `paired_forgetting_by_seed.png`
5. `accuracy_difference_by_seed.png`
6. `forgetting_difference_by_seed.png`

## Statistical result

Final average accuracy:

- Replay mean: 0.7820
- MTM mean: 0.7851
- MTM minus replay: 0.0031
- 95% CI: [-0.0085, 0.0146]
- Paired t-test p-value: 0.5024

Average forgetting excluding final task:

- Replay mean: 0.1674
- MTM mean: 0.1535
- MTM minus replay: -0.0139
- 95% CI: [-0.0265, -0.0013]
- Paired t-test p-value: 0.0375

## Interpretation

The accuracy plot shows that tuned MTM has slightly higher mean final average accuracy than replay, but the confidence interval for the paired difference includes zero. Therefore, the accuracy improvement should be interpreted cautiously.

The forgetting plot shows a clearer result. Tuned MTM has lower mean forgetting than replay, and the paired difference confidence interval does not include zero. The paired t-test gives p = 0.0375, suggesting that MTM reduced forgetting significantly in this five-seed experiment.

The paired seed plots show how each seed changes from replay to MTM. These plots are useful because the experiment is paired: both methods were evaluated under the same seeds.

The difference plots show MTM minus replay for each seed. For accuracy, positive values favor MTM. For forgetting, negative values favor MTM.

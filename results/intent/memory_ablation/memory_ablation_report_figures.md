# Day 9 Memory-Budget Ablation Figures

## Goal

Visualize the Day 8 memory-budget ablation comparing replay and the best MTM configuration.

## Figures created

1. `final_accuracy_by_memory_budget.png`
2. `forgetting_by_memory_budget.png`
3. `accuracy_forgetting_tradeoff.png`
4. `memory_ablation_grouped_bars.png`

## Final average accuracy

|   memory_per_class |   mtm_no_fast_90_10 |   replay_frozen |
|-------------------:|--------------------:|----------------:|
|                  5 |              0.1044 |          0.1131 |
|                 10 |              0.286  |          0.2707 |
|                 20 |              0.5922 |          0.5722 |

## Average forgetting excluding final task

|   memory_per_class |   mtm_no_fast_90_10 |   replay_frozen |
|-------------------:|--------------------:|----------------:|
|                  5 |              0.781  |          0.8037 |
|                 10 |              0.6217 |          0.658  |
|                 20 |              0.2909 |          0.3133 |

## Interpretation

The memory-budget ablation shows that replay and MTM both improve as the replay memory budget increases.

At 5 examples per class, replay has slightly higher final average accuracy, while MTM has slightly lower forgetting.

At 10 and 20 examples per class, the tuned MTM configuration slightly outperforms replay in the controlled 1-epoch setting on both final average accuracy and forgetting.

This does not prove that MTM is universally better than replay. The earlier Day 4 replay experiment used 3 epochs and achieved higher performance. However, under a controlled 1-epoch setting, the tuned MTM prototype is competitive and sometimes better than replay.

The result supports the next research direction: improve multi-timescale memory through learned gating or better slow-memory consolidation.

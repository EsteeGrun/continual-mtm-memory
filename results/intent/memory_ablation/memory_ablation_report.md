# Day 8 Memory-Budget Ablation

## Goal

Compare replay and the best MTM configuration under different memory budgets.

The tested memory budgets were:

- 5 examples per class = 750 stored examples
- 10 examples per class = 1500 stored examples
- 20 examples per class = 3000 stored examples

All Day 8 runs used 1 epoch per task for a controlled comparison.

## Summary Table

| method            |   memory_per_class |   memory_examples |   final_average_accuracy |   avg_forgetting_including_final |   avg_forgetting_excluding_final |
|:------------------|-------------------:|------------------:|-------------------------:|---------------------------------:|---------------------------------:|
| mtm_no_fast_90_10 |                  5 |               750 |                   0.1044 |                           0.7029 |                           0.781  |
| replay_frozen     |                  5 |               750 |                   0.1131 |                           0.7233 |                           0.8037 |
| mtm_no_fast_90_10 |                 10 |              1500 |                   0.286  |                           0.5596 |                           0.6217 |
| replay_frozen     |                 10 |              1500 |                   0.2707 |                           0.5922 |                           0.658  |
| mtm_no_fast_90_10 |                 20 |              3000 |                   0.5922 |                           0.2618 |                           0.2909 |
| replay_frozen     |                 20 |              3000 |                   0.5722 |                           0.282  |                           0.3133 |

## Interpretation Template

Replay is expected to remain a strong baseline because it trains directly on stored examples.

The MTM configuration uses the best Day 7 fusion rule:

- fast weight = 0.0
- medium weight = 0.9
- slow weight = 0.1

The key question is whether MTM becomes more competitive when memory is limited.

If replay remains stronger at all budgets, then the current MTM prototype still needs better gating, better slow-memory training, or a smaller memory footprint advantage.

If MTM closes the gap at lower memory budgets, then multi-timescale memory may be useful under constrained memory conditions.

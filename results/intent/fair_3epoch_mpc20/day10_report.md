# Day 10 Fair 3-Epoch Comparison

## Goal

Compare replay and tuned MTM under the same memory and epoch setting.

Both methods used:

- 20 examples per class
- 3,000 stored examples
- 3 epochs per task

The tuned MTM configuration used:

- fast weight = 0.0
- medium weight = 0.9
- slow weight = 0.1

## Results

                    method  epochs  memory_examples  final_average_accuracy  avg_forgetting_including_final  avg_forgetting_excluding_final
    replay_frozen_mpc20_e3       3             3000                  0.7836                          0.1422                           0.158
mtm_no_fast_90_10_mpc20_e3       3             3000                  0.7931                          0.1296                           0.144

## Difference: MTM minus Replay

Final average accuracy difference: 0.0096

Average forgetting excluding final task difference: -0.0141

## Interpretation

MTM has higher final average accuracy than replay.

MTM has lower forgetting than replay.

This comparison is fairer than comparing the original 3-epoch replay result against the 1-epoch memory-budget ablation. However, MTM still trains multiple classifier heads, so this is matched by epoch count rather than exactly matched by compute.

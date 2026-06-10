# Continual Multi-Timescale Memory

This project evaluates multi-timescale memory for continual learning.

## Main research question

Can fast, medium, and slow memory components reduce catastrophic forgetting in continual intent classification and factual QA?

## Day 1 progress

- Created project repository.
- Created Python virtual environment.
- Installed core packages.
- Added dataset-loading script.
- Loaded CLINC150 / BANKING77 intent classification datasets.

## First experimental track

Continual intent classification.

Planned datasets:

- CLINC150
- BANKING77

Planned baselines:

- Sequential fine-tuning
- Replay
- LoRA continual tuning
- Multi-timescale memory model


## Day 3 result

Sequential frozen-encoder baseline completed.

Main finding:

The model learns each new CLINC150 task well immediately after training, with per-task accuracies around 90–96%. However, after training on subsequent tasks, accuracy on previous tasks collapses to approximately 0%, showing severe catastrophic forgetting.

Final average accuracy after Task 9:

- 8.98%

Average forgetting:

- Approximately 84.33% including the final task
- Approximately 93.7% excluding the final task

Interpretation:

This provides a strong baseline showing that single-timescale sequential classifier training is not sufficient for continual intent classification.

## Day 4 result

Replay baseline completed.

Main result:

- Sequential frozen baseline final average accuracy: 8.98%
- Replay frozen baseline final average accuracy: 78.36%
- Sequential frozen average forgetting excluding final task: 93.7%
- Replay frozen average forgetting excluding final task: 15.8%

Interpretation:

Replay substantially reduces catastrophic forgetting in continual CLINC150 intent classification. This gives a strong baseline for comparison against the future multi-timescale memory method.

## Day 5 progress

Created analysis and visualization pipeline for Day 3 and Day 4 results.

Generated:

- Final average accuracy comparison
- Average forgetting comparison
- Average seen accuracy over tasks
- Results matrix heatmaps
- Final task accuracy comparison
- Markdown analysis report

Main comparison:

- Sequential frozen final average accuracy: 8.98%
- Replay frozen final average accuracy: 78.36%
- Sequential frozen average forgetting excluding final task: 93.7%
- Replay frozen average forgetting excluding final task: 15.8%

Interpretation:

Replay strongly reduces catastrophic forgetting, but requires storing 3000 examples. This gives a baseline for the future multi-timescale memory method.

## Day 7 progress

Performed an inference-time weight sweep over the multi-timescale memory heads.

Goal:

Test whether the Day 6 MTM prototype underperformed because of the architecture itself or because of poor fixed fusion weights.

Tested configurations included:

- medium only
- medium-heavy combinations
- no-fast combinations
- balanced fast/medium/slow baseline from Day 6

Interpretation:

The Day 6 prototype showed that fast, medium, and slow heads behave differently. Day 7 tests which memory combination gives the best stability-plasticity tradeoff.

Output directory:

`results/intent/mtm_weight_sweep/`

Main comparison file:

`results/intent/day7_comparison.csv`

## Day 7 progress

Performed an inference-time weight sweep over the multi-timescale memory heads.

Goal:

Test whether the Day 6 MTM prototype underperformed because of the architecture itself or because of poor fixed fusion weights.

Tested configurations included:

- medium only
- medium-heavy combinations
- no-fast combinations
- balanced fast/medium/slow baseline from Day 6

Interpretation:

The Day 6 prototype showed that fast, medium, and slow heads behave differently. Day 7 tests which memory combination gives the best stability-plasticity tradeoff.

Output directory:

`results/intent/mtm_weight_sweep/`

Main comparison file:

`results/intent/day7_comparison.csv`

## Day 7 result

Performed an inference-time weight sweep over the multi-timescale memory heads.

Best MTM configuration:

- fast weight: 0.0
- medium weight: 0.9
- slow weight: 0.1

Results:

- MTM balanced v1 final average accuracy: 26.49%
- MTM balanced v1 average forgetting excluding final task: 68.05%
- Best MTM final average accuracy: 59.22%
- Best MTM average forgetting excluding final task: 29.09%

Interpretation:

The medium head is currently the strongest memory component. The slow head helps slightly when used carefully. The fast head hurts retention under fixed fusion because it resets each task and does not preserve old-task knowledge.

This suggests that future MTM models need learned gating or task-aware memory selection instead of fixed weighted averaging.

## Day 8 progress

Performed memory-budget ablation.

Compared:

- Replay frozen baseline
- Best MTM configuration from Day 7: no_fast_90_10

Memory budgets:

- 5 examples per class = 750 examples
- 10 examples per class = 1500 examples
- 20 examples per class = 3000 examples

All Day 8 runs used 1 epoch per task for a controlled comparison.

Output directory:

`results/intent/memory_ablation/`

Summary file:

`results/intent/memory_ablation/memory_ablation_summary.csv`

Research question:

Does multi-timescale memory become more competitive when replay memory is limited?

## Day 8 result

Memory-budget ablation completed.

Controlled setting:

- Replay and MTM both used 1 epoch per task.
- Memory budgets tested: 5, 10, and 20 examples per class.
- Best MTM configuration used: fast = 0.0, medium = 0.9, slow = 0.1.

Results:

At 5 examples per class:

- MTM final average accuracy: 10.44%
- Replay final average accuracy: 11.31%
- MTM forgetting excluding final task: 78.10%
- Replay forgetting excluding final task: 80.37%

At 10 examples per class:

- MTM final average accuracy: 28.60%
- Replay final average accuracy: 27.07%
- MTM forgetting excluding final task: 62.17%
- Replay forgetting excluding final task: 65.80%

At 20 examples per class:

- MTM final average accuracy: 59.22%
- Replay final average accuracy: 57.22%
- MTM forgetting excluding final task: 29.09%
- Replay forgetting excluding final task: 31.33%

Interpretation:

Under the controlled 1-epoch setting, tuned MTM is competitive with replay and slightly outperforms replay at 10 and 20 examples per class.

## Day 9 progress

Created plots for the Day 8 memory-budget ablation.

Generated figures:

- Final average accuracy by memory budget
- Average forgetting by memory budget
- Accuracy-forgetting tradeoff
- Grouped bar comparison of replay and MTM

Output directory:

`results/intent/memory_ablation/plots/`

Figure report:

`results/intent/memory_ablation/memory_ablation_report_figures.md`

Interpretation:

Both replay and MTM improve as memory budget increases. Under the controlled 1-epoch setting, the tuned MTM configuration is competitive with replay and slightly outperforms replay at 10 and 20 examples per class.

## Day 10 progress

Performed a fair 3-epoch comparison between replay and tuned MTM at the same memory budget.

Setting:

- 20 examples per class
- 3,000 stored examples
- 3 epochs per task

Compared:

- Replay frozen baseline
- Tuned MTM configuration from Day 7: no_fast_90_10

Output directory:

`results/intent/fair_3epoch_mpc20/`

Summary file:

`results/intent/fair_3epoch_mpc20/day10_comparison.csv`

Purpose:

This experiment checks whether tuned MTM remains competitive with replay when both methods use the same memory budget and the same number of epochs per task.

## Day 12 progress

Created plots for the Day 11 multi-seed statistical analysis.

Generated figures:

- Mean final accuracy with 95% confidence intervals
- Mean forgetting with 95% confidence intervals
- Paired final accuracy by seed
- Paired forgetting by seed
- Accuracy difference by seed
- Forgetting difference by seed

Output directory:

`results/intent/multiseed_3epoch_mpc20/plots/`

Figure report:

`results/intent/multiseed_3epoch_mpc20/day12_multiseed_figures_report.md`

Main interpretation:

Across five seeds, tuned MTM had slightly higher mean final accuracy than replay, but the accuracy difference was not statistically reliable. Tuned MTM significantly reduced average forgetting excluding the final task compared with replay.

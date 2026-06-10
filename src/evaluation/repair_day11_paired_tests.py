"""
Repair Day 11 paired tests.

This script uses the already-saved:
results/intent/multiseed_3epoch_mpc20/day11_per_seed_results.csv

It computes:
- paired differences
- paired t-tests
- 95% confidence intervals
- Cohen's dz
- final day11_report.md
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values, confidence=0.95):
    values = np.array(values, dtype=float)
    n = len(values)

    mean = values.mean()
    sd = values.std(ddof=1)
    se = sd / math.sqrt(n)

    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    return mean, sd, se, ci_low, ci_high


def paired_test(per_seed, metric):
    replay = (
        per_seed[per_seed["method"] == "replay_frozen"]
        .sort_values("seed")[metric]
        .to_numpy()
    )

    mtm = (
        per_seed[per_seed["method"] == "mtm_no_fast_90_10"]
        .sort_values("seed")[metric]
        .to_numpy()
    )

    difference = mtm - replay

    mean, sd, se, ci_low, ci_high = mean_ci(difference)

    t_stat, p_value = stats.ttest_rel(mtm, replay)

    cohens_dz = mean / sd if sd != 0 else np.nan

    return {
        "metric": metric,
        "mean_replay": replay.mean(),
        "mean_mtm": mtm.mean(),
        "mean_difference_mtm_minus_replay": mean,
        "sd_difference": sd,
        "se_difference": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_stat": t_stat,
        "paired_t_p_value": p_value,
        "cohens_dz": cohens_dz,
    }


def main():
    base = Path("results/intent/multiseed_3epoch_mpc20")

    per_seed_path = base / "day11_per_seed_results.csv"
    aggregate_path = base / "day11_aggregate_summary.csv"

    per_seed = pd.read_csv(per_seed_path)
    aggregate = pd.read_csv(aggregate_path)

    paired = pd.DataFrame(
        [
            paired_test(per_seed, "final_average_accuracy"),
            paired_test(per_seed, "avg_forgetting_excluding_final"),
        ]
    )

    paired_path = base / "day11_paired_tests.csv"
    paired.to_csv(paired_path, index=False)

    print("\nDAY 11 PAIRED TESTS")
    print(paired.round(4))

    acc = paired[paired["metric"] == "final_average_accuracy"].iloc[0]
    forget = paired[paired["metric"] == "avg_forgetting_excluding_final"].iloc[0]

    report = f"""# Day 11 Multi-Seed Analysis

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

{per_seed.round(4).to_markdown(index=False)}

## Aggregate summary

{aggregate.round(4).to_markdown(index=False)}

## Paired tests

{paired.round(4).to_markdown(index=False)}

## Main interpretation

Across five random seeds, tuned MTM achieved slightly higher mean final average accuracy than replay:

- Replay mean final average accuracy: {acc['mean_replay']:.4f}
- MTM mean final average accuracy: {acc['mean_mtm']:.4f}
- Mean difference, MTM minus replay: {acc['mean_difference_mtm_minus_replay']:.4f}
- 95% CI: [{acc['ci95_low']:.4f}, {acc['ci95_high']:.4f}]
- Paired t-test p-value: {acc['paired_t_p_value']:.4f}

The accuracy difference favors MTM on average, but the confidence interval includes zero, so this accuracy improvement should be interpreted cautiously.

For forgetting, tuned MTM achieved lower mean average forgetting excluding the final task:

- Replay mean forgetting: {forget['mean_replay']:.4f}
- MTM mean forgetting: {forget['mean_mtm']:.4f}
- Mean difference, MTM minus replay: {forget['mean_difference_mtm_minus_replay']:.4f}
- 95% CI: [{forget['ci95_low']:.4f}, {forget['ci95_high']:.4f}]
- Paired t-test p-value: {forget['paired_t_p_value']:.4f}

A negative forgetting difference favors MTM. In this five-seed experiment, MTM reduced forgetting more consistently than it improved accuracy.

Because this analysis uses only five seeds, the results should still be interpreted cautiously. However, the multi-seed analysis supports the claim that tuned MTM is at least competitive with replay and may reduce forgetting more reliably.
"""

    report_path = base / "day11_report.md"
    report_path.write_text(report)

    print(f"\nSaved paired tests to: {paired_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

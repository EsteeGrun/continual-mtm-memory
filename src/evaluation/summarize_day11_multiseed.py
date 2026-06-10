"""
Day 11 summary.

Goal:
Summarize replay vs tuned MTM across multiple random seeds.

Setting:
- memory_per_class = 20
- memory_examples = 3000
- epochs = 3
- seeds = 1, 2, 3, 4, 5

Run:

    python src/evaluation/summarize_day11_multiseed.py
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats


def summarize_one_run(seed, method, folder):
    folder = Path(folder)

    results_path = folder / "results_matrix.csv"
    forgetting_path = folder / "forgetting.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    if not forgetting_path.exists():
        raise FileNotFoundError(f"Missing forgetting file: {forgetting_path}")

    results = pd.read_csv(results_path, index_col=0)
    forgetting = pd.read_csv(forgetting_path)

    final_average_accuracy = results.iloc[-1].dropna().mean()
    last_task = results.columns[-1]

    avg_forgetting_including_final = forgetting["forgetting"].mean()
    avg_forgetting_excluding_final = forgetting[
        forgetting["task"] != last_task
    ]["forgetting"].mean()

    return {
        "seed": seed,
        "method": method,
        "final_average_accuracy": final_average_accuracy,
        "avg_forgetting_including_final": avg_forgetting_including_final,
        "avg_forgetting_excluding_final": avg_forgetting_excluding_final,
        "memory_examples": 3000,
        "epochs": 3,
    }


def mean_ci(series, confidence=0.95):
    values = np.array(series, dtype=float)
    n = len(values)

    mean = values.mean()
    sd = values.std(ddof=1)

    if n <= 1:
        return mean, sd, np.nan, np.nan, np.nan

    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    return mean, sd, se, ci_low, ci_high


def paired_stats(paired_df, metric, difference_name):
    replay = paired_df[f"{metric}_replay"]
    mtm = paired_df[f"{metric}_mtm"]

    difference = mtm - replay

    mean, sd, se, ci_low, ci_high = mean_ci(difference)

    t_stat, p_value = stats.ttest_rel(mtm, replay)

    # Cohen's dz for paired samples: mean difference / sd difference
    cohens_dz = mean / sd if sd != 0 else np.nan

    return {
        "difference": difference_name,
        "metric": metric,
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
    seeds = [1, 2, 3, 4, 5]

    rows = []

    for seed in seeds:
        seed_dir = base / f"seed_{seed}"

        rows.append(
            summarize_one_run(
                seed=seed,
                method="replay_frozen",
                folder=seed_dir / "replay_mpc20_e3",
            )
        )

        rows.append(
            summarize_one_run(
                seed=seed,
                method="mtm_no_fast_90_10",
                folder=seed_dir / "mtm_mpc20_e3",
            )
        )

    per_seed = pd.DataFrame(rows)

    per_seed_path = base / "day11_per_seed_results.csv"
    per_seed.to_csv(per_seed_path, index=False)

    print("\nDAY 11 PER-SEED RESULTS")
    print(per_seed.round(4))

    aggregate_rows = []

    for method, method_df in per_seed.groupby("method"):
        for metric in [
            "final_average_accuracy",
            "avg_forgetting_including_final",
            "avg_forgetting_excluding_final",
        ]:
            mean, sd, se, ci_low, ci_high = mean_ci(method_df[metric])

            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "se": se,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_seeds": len(method_df),
                }
            )

    aggregate = pd.DataFrame(aggregate_rows)

    aggregate_path = base / "day11_aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print("\nDAY 11 AGGREGATE SUMMARY")
    print(aggregate.round(4))

    wide = per_seed.pivot(
        index="seed",
        columns="method",
        values=[
            "final_average_accuracy",
            "avg_forgetting_including_final",
            "avg_forgetting_excluding_final",
        ],
    )

    wide.columns = [
        f"{metric}_{method}"
        for metric, method in wide.columns
    ]

    wide = wide.reset_index()

    paired_rows = [
        paired_stats(
            paired_df=wide,
            metric="final_average_accuracy",
            difference_name="accuracy_mtm_minus_replay",
        ),
        paired_stats(
            paired_df=wide,
            metric="avg_forgetting_excluding_final",
            difference_name="forgetting_mtm_minus_replay",
        ),
    ]

    paired = pd.DataFrame(paired_rows)

    paired_path = base / "day11_paired_tests.csv"
    paired.to_csv(paired_path, index=False)

    print("\nDAY 11 PAIRED TESTS")
    print(paired.round(4))

    accuracy_diff = paired[paired["metric"] == "final_average_accuracy"].iloc[0]
    forgetting_diff = paired[paired["metric"] == "avg_forgetting_excluding_final"].iloc[0]

    report = f"""# Day 11 Multi-Seed Analysis

## Goal

Repeat the fair 3-epoch comparison across multiple random seeds.

Setting:

- Seeds: {seeds}
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

Final average accuracy, MTM minus replay:

Mean difference: {accuracy_diff['mean_difference_mtm_minus_replay']:.4f}

95% CI: [{accuracy_diff['ci95_low']:.4f}, {accuracy_diff['ci95_high']:.4f}]

Paired t-test p-value: {accuracy_diff['paired_t_p_value']:.4f}

Average forgetting excluding final task, MTM minus replay:

Mean difference: {forgetting_diff['mean_difference_mtm_minus_replay']:.4f}

95% CI: [{forgetting_diff['ci95_low']:.4f}, {forgetting_diff['ci95_high']:.4f}]

Paired t-test p-value: {forgetting_diff['paired_t_p_value']:.4f}

A positive accuracy difference favors MTM. A negative forgetting difference favors MTM.

Because this uses only five seeds, the statistical results should be interpreted cautiously. The main goal is to check whether the Day 10 result is stable across random seeds.
"""

    report_path = base / "day11_report.md"
    report_path.write_text(report)

    print(f"\nSaved per-seed results to: {per_seed_path}")
    print(f"Saved aggregate summary to: {aggregate_path}")
    print(f"Saved paired tests to: {paired_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

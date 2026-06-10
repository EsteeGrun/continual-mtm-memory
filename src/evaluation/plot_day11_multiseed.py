"""
Day 12 script.

Goal:
Create plots for the Day 11 multi-seed replay vs MTM comparison.

Input:
    results/intent/multiseed_3epoch_mpc20/day11_per_seed_results.csv
    results/intent/multiseed_3epoch_mpc20/day11_aggregate_summary.csv
    results/intent/multiseed_3epoch_mpc20/day11_paired_tests.csv

Output:
    results/intent/multiseed_3epoch_mpc20/plots/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def method_label(method):
    if method == "replay_frozen":
        return "Replay"
    if method == "mtm_no_fast_90_10":
        return "Tuned MTM"
    return method


def get_metric_summary(aggregate, metric):
    metric_df = aggregate[aggregate["metric"] == metric].copy()
    metric_df["label"] = metric_df["method"].apply(method_label)
    return metric_df


def plot_mean_metric_with_ci(aggregate, metric, title, ylabel, output_path):
    metric_df = get_metric_summary(aggregate, metric)

    preferred_order = ["replay_frozen", "mtm_no_fast_90_10"]
    metric_df["order"] = metric_df["method"].apply(lambda x: preferred_order.index(x))
    metric_df = metric_df.sort_values("order")

    labels = metric_df["label"].tolist()
    means = metric_df["mean"].to_numpy()
    ci_low = metric_df["ci95_low"].to_numpy()
    ci_high = metric_df["ci95_high"].to_numpy()

    lower_errors = means - ci_low
    upper_errors = ci_high - means
    yerr = np.vstack([lower_errors, upper_errors])

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, means, yerr=yerr, capsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_wide(per_seed, metric):
    wide = per_seed.pivot(
        index="seed",
        columns="method",
        values=metric,
    ).reset_index()

    wide = wide.sort_values("seed")
    return wide


def plot_paired_by_seed(per_seed, metric, title, ylabel, output_path):
    wide = make_wide(per_seed, metric)

    fig, ax = plt.subplots(figsize=(8, 5))

    x_replay = 0
    x_mtm = 1

    for _, row in wide.iterrows():
        y_values = [
            row["replay_frozen"],
            row["mtm_no_fast_90_10"],
        ]

        ax.plot(
            [x_replay, x_mtm],
            y_values,
            marker="o",
        )

        ax.text(
            x_mtm + 0.03,
            row["mtm_no_fast_90_10"],
            f"seed {int(row['seed'])}",
            va="center",
            fontsize=8,
        )

    ax.set_xticks([x_replay, x_mtm])
    ax.set_xticklabels(["Replay", "Tuned MTM"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_difference_by_seed(per_seed, metric, title, ylabel, output_path, zero_line_label):
    wide = make_wide(per_seed, metric)

    wide["difference"] = wide["mtm_no_fast_90_10"] - wide["replay_frozen"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        wide["seed"].astype(str),
        wide["difference"],
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, wide["difference"]):
        offset = 0.001 if value >= 0 else -0.001
        va = "bottom" if value >= 0 else "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.text(
        0.02,
        0.95,
        zero_line_label,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(base_dir, per_seed, aggregate, paired, plots_dir):
    accuracy_row = paired[paired["metric"] == "final_average_accuracy"].iloc[0]
    forgetting_row = paired[paired["metric"] == "avg_forgetting_excluding_final"].iloc[0]

    report = f"""# Day 12 Multi-Seed Figures Report

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

- Replay mean: {accuracy_row['mean_replay']:.4f}
- MTM mean: {accuracy_row['mean_mtm']:.4f}
- MTM minus replay: {accuracy_row['mean_difference_mtm_minus_replay']:.4f}
- 95% CI: [{accuracy_row['ci95_low']:.4f}, {accuracy_row['ci95_high']:.4f}]
- Paired t-test p-value: {accuracy_row['paired_t_p_value']:.4f}

Average forgetting excluding final task:

- Replay mean: {forgetting_row['mean_replay']:.4f}
- MTM mean: {forgetting_row['mean_mtm']:.4f}
- MTM minus replay: {forgetting_row['mean_difference_mtm_minus_replay']:.4f}
- 95% CI: [{forgetting_row['ci95_low']:.4f}, {forgetting_row['ci95_high']:.4f}]
- Paired t-test p-value: {forgetting_row['paired_t_p_value']:.4f}

## Interpretation

The accuracy plot shows that tuned MTM has slightly higher mean final average accuracy than replay, but the confidence interval for the paired difference includes zero. Therefore, the accuracy improvement should be interpreted cautiously.

The forgetting plot shows a clearer result. Tuned MTM has lower mean forgetting than replay, and the paired difference confidence interval does not include zero. The paired t-test gives p = {forgetting_row['paired_t_p_value']:.4f}, suggesting that MTM reduced forgetting significantly in this five-seed experiment.

The paired seed plots show how each seed changes from replay to MTM. These plots are useful because the experiment is paired: both methods were evaluated under the same seeds.

The difference plots show MTM minus replay for each seed. For accuracy, positive values favor MTM. For forgetting, negative values favor MTM.
"""

    report_path = base_dir / "day12_multiseed_figures_report.md"
    report_path.write_text(report)


def main():
    base_dir = Path("results/intent/multiseed_3epoch_mpc20")
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = base_dir / "day11_per_seed_results.csv"
    aggregate_path = base_dir / "day11_aggregate_summary.csv"
    paired_path = base_dir / "day11_paired_tests.csv"

    per_seed = pd.read_csv(per_seed_path)
    aggregate = pd.read_csv(aggregate_path)
    paired = pd.read_csv(paired_path)

    print("\nLoaded Day 11 per-seed results:")
    print(per_seed.round(4))

    print("\nLoaded Day 11 paired tests:")
    print(paired.round(4))

    plot_mean_metric_with_ci(
        aggregate=aggregate,
        metric="final_average_accuracy",
        title="Final Average Accuracy Across Seeds",
        ylabel="Final average accuracy",
        output_path=plots_dir / "mean_accuracy_ci.png",
    )

    plot_mean_metric_with_ci(
        aggregate=aggregate,
        metric="avg_forgetting_excluding_final",
        title="Average Forgetting Across Seeds",
        ylabel="Average forgetting excluding final task",
        output_path=plots_dir / "mean_forgetting_ci.png",
    )

    plot_paired_by_seed(
        per_seed=per_seed,
        metric="final_average_accuracy",
        title="Paired Final Accuracy by Seed",
        ylabel="Final average accuracy",
        output_path=plots_dir / "paired_accuracy_by_seed.png",
    )

    plot_paired_by_seed(
        per_seed=per_seed,
        metric="avg_forgetting_excluding_final",
        title="Paired Forgetting by Seed",
        ylabel="Average forgetting excluding final task",
        output_path=plots_dir / "paired_forgetting_by_seed.png",
    )

    plot_difference_by_seed(
        per_seed=per_seed,
        metric="final_average_accuracy",
        title="Accuracy Difference by Seed",
        ylabel="MTM minus Replay",
        output_path=plots_dir / "accuracy_difference_by_seed.png",
        zero_line_label="Positive favors MTM",
    )

    plot_difference_by_seed(
        per_seed=per_seed,
        metric="avg_forgetting_excluding_final",
        title="Forgetting Difference by Seed",
        ylabel="MTM minus Replay",
        output_path=plots_dir / "forgetting_difference_by_seed.png",
        zero_line_label="Negative favors MTM",
    )

    write_report(
        base_dir=base_dir,
        per_seed=per_seed,
        aggregate=aggregate,
        paired=paired,
        plots_dir=plots_dir,
    )

    print("\nDay 12 plots complete.")
    print("\nSaved plots:")
    for path in sorted(plots_dir.glob("*.png")):
        print("-", path)

    print("\nSaved report:")
    print(base_dir / "day12_multiseed_figures_report.md")


if __name__ == "__main__":
    main()

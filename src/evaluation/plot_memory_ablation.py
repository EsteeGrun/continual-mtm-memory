"""
Day 9 script.

Goal:
Create plots for the Day 8 memory-budget ablation.

Input:
    results/intent/memory_ablation/memory_ablation_summary.csv

Output:
    results/intent/memory_ablation/plots/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_value_labels(ax, fmt="{:.3f}", rotation=0):
    """
    Add labels above bars.
    """
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, padding=3, rotation=rotation, fontsize=8)


def plot_final_accuracy(summary, output_path):
    pivot = summary.pivot(
        index="memory_per_class",
        columns="method",
        values="final_average_accuracy",
    )

    ax = pivot.plot(kind="line", marker="o", figsize=(8, 5))

    ax.set_title("Final Average Accuracy by Memory Budget")
    ax.set_xlabel("Memory examples per class")
    ax.set_ylabel("Final average accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_forgetting(summary, output_path):
    pivot = summary.pivot(
        index="memory_per_class",
        columns="method",
        values="avg_forgetting_excluding_final",
    )

    ax = pivot.plot(kind="line", marker="o", figsize=(8, 5))

    ax.set_title("Average Forgetting by Memory Budget")
    ax.set_xlabel("Memory examples per class")
    ax.set_ylabel("Average forgetting excluding final task")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_tradeoff(summary, output_path):
    plt.figure(figsize=(8, 5))

    for method in sorted(summary["method"].unique()):
        method_df = summary[summary["method"] == method].sort_values("memory_per_class")

        plt.plot(
            method_df["avg_forgetting_excluding_final"],
            method_df["final_average_accuracy"],
            marker="o",
            label=method,
        )

        for _, row in method_df.iterrows():
            label = f"{int(row['memory_per_class'])}/class"
            plt.annotate(
                label,
                (
                    row["avg_forgetting_excluding_final"],
                    row["final_average_accuracy"],
                ),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.title("Accuracy–Forgetting Tradeoff")
    plt.xlabel("Average forgetting excluding final task")
    plt.ylabel("Final average accuracy")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grouped_bars(summary, output_path):
    methods = sorted(summary["method"].unique())
    budgets = sorted(summary["memory_per_class"].unique())

    x = np.arange(len(budgets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, method in enumerate(methods):
        method_df = summary[summary["method"] == method].sort_values("memory_per_class")

        offset = (idx - 0.5) * width

        axes[0].bar(
            x + offset,
            method_df["final_average_accuracy"],
            width,
            label=method,
        )

        axes[1].bar(
            x + offset,
            method_df["avg_forgetting_excluding_final"],
            width,
            label=method,
        )

    axes[0].set_title("Final Average Accuracy")
    axes[0].set_xlabel("Memory examples per class")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(budgets)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend()

    axes[1].set_title("Average Forgetting")
    axes[1].set_xlabel("Memory examples per class")
    axes[1].set_ylabel("Forgetting excluding final task")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(budgets)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    add_value_labels(axes[0])
    add_value_labels(axes[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(summary, output_path):
    accuracy_pivot = summary.pivot(
        index="memory_per_class",
        columns="method",
        values="final_average_accuracy",
    )

    forgetting_pivot = summary.pivot(
        index="memory_per_class",
        columns="method",
        values="avg_forgetting_excluding_final",
    )

    report = f"""# Day 9 Memory-Budget Ablation Figures

## Goal

Visualize the Day 8 memory-budget ablation comparing replay and the best MTM configuration.

## Figures created

1. `final_accuracy_by_memory_budget.png`
2. `forgetting_by_memory_budget.png`
3. `accuracy_forgetting_tradeoff.png`
4. `memory_ablation_grouped_bars.png`

## Final average accuracy

{accuracy_pivot.round(4).to_markdown()}

## Average forgetting excluding final task

{forgetting_pivot.round(4).to_markdown()}

## Interpretation

The memory-budget ablation shows that replay and MTM both improve as the replay memory budget increases.

At 5 examples per class, replay has slightly higher final average accuracy, while MTM has slightly lower forgetting.

At 10 and 20 examples per class, the tuned MTM configuration slightly outperforms replay in the controlled 1-epoch setting on both final average accuracy and forgetting.

This does not prove that MTM is universally better than replay. The earlier Day 4 replay experiment used 3 epochs and achieved higher performance. However, under a controlled 1-epoch setting, the tuned MTM prototype is competitive and sometimes better than replay.

The result supports the next research direction: improve multi-timescale memory through learned gating or better slow-memory consolidation.
"""

    output_path.write_text(report)


def main():
    base_dir = Path("results/intent/memory_ablation")
    summary_path = base_dir / "memory_ablation_summary.csv"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find {summary_path}. Run summarize_memory_ablation.py first."
        )

    summary = pd.read_csv(summary_path)

    print("\nLoaded memory ablation summary:")
    print(summary.round(4))

    plot_final_accuracy(
        summary=summary,
        output_path=plots_dir / "final_accuracy_by_memory_budget.png",
    )

    plot_forgetting(
        summary=summary,
        output_path=plots_dir / "forgetting_by_memory_budget.png",
    )

    plot_tradeoff(
        summary=summary,
        output_path=plots_dir / "accuracy_forgetting_tradeoff.png",
    )

    plot_grouped_bars(
        summary=summary,
        output_path=plots_dir / "memory_ablation_grouped_bars.png",
    )

    write_report(
        summary=summary,
        output_path=base_dir / "memory_ablation_report_figures.md",
    )

    print("\nDay 9 plots complete.")
    print("\nSaved plots:")
    for path in sorted(plots_dir.glob("*.png")):
        print("-", path)

    print("\nSaved figure report:")
    print(base_dir / "memory_ablation_report_figures.md")


if __name__ == "__main__":
    main()

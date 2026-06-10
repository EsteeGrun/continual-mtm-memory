"""
Day 5 script.

Goal:
Analyze and visualize Day 3 and Day 4 continual-learning results.

Compares:
- Sequential frozen encoder baseline
- Replay frozen encoder baseline

Outputs:
- Summary CSV
- Markdown report
- Plots in results/intent/plots/

Run from project root:

    python src/evaluation/plot_intent_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_method_results(base_dir, method_name):
    method_dir = Path(base_dir) / method_name

    results = pd.read_csv(method_dir / "results_matrix.csv", index_col=0)
    metrics = pd.read_csv(method_dir / "metrics.csv")
    forgetting = pd.read_csv(method_dir / "forgetting.csv")

    return results, metrics, forgetting


def compute_summary(method_label, results, forgetting, memory_examples):
    final_average_accuracy = results.iloc[-1].dropna().mean()

    last_task = results.columns[-1]

    avg_forgetting_including_final = forgetting["forgetting"].mean()

    avg_forgetting_excluding_final = forgetting[
        forgetting["task"] != last_task
    ]["forgetting"].mean()

    return {
        "method": method_label,
        "final_average_accuracy": final_average_accuracy,
        "avg_forgetting_including_final": avg_forgetting_including_final,
        "avg_forgetting_excluding_final": avg_forgetting_excluding_final,
        "memory_examples": memory_examples,
    }


def plot_bar(summary_df, column, title, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["method"], summary_df[column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_average_seen_accuracy(seq_metrics, rep_metrics, output_path):
    plt.figure(figsize=(8, 5))

    plt.plot(
        seq_metrics["after_task"],
        seq_metrics["average_seen_accuracy"],
        marker="o",
        label="Sequential frozen",
    )

    plt.plot(
        rep_metrics["after_task"],
        rep_metrics["average_seen_accuracy"],
        marker="o",
        label="Replay frozen",
    )

    plt.title("Average Seen Accuracy Over Continual Tasks")
    plt.xlabel("After training task")
    plt.ylabel("Average accuracy on seen tasks")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_results_matrix(results, title, output_path):
    matrix = results.to_numpy(dtype=float)

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Accuracy")

    plt.title(title)
    plt.xlabel("Evaluation task")
    plt.ylabel("Training stage")

    plt.xticks(
        ticks=np.arange(len(results.columns)),
        labels=results.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(results.index)),
        labels=results.index,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_final_task_accuracies(seq_results, rep_results, output_path):
    final_seq = seq_results.iloc[-1]
    final_rep = rep_results.iloc[-1]

    tasks = seq_results.columns
    x = np.arange(len(tasks))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(x - width / 2, final_seq.values, width, label="Sequential frozen")
    plt.bar(x + width / 2, final_rep.values, width, label="Replay frozen")

    plt.title("Final Accuracy on Each Task After Task 9")
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_markdown_report(summary_df, output_path):
    seq = summary_df[summary_df["method"] == "sequential_frozen"].iloc[0]
    rep = summary_df[summary_df["method"] == "replay_frozen"].iloc[0]

    report = f"""# Day 5 Analysis Report

## Goal

Compare the Day 3 sequential frozen baseline against the Day 4 replay frozen baseline for continual CLINC150 intent classification.

## Main results

| Method | Final average accuracy | Avg forgetting excluding final task | Memory examples |
|---|---:|---:|---:|
| Sequential frozen | {seq["final_average_accuracy"]:.4f} | {seq["avg_forgetting_excluding_final"]:.4f} | {int(seq["memory_examples"])} |
| Replay frozen | {rep["final_average_accuracy"]:.4f} | {rep["avg_forgetting_excluding_final"]:.4f} | {int(rep["memory_examples"])} |

## Interpretation

The sequential frozen baseline shows severe catastrophic forgetting. It learns each task when trained on it, but performance on earlier tasks collapses after later tasks.

The replay frozen baseline substantially reduces forgetting by storing a small buffer of old examples and mixing them with the current task during training.

Replay improves final average accuracy from {seq["final_average_accuracy"]:.4f} to {rep["final_average_accuracy"]:.4f}.

Replay reduces average forgetting excluding the final task from {seq["avg_forgetting_excluding_final"]:.4f} to {rep["avg_forgetting_excluding_final"]:.4f}.

## Research implication

Replay is a strong anti-forgetting baseline, but it requires storing examples. This creates a useful comparison point for the future multi-timescale memory method: the new method should aim to match or improve replay while using memory more efficiently, consolidating knowledge more systematically, or improving the stability-plasticity tradeoff.
"""

    Path(output_path).write_text(report)


def main():
    base_dir = Path("results/intent")
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    seq_results, seq_metrics, seq_forgetting = load_method_results(
        base_dir=base_dir,
        method_name="sequential_frozen",
    )

    rep_results, rep_metrics, rep_forgetting = load_method_results(
        base_dir=base_dir,
        method_name="replay_frozen",
    )

    summary_rows = [
        compute_summary(
            method_label="sequential_frozen",
            results=seq_results,
            forgetting=seq_forgetting,
            memory_examples=0,
        ),
        compute_summary(
            method_label="replay_frozen",
            results=rep_results,
            forgetting=rep_forgetting,
            memory_examples=3000,
        ),
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = base_dir / "day5_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Also save/overwrite the Day 3 vs Day 4 comparison file.
    comparison_path = base_dir / "day3_day4_comparison.csv"
    summary_df.to_csv(comparison_path, index=False)

    plot_bar(
        summary_df=summary_df,
        column="final_average_accuracy",
        title="Final Average Accuracy Comparison",
        ylabel="Final average accuracy",
        output_path=plots_dir / "final_average_accuracy_comparison.png",
    )

    plot_bar(
        summary_df=summary_df,
        column="avg_forgetting_excluding_final",
        title="Average Forgetting Comparison",
        ylabel="Average forgetting excluding final task",
        output_path=plots_dir / "average_forgetting_comparison.png",
    )

    plot_average_seen_accuracy(
        seq_metrics=seq_metrics,
        rep_metrics=rep_metrics,
        output_path=plots_dir / "average_seen_accuracy_over_tasks.png",
    )

    plot_results_matrix(
        results=seq_results,
        title="Sequential Frozen Results Matrix",
        output_path=plots_dir / "sequential_results_matrix.png",
    )

    plot_results_matrix(
        results=rep_results,
        title="Replay Frozen Results Matrix",
        output_path=plots_dir / "replay_results_matrix.png",
    )

    plot_final_task_accuracies(
        seq_results=seq_results,
        rep_results=rep_results,
        output_path=plots_dir / "final_task_accuracies.png",
    )

    report_path = base_dir / "day5_report.md"
    write_markdown_report(summary_df, report_path)

    print("\nDay 5 analysis complete.")
    print("\nSummary:")
    print(summary_df.round(4))

    print("\nSaved summary:")
    print(summary_path)

    print("\nSaved report:")
    print(report_path)

    print("\nSaved plots:")
    for path in sorted(plots_dir.glob("*.png")):
        print("-", path)


if __name__ == "__main__":
    main()

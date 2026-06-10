"""
Day 8 script.

Goal:
Summarize replay vs MTM performance under different replay-memory budgets.

Run from project root:

    python src/evaluation/summarize_memory_ablation.py
"""

from pathlib import Path
import pandas as pd


def summarize_result(method, memory_per_class, folder):
    folder = Path(folder)

    results = pd.read_csv(folder / "results_matrix.csv", index_col=0)
    forgetting = pd.read_csv(folder / "forgetting.csv")

    final_average_accuracy = results.iloc[-1].dropna().mean()

    last_task = results.columns[-1]

    avg_forgetting_including_final = forgetting["forgetting"].mean()

    avg_forgetting_excluding_final = forgetting[
        forgetting["task"] != last_task
    ]["forgetting"].mean()

    memory_examples = 150 * memory_per_class

    return {
        "method": method,
        "memory_per_class": memory_per_class,
        "memory_examples": memory_examples,
        "final_average_accuracy": final_average_accuracy,
        "avg_forgetting_including_final": avg_forgetting_including_final,
        "avg_forgetting_excluding_final": avg_forgetting_excluding_final,
    }


def main():
    base = Path("results/intent/memory_ablation")

    configs = [
        ("replay_frozen", 5, base / "replay_mpc_5"),
        ("replay_frozen", 10, base / "replay_mpc_10"),
        ("replay_frozen", 20, base / "replay_mpc_20"),
        ("mtm_no_fast_90_10", 5, base / "mtm_mpc_5"),
        ("mtm_no_fast_90_10", 10, base / "mtm_mpc_10"),
        ("mtm_no_fast_90_10", 20, base / "mtm_mpc_20"),
    ]

    rows = []

    for method, memory_per_class, folder in configs:
        rows.append(
            summarize_result(
                method=method,
                memory_per_class=memory_per_class,
                folder=folder,
            )
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["memory_per_class", "method"]
    ).reset_index(drop=True)

    output_path = base / "memory_ablation_summary.csv"
    summary.to_csv(output_path, index=False)

    print("\nDAY 8 MEMORY ABLATION SUMMARY")
    print(summary.round(4))

    # Create a compact pivot table.
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

    print("\nFINAL AVERAGE ACCURACY BY MEMORY BUDGET")
    print(accuracy_pivot.round(4))

    print("\nAVERAGE FORGETTING EXCLUDING FINAL TASK BY MEMORY BUDGET")
    print(forgetting_pivot.round(4))

    report = f"""# Day 8 Memory-Budget Ablation

## Goal

Compare replay and the best MTM configuration under different memory budgets.

The tested memory budgets were:

- 5 examples per class = 750 stored examples
- 10 examples per class = 1500 stored examples
- 20 examples per class = 3000 stored examples

All Day 8 runs used 1 epoch per task for a controlled comparison.

## Summary Table

{summary.round(4).to_markdown(index=False)}

## Interpretation Template

Replay is expected to remain a strong baseline because it trains directly on stored examples.

The MTM configuration uses the best Day 7 fusion rule:

- fast weight = 0.0
- medium weight = 0.9
- slow weight = 0.1

The key question is whether MTM becomes more competitive when memory is limited.

If replay remains stronger at all budgets, then the current MTM prototype still needs better gating, better slow-memory training, or a smaller memory footprint advantage.

If MTM closes the gap at lower memory budgets, then multi-timescale memory may be useful under constrained memory conditions.
"""

    report_path = base / "memory_ablation_report.md"
    report_path.write_text(report)

    print(f"\nSaved summary to: {output_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

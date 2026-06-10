"""
Day 10 summary.

Goal:
Compare replay and tuned MTM under the same setting:

- memory_per_class = 20
- memory_examples = 3000
- epochs = 3

Run from project root:

    python src/evaluation/summarize_day10_fair_comparison.py
"""

from pathlib import Path
import pandas as pd


def summarize(method, folder, memory_examples, epochs):
    folder = Path(folder)

    results = pd.read_csv(folder / "results_matrix.csv", index_col=0)
    forgetting = pd.read_csv(folder / "forgetting.csv")

    final_average_accuracy = results.iloc[-1].dropna().mean()
    last_task = results.columns[-1]

    avg_forgetting_including_final = forgetting["forgetting"].mean()
    avg_forgetting_excluding_final = forgetting[
        forgetting["task"] != last_task
    ]["forgetting"].mean()

    return {
        "method": method,
        "epochs": epochs,
        "memory_examples": memory_examples,
        "final_average_accuracy": final_average_accuracy,
        "avg_forgetting_including_final": avg_forgetting_including_final,
        "avg_forgetting_excluding_final": avg_forgetting_excluding_final,
    }


def main():
    base = Path("results/intent/fair_3epoch_mpc20")

    rows = [
        summarize(
            method="replay_frozen_mpc20_e3",
            folder=base / "replay_mpc20_e3",
            memory_examples=3000,
            epochs=3,
        ),
        summarize(
            method="mtm_no_fast_90_10_mpc20_e3",
            folder=base / "mtm_mpc20_e3",
            memory_examples=3000,
            epochs=3,
        ),
    ]

    comparison = pd.DataFrame(rows)
    comparison_path = base / "day10_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nDAY 10 FAIR 3-EPOCH COMPARISON")
    print(comparison.round(4))

    replay = comparison[comparison["method"] == "replay_frozen_mpc20_e3"].iloc[0]
    mtm = comparison[comparison["method"] == "mtm_no_fast_90_10_mpc20_e3"].iloc[0]

    accuracy_difference = mtm["final_average_accuracy"] - replay["final_average_accuracy"]
    forgetting_difference = mtm["avg_forgetting_excluding_final"] - replay["avg_forgetting_excluding_final"]

    print("\nDIFFERENCES: MTM minus Replay")
    print("Final average accuracy difference:", round(accuracy_difference, 4))
    print("Forgetting difference:", round(forgetting_difference, 4))

    if accuracy_difference > 0:
        accuracy_sentence = "MTM has higher final average accuracy than replay."
    else:
        accuracy_sentence = "Replay has higher final average accuracy than MTM."

    if forgetting_difference < 0:
        forgetting_sentence = "MTM has lower forgetting than replay."
    else:
        forgetting_sentence = "Replay has lower forgetting than MTM."

    report = f"""# Day 10 Fair 3-Epoch Comparison

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

{comparison.round(4).to_string(index=False)}

## Difference: MTM minus Replay

Final average accuracy difference: {accuracy_difference:.4f}

Average forgetting excluding final task difference: {forgetting_difference:.4f}

## Interpretation

{accuracy_sentence}

{forgetting_sentence}

This comparison is fairer than comparing the original 3-epoch replay result against the 1-epoch memory-budget ablation. However, MTM still trains multiple classifier heads, so this is matched by epoch count rather than exactly matched by compute.
"""

    report_path = base / "day10_report.md"
    report_path.write_text(report)

    print(f"\nSaved comparison to: {comparison_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

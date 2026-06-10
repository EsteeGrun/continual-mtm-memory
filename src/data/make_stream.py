"""
Day 2 script.

Goal:
Create a continual-learning task stream from CLINC150.

We will:
1. Load DeepPavlov/clinc_oos, plus configuration.
2. Remove the out-of-scope / OOS class.
3. Keep the 150 in-scope intents.
4. Shuffle intent names with a fixed seed.
5. Split them into 10 tasks with 15 intents each.
6. Save train/validation/test CSV files for every task.
7. Save metadata files describing the stream.

Run from project root:

    python src/data/make_stream.py
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset


OOS_NAMES = {"oos", "out_of_scope", "out-of-scope", "out of scope"}


def is_oos_label(label_text: str) -> bool:
    clean = str(label_text).strip().lower()
    return clean in OOS_NAMES or "oos" == clean


def load_clinc150_plus():
    return load_dataset("DeepPavlov/clinc_oos", "plus")


def dataset_split_to_dataframe(dataset_split):
    return pd.DataFrame(dataset_split)


def filter_in_scope(df: pd.DataFrame) -> pd.DataFrame:
    if "label_text" not in df.columns:
        raise ValueError("Expected column 'label_text' not found.")

    filtered = df[~df["label_text"].apply(is_oos_label)].copy()
    return filtered


def build_label_mapping(all_dfs):
    """
    Build a stable global label mapping from label_text to global_label.

    We use alphabetical order for global labels so the mapping is stable and readable.
    Task order is shuffled separately.
    """
    label_names = sorted(
        set(
            label_name
            for df in all_dfs
            for label_name in df["label_text"].unique().tolist()
            if not is_oos_label(label_name)
        )
    )

    label_to_global = {label_name: idx for idx, label_name in enumerate(label_names)}
    global_to_label = {idx: label_name for label_name, idx in label_to_global.items()}

    return label_names, label_to_global, global_to_label


def split_intents_into_tasks(label_names, num_tasks, seed):
    if len(label_names) % num_tasks != 0:
        raise ValueError(
            f"Number of labels ({len(label_names)}) must divide evenly into "
            f"num_tasks ({num_tasks}) for this first version."
        )

    rng = random.Random(seed)
    shuffled_labels = list(label_names)
    rng.shuffle(shuffled_labels)

    intents_per_task = len(shuffled_labels) // num_tasks
    tasks = []

    for task_id in range(num_tasks):
        start = task_id * intents_per_task
        end = start + intents_per_task
        task_intents = shuffled_labels[start:end]

        tasks.append(
            {
                "task_id": task_id,
                "num_intents": len(task_intents),
                "intent_names": task_intents,
            }
        )

    return tasks


def add_global_labels(df, label_to_global):
    df = df.copy()
    df["global_label"] = df["label_text"].map(label_to_global)

    if df["global_label"].isna().any():
        missing = df[df["global_label"].isna()]["label_text"].unique()
        raise ValueError(f"Some labels were not mapped: {missing}")

    df["global_label"] = df["global_label"].astype(int)
    return df


def save_task_files(train_df, val_df, test_df, tasks, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for task in tasks:
        task_id = task["task_id"]
        task_intents = set(task["intent_names"])

        task_dir = output_dir / f"task_{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        task_train = train_df[train_df["label_text"].isin(task_intents)].copy()
        task_val = val_df[val_df["label_text"].isin(task_intents)].copy()
        task_test = test_df[test_df["label_text"].isin(task_intents)].copy()

        task_train["task_id"] = task_id
        task_val["task_id"] = task_id
        task_test["task_id"] = task_id

        keep_cols = ["text", "label", "label_text", "global_label", "task_id"]

        task_train[keep_cols].to_csv(task_dir / "train.csv", index=False)
        task_val[keep_cols].to_csv(task_dir / "validation.csv", index=False)
        task_test[keep_cols].to_csv(task_dir / "test.csv", index=False)

        summary_rows.append(
            {
                "task_id": task_id,
                "num_intents": len(task_intents),
                "train_examples": len(task_train),
                "validation_examples": len(task_val),
                "test_examples": len(task_test),
                "intent_names": ", ".join(task["intent_names"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "stream_summary.csv", index=False)

    with open(output_dir / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/clinc150_stream"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Loading CLINC150 / clinc_oos plus dataset")
    print("=" * 80)

    dataset = load_clinc150_plus()

    train_df = dataset_split_to_dataframe(dataset["train"])
    val_df = dataset_split_to_dataframe(dataset["validation"])
    test_df = dataset_split_to_dataframe(dataset["test"])

    print("\nOriginal split sizes:")
    print(f"train: {len(train_df)}")
    print(f"validation: {len(val_df)}")
    print(f"test: {len(test_df)}")

    print("\nOriginal columns:")
    print(train_df.columns.tolist())

    train_df = filter_in_scope(train_df)
    val_df = filter_in_scope(val_df)
    test_df = filter_in_scope(test_df)

    print("\nAfter removing OOS:")
    print(f"train: {len(train_df)}")
    print(f"validation: {len(val_df)}")
    print(f"test: {len(test_df)}")

    label_names, label_to_global, global_to_label = build_label_mapping(
        [train_df, val_df, test_df]
    )

    print(f"\nNumber of in-scope intent labels: {len(label_names)}")

    if len(label_names) != 150:
        print("WARNING: Expected 150 in-scope labels. Check OOS filtering.")

    train_df = add_global_labels(train_df, label_to_global)
    val_df = add_global_labels(val_df, label_to_global)
    test_df = add_global_labels(test_df, label_to_global)

    tasks = split_intents_into_tasks(
        label_names=label_names,
        num_tasks=args.num_tasks,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_task_files(train_df, val_df, test_df, tasks, output_dir)

    with open(output_dir / "label_to_global.json", "w") as f:
        json.dump(label_to_global, f, indent=2)

    with open(output_dir / "global_to_label.json", "w") as f:
        json.dump(global_to_label, f, indent=2)

    print("\nCreated continual-learning stream.")
    print(f"Output directory: {output_dir}")

    print("\nTask summary:")
    for task in tasks:
        print(
            f"Task {task['task_id']:02d}: "
            f"{task['num_intents']} intents | "
            f"{task['intent_names'][:3]} ..."
        )

    print("\nSaved files:")
    print(f"- {output_dir / 'tasks.json'}")
    print(f"- {output_dir / 'stream_summary.csv'}")
    print(f"- {output_dir / 'label_to_global.json'}")
    print(f"- {output_dir / 'global_to_label.json'}")
    print("- task_00/train.csv, validation.csv, test.csv, etc.")


if __name__ == "__main__":
    main()

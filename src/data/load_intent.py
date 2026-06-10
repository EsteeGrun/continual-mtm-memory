"""
Day 1 script.

Goal:
Load an intent-classification dataset and inspect its structure.

For now, we support:
- CLINC150 through Hugging Face dataset name: clinc_oos
- BANKING77 through Hugging Face dataset name: banking77

Run from the project root:

    python src/data/load_intent.py --dataset clinc_oos
    python src/data/load_intent.py --dataset banking77
"""

import argparse
from datasets import load_dataset


def load_clinc150():
    """
    Loads CLINC150 from Hugging Face.

    The Hugging Face dataset is called 'clinc_oos'.
    The 'plus' configuration includes in-scope examples plus out-of-scope examples.
    For our first intent-classification experiment, we will inspect it first.
    """
    dataset = load_dataset("DeepPavlov/clinc_oos", "plus")
    return dataset


def load_banking77():
    """
    Loads BANKING77 from Hugging Face.
    """
    dataset = load_dataset("PolyAI/banking77")
    return dataset


def inspect_dataset(dataset, dataset_name):
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    print("\nAvailable splits:")
    print(dataset)

    print("\nSplit sizes:")
    for split_name in dataset.keys():
        print(f"{split_name}: {len(dataset[split_name])}")

    train_split = dataset["train"]

    print("\nColumn names:")
    print(train_split.column_names)

    print("\nFirst training example:")
    print(train_split[0])

    print("\nFirst five training examples:")
    for i in range(5):
        print("-" * 80)
        print(train_split[i])

    # Try to find the label column.
    possible_label_columns = ["intent", "label"]
    label_column = None

    for col in possible_label_columns:
        if col in train_split.column_names:
            label_column = col
            break

    if label_column is None:
        print("\nCould not automatically find the label column.")
        return

    print(f"\nDetected label column: {label_column}")

    label_feature = train_split.features[label_column]

    print("\nLabel feature:")
    print(label_feature)

    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = label_feature.names
        print(f"\nNumber of labels: {len(label_names)}")

        print("\nFirst 20 label names:")
        for idx, name in enumerate(label_names[:20]):
            print(f"{idx}: {name}")
    else:
        unique_labels = sorted(set(train_split[label_column]))
        print(f"\nNumber of unique labels: {len(unique_labels)}")
        print("\nFirst 20 labels:")
        print(unique_labels[:20])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="clinc_oos",
        choices=["clinc_oos", "banking77"],
        help="Which dataset to load."
    )
    args = parser.parse_args()

    if args.dataset == "clinc_oos":
        dataset = load_clinc150()
    elif args.dataset == "banking77":
        dataset = load_banking77()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    inspect_dataset(dataset, args.dataset)


if __name__ == "__main__":
    main()

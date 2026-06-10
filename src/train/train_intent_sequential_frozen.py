"""
Day 3 script.

Goal:
Train a frozen encoder + classifier head sequentially on CLINC150 tasks.

This is our first catastrophic-forgetting baseline.

Method:
- Use DistilBERT as a frozen text encoder.
- Train only a linear classifier head.
- Train sequentially on task_00, task_01, ..., task_09.
- After each task, evaluate on all tasks seen so far.
- Save results matrix and forgetting metrics.

Run quick test:

    python src/train/train_intent_sequential_frozen.py --num_tasks 2 --epochs 1 --max_train_batches 5 --max_eval_batches 5

Run fuller Day 3 experiment:

    python src/train/train_intent_sequential_frozen.py --num_tasks 10 --epochs 3
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class IntentCSVDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=64):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df["global_label"].astype(int).tolist()

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class FrozenEncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze encoder parameters.
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # DistilBERT/BERT CLS representation.
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)
        return logits


def make_loader(csv_path, tokenizer, batch_size, shuffle, max_length):
    dataset = IntentCSVDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_task(
    model,
    train_loader,
    optimizer,
    device,
    epoch_count,
    max_train_batches=None,
):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epoch_count):
        total_loss = 0.0
        total_examples = 0

        progress = tqdm(
            train_loader,
            desc=f"training epoch {epoch + 1}/{epoch_count}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(total_examples, 1)
        print(f"    epoch {epoch + 1}: train_loss={avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, data_loader, device, max_eval_batches=None):
    model.eval()

    correct = 0
    total = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        predictions = torch.argmax(logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_forgetting(results_matrix: pd.DataFrame):
    """
    results_matrix rows:
        after_task_00, after_task_01, ...
    columns:
        task_00, task_01, ...

    Forgetting for task i:
        best accuracy achieved after learning task i and later
        minus final accuracy.
    """
    final_row = results_matrix.iloc[-1]
    forgetting_rows = []

    for column in results_matrix.columns:
        series = results_matrix[column].dropna()

        if len(series) == 0:
            continue

        best_accuracy = series.max()
        final_accuracy = final_row[column]

        forgetting_rows.append(
            {
                "task": column,
                "best_accuracy": best_accuracy,
                "final_accuracy": final_accuracy,
                "forgetting": best_accuracy - final_accuracy,
            }
        )

    return pd.DataFrame(forgetting_rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/clinc150_stream",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/intent/sequential_frozen",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
    )
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Useful for quick debugging.
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("Day 3: Sequential frozen encoder baseline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    with open(data_dir / "global_to_label.json") as f:
        global_to_label = json.load(f)

    num_labels = len(global_to_label)
    print(f"Number of labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = FrozenEncoderClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
    )
    model.to(device)

    # Only classifier parameters are trainable.
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
    )

    results = {}
    metrics_rows = []

    for task_id in range(args.num_tasks):
        print("\n" + "=" * 80)
        print(f"Training on task_{task_id:02d}")
        print("=" * 80)

        task_dir = data_dir / f"task_{task_id:02d}"
        train_csv = task_dir / "train.csv"

        train_loader = make_loader(
            csv_path=train_csv,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            max_length=args.max_length,
        )

        train_one_task(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_count=args.epochs,
            max_train_batches=args.max_train_batches,
        )

        # Evaluate on all tasks seen so far.
        row_name = f"after_task_{task_id:02d}"
        results[row_name] = {}

        seen_accuracies = []

        print("\nEvaluating on seen tasks:")

        for eval_task_id in range(task_id + 1):
            eval_task_dir = data_dir / f"task_{eval_task_id:02d}"
            test_csv = eval_task_dir / "test.csv"

            test_loader = make_loader(
                csv_path=test_csv,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                shuffle=False,
                max_length=args.max_length,
            )

            accuracy = evaluate(
                model=model,
                data_loader=test_loader,
                device=device,
                max_eval_batches=args.max_eval_batches,
            )

            task_col = f"task_{eval_task_id:02d}"
            results[row_name][task_col] = accuracy
            seen_accuracies.append(accuracy)

            print(f"    {task_col}: accuracy={accuracy:.4f}")

        avg_seen_accuracy = float(np.mean(seen_accuracies))

        metrics_rows.append(
            {
                "after_task": task_id,
                "average_seen_accuracy": avg_seen_accuracy,
            }
        )

        print(f"\nAverage accuracy on seen tasks: {avg_seen_accuracy:.4f}")

        # Save intermediate results after every task.
        results_matrix = pd.DataFrame.from_dict(results, orient="index")
        results_matrix.to_csv(output_dir / "results_matrix.csv")

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        torch.save(
            model.classifier.state_dict(),
            output_dir / f"classifier_after_task_{task_id:02d}.pt",
        )

    # Final forgetting.
    results_matrix = pd.read_csv(output_dir / "results_matrix.csv", index_col=0)
    forgetting_df = compute_forgetting(results_matrix)
    forgetting_df.to_csv(output_dir / "forgetting.csv", index=False)

    torch.save(
        model.classifier.state_dict(),
        output_dir / "classifier_final.pt",
    )

    print("\n" + "=" * 80)
    print("Day 3 complete.")
    print("=" * 80)
    print(f"Saved results to: {output_dir}")
    print("\nFinal results matrix:")
    print(results_matrix)

    print("\nForgetting:")
    print(forgetting_df)

    if len(forgetting_df) > 0:
        avg_forgetting = forgetting_df["forgetting"].mean()
        print(f"\nAverage forgetting: {avg_forgetting:.4f}")


if __name__ == "__main__":
    main()

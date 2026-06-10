"""
Day 6 script.

Goal:
Train a first multi-timescale memory prototype for continual intent classification.

This is NOT full HOPE. It is a simple HOPE/CMS-inspired prototype.

Model:
- Frozen DistilBERT encoder
- Fast classifier head:
    updated on current task only
    optionally reset each task
- Medium classifier head:
    updated on current task + replay buffer
- Slow classifier head:
    updated every K tasks on replay buffer only
- Combined prediction:
    weighted average of fast, medium, and slow logits

Run quick test:

    python src/train/train_intent_mtm_frozen.py --num_tasks 2 --epochs 1 --slow_epochs 1 --max_train_batches 5 --max_eval_batches 5

Run Day 6 prototype:

    python src/train/train_intent_mtm_frozen.py --num_tasks 10 --epochs 1 --slow_epochs 1 --memory_per_class 20

Stronger but slower run:

    python src/train/train_intent_mtm_frozen.py --num_tasks 10 --epochs 3 --slow_epochs 2 --memory_per_class 20
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


class IntentDataFrameDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.df = df.reset_index(drop=True).copy()
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


class MultiTimescaleFrozenClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_size = self.encoder.config.hidden_size
        self.num_labels = num_labels

        self.fast_head = nn.Linear(self.hidden_size, num_labels)
        self.medium_head = nn.Linear(self.hidden_size, num_labels)
        self.slow_head = nn.Linear(self.hidden_size, num_labels)

        self.slow_trained = False

    def reset_fast_head(self, device):
        self.fast_head = nn.Linear(self.hidden_size, self.num_labels).to(device)

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def logits_for_head(self, input_ids, attention_mask, head_name):
        h = self.encode(input_ids, attention_mask)

        if head_name == "fast":
            return self.fast_head(h)
        if head_name == "medium":
            return self.medium_head(h)
        if head_name == "slow":
            return self.slow_head(h)

        raise ValueError(f"Unknown head: {head_name}")

    def combined_logits(
        self,
        input_ids,
        attention_mask,
        fast_weight=0.15,
        medium_weight=0.70,
        slow_weight=0.15,
    ):
        h = self.encode(input_ids, attention_mask)

        logits = []
        weights = []

        logits.append(self.fast_head(h))
        weights.append(fast_weight)

        logits.append(self.medium_head(h))
        weights.append(medium_weight)

        if self.slow_trained:
            logits.append(self.slow_head(h))
            weights.append(slow_weight)

        weights = torch.tensor(weights, device=h.device, dtype=h.dtype)
        weights = weights / weights.sum()

        combined = sum(w * l for w, l in zip(weights, logits))
        return combined


def make_loader(df, tokenizer, batch_size, shuffle, max_length):
    dataset = IntentDataFrameDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def sample_examples_per_class(df, memory_per_class, seed):
    samples = []

    for label, group in df.groupby("global_label"):
        n = min(memory_per_class, len(group))
        sampled = group.sample(n=n, random_state=seed)
        samples.append(sampled)

    if not samples:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(samples, ignore_index=True)


def update_replay_buffer(replay_df, current_task_train_df, memory_per_class, seed):
    combined = pd.concat([replay_df, current_task_train_df], ignore_index=True)

    updated = sample_examples_per_class(
        df=combined,
        memory_per_class=memory_per_class,
        seed=seed,
    )

    return updated


def combine_current_and_replay(current_df, replay_df):
    if len(replay_df) == 0:
        combined = current_df.copy()
    else:
        combined = pd.concat([current_df, replay_df], ignore_index=True)

    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return combined


def train_head(
    model,
    head_name,
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
            desc=f"training {head_name} epoch {epoch + 1}/{epoch_count}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model.logits_for_head(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_name=head_name,
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(total_examples, 1)
        print(f"    {head_name} epoch {epoch + 1}: train_loss={avg_loss:.4f}")


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    mode,
    max_eval_batches=None,
    fast_weight=0.15,
    medium_weight=0.70,
    slow_weight=0.15,
):
    model.eval()

    correct = 0
    total = 0

    for batch_idx, batch in enumerate(data_loader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if mode in {"fast", "medium", "slow"}:
            logits = model.logits_for_head(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_name=mode,
            )
        elif mode == "combined":
            logits = model.combined_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                fast_weight=fast_weight,
                medium_weight=medium_weight,
                slow_weight=slow_weight,
            )
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

        predictions = torch.argmax(logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_forgetting(results_matrix: pd.DataFrame):
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
        default="results/intent/mtm_frozen",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
    )

    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--slow_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--memory_per_class", type=int, default=20)
    parser.add_argument("--slow_update_frequency", type=int, default=2)

    parser.add_argument("--fast_lr", type=float, default=2e-3)
    parser.add_argument("--medium_lr", type=float, default=1e-3)
    parser.add_argument("--slow_lr", type=float, default=5e-4)

    parser.add_argument("--fast_weight", type=float, default=0.15)
    parser.add_argument("--medium_weight", type=float, default=0.70)
    parser.add_argument("--slow_weight", type=float, default=0.15)

    parser.add_argument("--reset_fast_each_task", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("Day 6: Multi-timescale memory frozen encoder prototype")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Epochs for fast/medium: {args.epochs}")
    print(f"Slow epochs: {args.slow_epochs}")
    print(f"Memory per class: {args.memory_per_class}")
    print(f"Slow update frequency: every {args.slow_update_frequency} tasks")
    print(f"Fast/medium/slow weights: {args.fast_weight}, {args.medium_weight}, {args.slow_weight}")

    with open(data_dir / "global_to_label.json") as f:
        global_to_label = json.load(f)

    num_labels = len(global_to_label)
    print(f"Number of labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = MultiTimescaleFrozenClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
    )
    model.to(device)

    fast_optimizer = torch.optim.AdamW(model.fast_head.parameters(), lr=args.fast_lr)
    medium_optimizer = torch.optim.AdamW(model.medium_head.parameters(), lr=args.medium_lr)
    slow_optimizer = torch.optim.AdamW(model.slow_head.parameters(), lr=args.slow_lr)

    replay_df = pd.DataFrame(
        columns=["text", "label", "label_text", "global_label", "task_id"]
    )

    modes = ["fast", "medium", "slow", "combined"]
    results_by_mode = {mode: {} for mode in modes}
    metrics_rows = []

    for task_id in range(args.num_tasks):
        print("\n" + "=" * 80)
        print(f"Training multi-timescale model on task_{task_id:02d}")
        print("=" * 80)

        task_dir = data_dir / f"task_{task_id:02d}"
        current_train_df = pd.read_csv(task_dir / "train.csv")

        print(f"Current task train examples: {len(current_train_df)}")
        print(f"Replay buffer examples before training: {len(replay_df)}")

        if args.reset_fast_each_task:
            model.reset_fast_head(device)
            fast_optimizer = torch.optim.AdamW(model.fast_head.parameters(), lr=args.fast_lr)
            print("Fast head reset for new task.")

        fast_train_df = current_train_df.copy()
        medium_train_df = combine_current_and_replay(current_train_df, replay_df)

        print(f"Fast training examples: {len(fast_train_df)}")
        print(f"Medium training examples: {len(medium_train_df)}")

        fast_loader = make_loader(
            df=fast_train_df,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            max_length=args.max_length,
        )

        medium_loader = make_loader(
            df=medium_train_df,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            max_length=args.max_length,
        )

        print("\nUpdating fast memory/head...")
        train_head(
            model=model,
            head_name="fast",
            train_loader=fast_loader,
            optimizer=fast_optimizer,
            device=device,
            epoch_count=args.epochs,
            max_train_batches=args.max_train_batches,
        )

        print("\nUpdating medium memory/head...")
        train_head(
            model=model,
            head_name="medium",
            train_loader=medium_loader,
            optimizer=medium_optimizer,
            device=device,
            epoch_count=args.epochs,
            max_train_batches=args.max_train_batches,
        )

        # After fast and medium adaptation, update replay memory with current task.
        replay_df = update_replay_buffer(
            replay_df=replay_df,
            current_task_train_df=current_train_df,
            memory_per_class=args.memory_per_class,
            seed=args.seed + task_id,
        )

        print(f"Replay buffer examples after update: {len(replay_df)}")

        should_update_slow = (
            task_id % args.slow_update_frequency == 0
            or task_id == args.num_tasks - 1
        )

        if should_update_slow and len(replay_df) > 0:
            print("\nUpdating slow memory/head on replay buffer...")

            slow_train_df = replay_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

            slow_loader = make_loader(
                df=slow_train_df,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                shuffle=True,
                max_length=args.max_length,
            )

            train_head(
                model=model,
                head_name="slow",
                train_loader=slow_loader,
                optimizer=slow_optimizer,
                device=device,
                epoch_count=args.slow_epochs,
                max_train_batches=args.max_train_batches,
            )

            model.slow_trained = True
        else:
            print("\nSlow memory/head not updated on this task.")

        replay_df.to_csv(output_dir / f"replay_buffer_after_task_{task_id:02d}.csv", index=False)

        row_name = f"after_task_{task_id:02d}"

        for mode in modes:
            results_by_mode[mode][row_name] = {}

        seen_combined_accuracies = []

        print("\nEvaluating on seen tasks:")

        for eval_task_id in range(task_id + 1):
            eval_task_dir = data_dir / f"task_{eval_task_id:02d}"
            test_df = pd.read_csv(eval_task_dir / "test.csv")

            test_loader = make_loader(
                df=test_df,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                shuffle=False,
                max_length=args.max_length,
            )

            task_col = f"task_{eval_task_id:02d}"

            for mode in modes:
                if mode == "slow" and not model.slow_trained:
                    accuracy = np.nan
                else:
                    accuracy = evaluate(
                        model=model,
                        data_loader=test_loader,
                        device=device,
                        mode=mode,
                        max_eval_batches=args.max_eval_batches,
                        fast_weight=args.fast_weight,
                        medium_weight=args.medium_weight,
                        slow_weight=args.slow_weight,
                    )

                results_by_mode[mode][row_name][task_col] = accuracy

            combined_accuracy = results_by_mode["combined"][row_name][task_col]
            seen_combined_accuracies.append(combined_accuracy)

            print(
                f"    {task_col}: "
                f"combined={combined_accuracy:.4f} | "
                f"fast={results_by_mode['fast'][row_name][task_col]:.4f} | "
                f"medium={results_by_mode['medium'][row_name][task_col]:.4f} | "
                f"slow={results_by_mode['slow'][row_name][task_col]:.4f}"
            )

        avg_seen_combined_accuracy = float(np.mean(seen_combined_accuracies))

        metrics_rows.append(
            {
                "after_task": task_id,
                "average_seen_accuracy": avg_seen_combined_accuracy,
                "replay_buffer_size": len(replay_df),
                "memory_per_class": args.memory_per_class,
                "slow_trained": model.slow_trained,
            }
        )

        print(f"\nAverage combined accuracy on seen tasks: {avg_seen_combined_accuracy:.4f}")

        # Save intermediate results.
        for mode in modes:
            mode_matrix = pd.DataFrame.from_dict(results_by_mode[mode], orient="index")
            mode_matrix.to_csv(output_dir / f"{mode}_results_matrix.csv")

        combined_matrix = pd.DataFrame.from_dict(results_by_mode["combined"], orient="index")
        combined_matrix.to_csv(output_dir / "results_matrix.csv")

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        torch.save(
            {
                "fast_head": model.fast_head.state_dict(),
                "medium_head": model.medium_head.state_dict(),
                "slow_head": model.slow_head.state_dict(),
                "slow_trained": model.slow_trained,
            },
            output_dir / f"classifier_after_task_{task_id:02d}.pt",
        )

    results_matrix = pd.read_csv(output_dir / "results_matrix.csv", index_col=0)
    forgetting_df = compute_forgetting(results_matrix)
    forgetting_df.to_csv(output_dir / "forgetting.csv", index=False)

    replay_df.to_csv(output_dir / "replay_buffer_final.csv", index=False)

    torch.save(
        {
            "fast_head": model.fast_head.state_dict(),
            "medium_head": model.medium_head.state_dict(),
            "slow_head": model.slow_head.state_dict(),
            "slow_trained": model.slow_trained,
        },
        output_dir / "classifier_final.pt",
    )

    print("\n" + "=" * 80)
    print("Day 6 complete.")
    print("=" * 80)
    print(f"Saved results to: {output_dir}")

    print("\nFinal combined results matrix:")
    print(results_matrix.round(4))

    print("\nForgetting:")
    print(forgetting_df.round(4))

    if len(forgetting_df) > 0:
        avg_forgetting_including_final = forgetting_df["forgetting"].mean()
        avg_forgetting_excluding_final = forgetting_df[
            forgetting_df["task"] != f"task_{args.num_tasks - 1:02d}"
        ]["forgetting"].mean()

        print(f"\nAverage forgetting including final task: {avg_forgetting_including_final:.4f}")
        print(f"Average forgetting excluding final task: {avg_forgetting_excluding_final:.4f}")

    print("\nFinal average accuracy:")
    print(f"{results_matrix.iloc[-1].dropna().mean():.4f}")


if __name__ == "__main__":
    main()

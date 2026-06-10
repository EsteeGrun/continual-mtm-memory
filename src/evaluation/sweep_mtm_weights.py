"""
Day 7 script.

Goal:
Re-evaluate the Day 6 multi-timescale memory checkpoints with different
fast / medium / slow fusion weights.

Important:
This does NOT retrain the model. It loads the saved MTM checkpoints from Day 6
and tests different inference-time combinations of the fast, medium, and slow heads.

Run from project root:

    python src/evaluation/sweep_mtm_weights.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


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


class MTMClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.fast_head = nn.Linear(hidden_size, num_labels)
        self.medium_head = nn.Linear(hidden_size, num_labels)
        self.slow_head = nn.Linear(hidden_size, num_labels)

        self.slow_trained = False

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return outputs.last_hidden_state[:, 0, :]

    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.fast_head.load_state_dict(checkpoint["fast_head"])
        self.medium_head.load_state_dict(checkpoint["medium_head"])
        self.slow_head.load_state_dict(checkpoint["slow_head"])
        self.slow_trained = bool(checkpoint.get("slow_trained", False))

        self.to(device)

    def combined_logits(
        self,
        input_ids,
        attention_mask,
        fast_weight,
        medium_weight,
        slow_weight,
    ):
        h = self.encode(input_ids, attention_mask)

        logits_list = []
        weights = []

        if fast_weight > 0:
            logits_list.append(self.fast_head(h))
            weights.append(fast_weight)

        if medium_weight > 0:
            logits_list.append(self.medium_head(h))
            weights.append(medium_weight)

        if slow_weight > 0 and self.slow_trained:
            logits_list.append(self.slow_head(h))
            weights.append(slow_weight)

        if not logits_list:
            raise ValueError("At least one active memory weight is required.")

        weights = torch.tensor(weights, device=h.device, dtype=h.dtype)
        weights = weights / weights.sum()

        combined = sum(w * logits for w, logits in zip(weights, logits_list))
        return combined


def make_loader(df, tokenizer, batch_size, max_length):
    dataset = IntentDataFrameDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


@torch.no_grad()
def evaluate(model, loader, device, fast_weight, medium_weight, slow_weight):
    model.eval()

    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model.combined_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fast_weight=fast_weight,
            medium_weight=medium_weight,
            slow_weight=slow_weight,
        )

        predictions = torch.argmax(logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_forgetting(results_matrix):
    final_row = results_matrix.iloc[-1]
    rows = []

    for task_col in results_matrix.columns:
        values = results_matrix[task_col].dropna()

        if len(values) == 0:
            continue

        best_accuracy = values.max()
        final_accuracy = final_row[task_col]

        rows.append(
            {
                "task": task_col,
                "best_accuracy": best_accuracy,
                "final_accuracy": final_accuracy,
                "forgetting": best_accuracy - final_accuracy,
            }
        )

    return pd.DataFrame(rows)


def summarize_config(config_name, results_matrix, forgetting_df, memory_examples):
    final_average_accuracy = results_matrix.iloc[-1].dropna().mean()

    last_task = results_matrix.columns[-1]

    avg_forgetting_including_final = forgetting_df["forgetting"].mean()

    avg_forgetting_excluding_final = forgetting_df[
        forgetting_df["task"] != last_task
    ]["forgetting"].mean()

    return {
        "config": config_name,
        "final_average_accuracy": final_average_accuracy,
        "avg_forgetting_including_final": avg_forgetting_including_final,
        "avg_forgetting_excluding_final": avg_forgetting_excluding_final,
        "memory_examples": memory_examples,
    }


def main():
    data_dir = Path("data/processed/clinc150_stream")
    checkpoint_dir = Path("results/intent/mtm_frozen")
    output_dir = Path("results/intent/mtm_weight_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "distilbert-base-uncased"
    num_tasks = 10
    batch_size = 32
    max_length = 64
    memory_examples = 3000

    configs = [
        {
            "name": "medium_only",
            "fast_weight": 0.00,
            "medium_weight": 1.00,
            "slow_weight": 0.00,
        },
        {
            "name": "medium_heavy_95",
            "fast_weight": 0.025,
            "medium_weight": 0.95,
            "slow_weight": 0.025,
        },
        {
            "name": "medium_heavy_90",
            "fast_weight": 0.05,
            "medium_weight": 0.90,
            "slow_weight": 0.05,
        },
        {
            "name": "no_fast_90_10",
            "fast_weight": 0.00,
            "medium_weight": 0.90,
            "slow_weight": 0.10,
        },
        {
            "name": "fast_light_slow_10",
            "fast_weight": 0.05,
            "medium_weight": 0.85,
            "slow_weight": 0.10,
        },
        {
            "name": "fast_10_medium_85_slow_05",
            "fast_weight": 0.10,
            "medium_weight": 0.85,
            "slow_weight": 0.05,
        },
        {
            "name": "balanced_v1",
            "fast_weight": 0.15,
            "medium_weight": 0.70,
            "slow_weight": 0.15,
        },
    ]

    device = get_device()

    print("=" * 80)
    print("Day 7: MTM weight sweep")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")

    with open(data_dir / "global_to_label.json") as f:
        global_to_label = json.load(f)

    num_labels = len(global_to_label)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preload all test loaders.
    test_loaders = {}

    for task_id in range(num_tasks):
        test_df = pd.read_csv(data_dir / f"task_{task_id:02d}" / "test.csv")

        test_loaders[task_id] = make_loader(
            df=test_df,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
        )

    model = MTMClassifier(
        model_name=model_name,
        num_labels=num_labels,
    )
    model.to(device)

    summary_rows = []

    for config in configs:
        config_name = config["name"]
        fast_weight = config["fast_weight"]
        medium_weight = config["medium_weight"]
        slow_weight = config["slow_weight"]

        print("\n" + "=" * 80)
        print(f"Evaluating config: {config_name}")
        print(
            f"Weights: fast={fast_weight}, "
            f"medium={medium_weight}, slow={slow_weight}"
        )
        print("=" * 80)

        config_output_dir = output_dir / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for train_task_id in tqdm(range(num_tasks), desc=config_name):
            checkpoint_path = checkpoint_dir / f"classifier_after_task_{train_task_id:02d}.pt"

            model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=device,
            )

            row_name = f"after_task_{train_task_id:02d}"
            results[row_name] = {}

            for eval_task_id in range(train_task_id + 1):
                task_col = f"task_{eval_task_id:02d}"

                accuracy = evaluate(
                    model=model,
                    loader=test_loaders[eval_task_id],
                    device=device,
                    fast_weight=fast_weight,
                    medium_weight=medium_weight,
                    slow_weight=slow_weight,
                )

                results[row_name][task_col] = accuracy

        results_matrix = pd.DataFrame.from_dict(results, orient="index")
        forgetting_df = compute_forgetting(results_matrix)

        results_matrix.to_csv(config_output_dir / "results_matrix.csv")
        forgetting_df.to_csv(config_output_dir / "forgetting.csv", index=False)

        with open(config_output_dir / "weights.json", "w") as f:
            json.dump(config, f, indent=2)

        summary = summarize_config(
            config_name=config_name,
            results_matrix=results_matrix,
            forgetting_df=forgetting_df,
            memory_examples=memory_examples,
        )

        summary["fast_weight"] = fast_weight
        summary["medium_weight"] = medium_weight
        summary["slow_weight"] = slow_weight

        summary_rows.append(summary)

        print("\nSummary for config:")
        print(pd.DataFrame([summary]).round(4))

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["final_average_accuracy", "avg_forgetting_excluding_final"],
        ascending=[False, True],
    )

    summary_path = output_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best = summary_df.iloc[0]

    best_text = (
        f"Best config by final average accuracy: {best['config']}\n"
        f"Final average accuracy: {best['final_average_accuracy']:.4f}\n"
        f"Average forgetting excluding final task: {best['avg_forgetting_excluding_final']:.4f}\n"
        f"Weights: fast={best['fast_weight']}, medium={best['medium_weight']}, slow={best['slow_weight']}\n"
    )

    (output_dir / "best_config.txt").write_text(best_text)

    print("\n" + "=" * 80)
    print("Day 7 sweep complete.")
    print("=" * 80)
    print("\nSweep summary:")
    print(summary_df.round(4))

    print("\nBest config:")
    print(best_text)

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

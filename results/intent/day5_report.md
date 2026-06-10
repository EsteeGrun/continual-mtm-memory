# Day 5 Analysis Report

## Goal

Compare the Day 3 sequential frozen baseline against the Day 4 replay frozen baseline for continual CLINC150 intent classification.

## Main results

| Method | Final average accuracy | Avg forgetting excluding final task | Memory examples |
|---|---:|---:|---:|
| Sequential frozen | 0.0898 | 0.9370 | 0 |
| Replay frozen | 0.7836 | 0.1580 | 3000 |

## Interpretation

The sequential frozen baseline shows severe catastrophic forgetting. It learns each task when trained on it, but performance on earlier tasks collapses after later tasks.

The replay frozen baseline substantially reduces forgetting by storing a small buffer of old examples and mixing them with the current task during training.

Replay improves final average accuracy from 0.0898 to 0.7836.

Replay reduces average forgetting excluding the final task from 0.9370 to 0.1580.

## Research implication

Replay is a strong anti-forgetting baseline, but it requires storing examples. This creates a useful comparison point for the future multi-timescale memory method: the new method should aim to match or improve replay while using memory more efficiently, consolidating knowledge more systematically, or improving the stability-plasticity tradeoff.

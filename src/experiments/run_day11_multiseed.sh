#!/bin/bash

set -euo pipefail

BASE_DIR="results/intent/multiseed_3epoch_mpc20"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

SEEDS=(1 2 3 4 5)

echo "============================================================"
echo "Day 11: Multi-seed 3-epoch comparison"
echo "Seeds: ${SEEDS[@]}"
echo "Memory per class: 20"
echo "Epochs: 3"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running seed $SEED"
    echo "============================================================"

    SEED_DIR="$BASE_DIR/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    REPLAY_DIR="$SEED_DIR/replay_mpc20_e3"
    MTM_DIR="$SEED_DIR/mtm_mpc20_e3"

    if [ -f "$REPLAY_DIR/results_matrix.csv" ]; then
        echo "Replay result already exists for seed $SEED. Skipping replay."
    else
        echo "Running replay for seed $SEED..."
        python src/train/train_intent_replay_frozen.py \
          --output_dir "$REPLAY_DIR" \
          --num_tasks 10 \
          --epochs 3 \
          --memory_per_class 20 \
          --seed "$SEED" \
          2>&1 | tee "$LOG_DIR/replay_seed_${SEED}.log"
    fi

    if [ -f "$MTM_DIR/results_matrix.csv" ]; then
        echo "MTM result already exists for seed $SEED. Skipping MTM."
    else
        echo "Running tuned MTM for seed $SEED..."
        python src/train/train_intent_mtm_frozen.py \
          --output_dir "$MTM_DIR" \
          --num_tasks 10 \
          --epochs 3 \
          --slow_epochs 3 \
          --memory_per_class 20 \
          --fast_weight 0.0 \
          --medium_weight 0.9 \
          --slow_weight 0.1 \
          --seed "$SEED" \
          2>&1 | tee "$LOG_DIR/mtm_seed_${SEED}.log"
    fi

    echo "Finished seed $SEED"
done

echo ""
echo "============================================================"
echo "Day 11 multi-seed runs complete."
echo "============================================================"

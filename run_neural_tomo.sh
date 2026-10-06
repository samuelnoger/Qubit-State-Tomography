#!/bin/bash
set -e

# Usage: ./run_neural_tomo.sh [SHOTS]       (default 1000 shots per measurement setting)
# Every setting can be overridden from the environment, for example:
#   N_TRAIN=800000 STATES="mixed" ./run_neural_tomo.sh 1000
#   EPOCHS=200 ./run_neural_tomo.sh 1000
SHOTS=${1:-1000}
STATES=${STATES:-"pure mixed"}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-200000}
N_VAL=${N_VAL:-5000}
N_TEST=${N_TEST:-10000}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DEVICE=${DEVICE:-cpu}

echo "Neural tomography: shots=$SHOTS, n_train=$N_TRAIN, epochs=$EPOCHS, states: $STATES"

for STATE in $STATES; do
    echo "=========================================="
    echo "Processing State Type: $STATE"
    echo "=========================================="

    DATA_PATH="data/neural_tomo_${STATE}_${SHOTS}_${N_TRAIN}.pt"

    # 1. Generate Dataset (skipped if it already exists)
    if [ ! -f "$DATA_PATH" ]; then
        python -m data.generate_neural_tomo_data \
            --tomo-data-path "$DATA_PATH" \
            --tomo-state-type "$STATE" \
            --tomo-shots $SHOTS \
            --n-train-tomo $N_TRAIN \
            --n-val-tomo $N_VAL \
            --n-test-tomo $N_TEST
    else
        echo "Using existing dataset $DATA_PATH"
    fi

    # 2. Train Network
    python -m train.train_neural_tomo \
        --tomo-data-path "$DATA_PATH" \
        --device $DEVICE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LR \
        --tomo-hidden-dim $HIDDEN_DIM \
        --checkpoint-dir "checkpoints/neural_tomo_${SHOTS}_${N_TRAIN}_${EPOCHS}ep/"

    echo ""
done
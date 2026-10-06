#!/bin/bash
set -e

# Default strictly to the two necessary priors
STATES=${STATES:-"broad near_bell"}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-600000}
N_VAL=${N_VAL:-5000}
N_TEST=${N_TEST:-10000}
BATCH_SIZE=${BATCH_SIZE:-1024}
LR=${LR:-1e-3}
HIDDEN_DIM=${HIDDEN_DIM:-264}
N_LAYERS=${N_LAYERS:-4}

DEVICE=${DEVICE:-cpu}

echo "Neural tomography (Variable Shots): n_train=$N_TRAIN, epochs=$EPOCHS, states=$STATES"

for STATE in $STATES; do
    echo "=========================================="
    echo "Processing State Type: $STATE (Variable Shots)"
    echo "=========================================="

    DATA_PATH="data/neural_tomo_${STATE}_variable_${N_TRAIN}.pt"
    CHECKPOINT_DIR="checkpoints/neural_tomo_variable_${N_TRAIN}_${EPOCHS}ep/"

    # 1. Generate Dataset
    if [ ! -f "$DATA_PATH" ]; then
        python -m data.generate_neural_tomo_data \
            --tomo-data-path "$DATA_PATH" \
            --tomo-state-type "$STATE" \
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
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --tomo-n-layers $N_LAYERS

    echo ""
done
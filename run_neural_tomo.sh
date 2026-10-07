#!/bin/bash
set -e

# Default strictly to the two necessary priors
STATES=${STATES:-"broad near_bell"}
EPOCHS=${EPOCHS:-100}
N_TRAIN=${N_TRAIN:-400000}
N_VAL=${N_VAL:-5000}
N_TEST=${N_TEST:-10000}
BATCH_SIZE=${BATCH_SIZE:-2048}
LR=${LR:-1e-3}
HIDDEN_DIM=${HIDDEN_DIM:-264}
N_LAYERS=${N_LAYERS:-4}
LOSS_GAMMA=${LOSS_GAMMA:-0.5}


# Optional: draw the counts through a classifier's confusion matrix (channel-aware network), e.g.
#   READOUT_JSON=results/sweeps/readout_tomography.json STATES=near_bell ./run_neural_tomo_variable.sh
READOUT_JSON=${READOUT_JSON:-"results/sweeps/readout_tomography.json"}
READOUT_CLF=${READOUT_CLF:-cnn}

DEVICE=${DEVICE:-mps}      # if training fails on MPS (complex tensors), use DEVICE=cpu

GEN_EXTRA=""
TAG=""
if [ -n "$READOUT_JSON" ]; then
    GEN_EXTRA="--tomo-readout-json $READOUT_JSON --tomo-readout-classifier $READOUT_CLF"
    TAG="_${READOUT_CLF}"
fi

echo "Neural tomography (Variable Shots): n_train=$N_TRAIN, epochs=$EPOCHS, gamma=$LOSS_GAMMA, states=$STATES, readout='${READOUT_JSON:-none}'"

for STATE in $STATES; do
    echo "=========================================="
    echo "Processing State Type: $STATE (Variable Shots)"
    echo "=========================================="

    DATA_PATH="data/neural_tomo_${STATE}${TAG}_variable_${N_TRAIN}.pt"
    CHECKPOINT_DIR="checkpoints/neural_tomo_variable${TAG}_gamma_${LOSS_GAMMA}_${N_TRAIN}_${EPOCHS}ep/"

    # 1. Generate Dataset
    if [ ! -f "$DATA_PATH" ]; then
        python -m data.generate_neural_tomo_data \
            --tomo-data-path "$DATA_PATH" \
            --tomo-state-type "$STATE" \
            --n-train-tomo $N_TRAIN \
            --n-val-tomo $N_VAL \
            --n-test-tomo $N_TEST \
            $GEN_EXTRA
    else
        echo "Using existing dataset $DATA_PATH"
    fi

    # 2. Train Network (the first line printed should show the gamma you asked for)
    python -m train.train_neural_tomo \
        --tomo-data-path "$DATA_PATH" \
        --device $DEVICE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LR \
        --tomo-hidden-dim $HIDDEN_DIM \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --tomo-n-layers $N_LAYERS \
        --tomo-loss-gamma $LOSS_GAMMA

    echo ""
done
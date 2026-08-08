#!/usr/bin/env bash

# Run ten one-step DQN seeds in five consecutive two-process batches:
#   seeds 42-43, 44-45, 46-47, 48-49, and 50-51.
#
# Usage:
#   bash code/scripts/run_one_step_dqn_seeds.sh 1D

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

FREQUENCY="${1:-1D}"
if [[ "${FREQUENCY}" != "1D" && "${FREQUENCY}" != "1H" ]]; then
    echo "frequency must be 1D or 1H" >&2
    exit 1
fi

REWARD_METRIC="qlike"
TRAINING_STEPS=500000
BATCH_SIZE=128
LEARNING_RATE=0.001
GAMMA=0.99
REPLAY_CAPACITY=50000
LEARNING_STARTS=1000
TRAIN_FREQUENCY=4
TARGET_UPDATE_INTERVAL=1000
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY_STEPS=100000
EVAL_INTERVAL=1000
PATIENCE=50
DEVICE="cuda"

RUN_TAG="tf4_ed100k_ei1k_p50"
CHECKPOINT_DIR="checkpoints/dqn_${RUN_TAG}"
RESULTS_DIR="results/dqn_${RUN_TAG}"
LOG_DIR="logs"

TRAIN_SCRIPT="code/scripts/train_one_step_dqn.py"
DATA_FILE="data/processed/one_step_dqn_data_${FREQUENCY}.csv"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found in PATH" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "training script not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${DATA_FILE}" ]]; then
    echo "DQN data file not found: ${DATA_FILE}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

overall_status=0

run_seed() {
    local seed="$1"
    local log_path
    log_path="${LOG_DIR}/dqn_${FREQUENCY}_${REWARD_METRIC}_seed${seed}_${RUN_TAG}.log"

    echo "[$(date '+%F %T')] seed=${seed} started; log=${log_path}"

    uv run --frozen python -u "${TRAIN_SCRIPT}" \
        --frequency "${FREQUENCY}" \
        --reward-metric "${REWARD_METRIC}" \
        --seed "${seed}" \
        --training-steps "${TRAINING_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --gamma "${GAMMA}" \
        --replay-capacity "${REPLAY_CAPACITY}" \
        --learning-starts "${LEARNING_STARTS}" \
        --train-frequency "${TRAIN_FREQUENCY}" \
        --target-update-interval "${TARGET_UPDATE_INTERVAL}" \
        --epsilon-start "${EPSILON_START}" \
        --epsilon-end "${EPSILON_END}" \
        --epsilon-decay-steps "${EPSILON_DECAY_STEPS}" \
        --eval-interval "${EVAL_INTERVAL}" \
        --patience "${PATIENCE}" \
        --device "${DEVICE}" \
        --deterministic \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --results-dir "${RESULTS_DIR}" \
        >"${log_path}" 2>&1
}

run_batch() {
    local seeds=("$@")
    local pids=()
    local seed
    local index
    local pid

    echo "[$(date '+%F %T')] starting batch: ${seeds[*]}"

    for seed in "${seeds[@]}"; do
        run_seed "${seed}" &
        pids+=("$!")
    done

    for index in "${!pids[@]}"; do
        pid="${pids[${index}]}"
        seed="${seeds[${index}]}"
        if wait "${pid}"; then
            echo "[$(date '+%F %T')] seed=${seed} completed"
        else
            echo "[$(date '+%F %T')] seed=${seed} failed; check its log" >&2
            overall_status=1
        fi
    done

    echo "[$(date '+%F %T')] batch finished: ${seeds[*]}"
}

echo "One-step DQN seed scheduler"
echo "project=${PROJECT_ROOT}"
echo "frequency=${FREQUENCY}, reward=${REWARD_METRIC}"
echo "schedule: [42 43] -> [44 45] -> [46 47] -> [48 49] -> [50 51]"

run_batch 42 43
run_batch 44 45
run_batch 46 47
run_batch 48 49
run_batch 50 51

if [[ "${overall_status}" -eq 0 ]]; then
    echo "[$(date '+%F %T')] all seeds completed successfully"
else
    echo "[$(date '+%F %T')] scheduler finished with failed seeds" >&2
fi

exit "${overall_status}"

#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Minimal Qwen3-4B GDPO example with independent correctness and format rewards.
#
# Usage:
#   MODEL_DIR=/path/to/models DATA_DIR=/path/to/data EXP_DIR=/path/to/exp \
#     bash examples/algorithms/run-qwen3-4B-8xgpu-gdpo.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
   --ref-load "${MODEL_DIR}/Qwen3-4B/"
   --load "${EXP_DIR}/Qwen3-4B_gdpo/"
   --save "${EXP_DIR}/Qwen3-4B_gdpo/"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path examples.algorithms.gdpo_reward.reward_func
   --reward-key correctness
   --reward-keys correctness format
   --reward-weights 1.0 1.0
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --global-batch-size 256
   --rollout-max-response-len 8192
   --rollout-temperature 1
)

GDPO_ARGS=(
   --advantage-estimator gdpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   --entropy-coef 0.0
)

TRAIN_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --sequence-parallel
   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --colocate \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${GDPO_ARGS[@]}" \
   "${TRAIN_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" 2>&1 | tee "log/qwen3-4b-GDPO-gpu8-${now}.log"

#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B 1xGPU GDPO smoke test — GSM8K train only (no eval).
#
# Colocate mode: actor and rollout time-share the same GPU.
# One rollout is exactly one GDPO batch-normalization window:
#   ROLLOUT_BATCH_SIZE × N_SAMPLES = GLOBAL_BATCH_SIZE = 4 × 4 = 16.
#
# With the defaults, train_iters = NUM_ROLLOUT = 2. Increase NUM_ROLLOUT
# after the smoke test reaches two successful training iterations.
#
# Dataset: openai/gsm8k. The script strips the rationale from `answer` and
# keeps only the number after #### in train_clean.parquet on first run.
#
# Usage:
#   MODEL_DIR=/path/to/models DATA_DIR=/path/to/data \
#     bash examples/algorithms/run-qwen3-0.6B-1xgpu-gdpo-smoke.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: ${now}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export NUM_GPUS="${NUM_GPUS:-1}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/dev/gdpo-smoke}"
MODEL_DIR="${MODEL_DIR:-/your/model}"
DATA_DIR="${DATA_DIR:-/your/data}"

GSM8K_RAW="${DATA_DIR}/gsm8k/main/train-00000-of-00001.parquet"
GSM8K_CLEAN="${DATA_DIR}/gsm8k/main/train_clean.parquet"
if [ ! -f "${GSM8K_CLEAN}" ]; then
    python3 - <<EOF
import pandas as pd

df = pd.read_parquet("${GSM8K_RAW}")
df["answer"] = df["answer"].str.split("####").str[-1].str.strip()
df.to_parquet("${GSM8K_CLEAN}", index=False)
print(f"Wrote {len(df)} rows to ${GSM8K_CLEAN}")
EOF
fi

NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SYSTEM_PROMPT='Solve the problem step by step. Put only the final numeric answer inside \boxed{...}.'

if ((ROLLOUT_BATCH_SIZE * N_SAMPLES != GLOBAL_BATCH_SIZE)); then
    echo "GDPO requires ROLLOUT_BATCH_SIZE * N_SAMPLES == GLOBAL_BATCH_SIZE." >&2
    exit 2
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data "${GSM8K_CLEAN}"
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle
    --system-prompt "${SYSTEM_PROMPT}"
    --use-streaming-dataset

    --custom-rm-path examples.algorithms.gdpo_reward.reward_func
    --reward-key correctness
    --reward-keys correctness format
    --reward-weights 1.0 1.0

    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-response-len 1024
    --rollout-temperature 1
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
    --log-probs-max-tokens-per-gpu 4096
)

GDPO_ARGS=(
    --advantage-estimator gdpo
    --entropy-coef 0.0
    --eps-clip 0.2
    --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.45
    --sglang-load-format dummy
)

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "qwen3-0.6b-GDPO-gsm8k-1xgpu-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GDPO_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-0.6b-GDPO-gsm8k-1xgpu-${now}.log"

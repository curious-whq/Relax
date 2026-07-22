#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-0.8B single-GPU colocate GRPO smoke test on Open-R1 multimodal data.
# Actor and rollout time-share one GPU. Defaults intentionally run two rollout
# steps so the second rollout exercises the first post-training weight update.
#
# The unfused attention backend is a conservative default for consumer and Ada
# GPUs with Qwen3.5's 256-wide text attention heads. Set
# MEGATRON_ATTENTION_BACKEND=flash only after validating that the target GPU and
# Transformer Engine version support this head dimension.
#
# Usage:
#   MODEL_DIR=/path/to/models \
#   DATA_DIR=/path/to/data \
#   bash scripts/training/multimodal/run-qwen35-0.8B-1xgpu-openr1mm-colocate.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-0.8B.sh"

EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the directory containing Qwen3.5-0.8B}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the directory containing multimodal-open-r1-8k-verified}"
NUM_ROLLOUT="${NUM_ROLLOUT:=2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:=2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=2}"
MEGATRON_ATTENTION_BACKEND="${MEGATRON_ATTENTION_BACKEND:=unfused}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:=0.5}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.5-0.8B"
    --ref-load "${MODEL_DIR}/Qwen3.5-0.8B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --save "${EXP_DIR}/Qwen3.5-0.8B_mcore_1xgpu"
    --save-interval 100
    --max-actor-ckpt-to-keep 1
)

PROMPT_SET="${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type openr1mm
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 512
    --rollout-max-prompt-len 1024
    --rollout-max-context-len 1536
    --rollout-temperature 0.8
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --multimodal-keys '{"image":"image"}'
    --system-prompt "${SYSTEM_PROMPT}"
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --calculate-per-token-loss
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
    --train-memory-margin-bytes 0
    --no-rope-fusion
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-disable-cuda-graph
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "${MEGATRON_ATTENTION_BACKEND}"
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
    --balance-data \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    2>&1 | tee "log/qwen35-0.8b-GRPO-gpu1-colocate-${now}.log"

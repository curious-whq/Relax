#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Two-GPU Hybrid-async recipe for Task 22 performance A/B:
#   GPU 0: one Qwen3-0.6B Actor replica
#   GPU 1: one single-GPU SGLang rollout engine
#
# Baseline:
#   bash scripts/training/text/run-qwen3-0.6B-2xgpu-hybrid-async.sh
# Optimized:
#   HYBRID_STREAM_ACTOR_LOGPROBS=1 \
#     bash scripts/training/text/run-qwen3-0.6B-2xgpu-hybrid-async.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/task22-qwen3-0.6b-2gpu}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"
if [[ "${HYBRID_STREAM_ACTOR_LOGPROBS:-0}" == "1" ]]; then
   RUN_VARIANT=optimized
else
   RUN_VARIANT=baseline
fi
RUN_TAG="${RUN_TAG:-${RUN_VARIANT}-${now}}"
TIMELINE_DUMP_DIR="${TIMELINE_DUMP_DIR:-/tmp/relax-task22-timeline/${RUN_TAG}}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B/"
   --ref-load "${MODEL_DIR}/Qwen3-0.6B/"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save "${EXP_DIR}/task22-qwen3-0.6B-2xgpu/${RUN_TAG}/"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 64
   --use-fault-tolerance
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --timeline-dump-dir "${TIMELINE_DUMP_DIR}"
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "qwen3-0.6b-GRPO-gpu2-hybrid-async-${RUN_TAG}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [[ "${HYBRID_STREAM_ACTOR_LOGPROBS:-0}" == "1" ]]; then
   MISC_ARGS+=(--hybrid-stream-actor-logprobs)
fi

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --hybrid \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-0.6b-GRPO-gpu2-hybrid-async-${RUN_TAG}.log"

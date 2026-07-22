# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Qwen3.5-0.8B text-backbone configuration. The vision configuration is read
# from the Hugging Face checkpoint by Megatron Bridge.
MODEL_ARGS=(
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 8
    --num-query-groups 2
    --kv-channels 256
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 3584
    --use-gated-attention

    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --vocab-size 248320

    --rotary-base 10000000

    # Qwen3.5 specific
    --attention-output-gate
)

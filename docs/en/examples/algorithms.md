# Algorithm Reference

Relax supports multiple policy gradient algorithms, all selected via the `--advantage-estimator` flag. This document covers all integrated algorithms (for On-Policy Distillation, see the [dedicated page](./on-policy-distillation.md)).

All algorithms share the same training scripts — simply replace the `GRPO_ARGS` block in the script with the corresponding algorithm's arguments.

---

## GRPO

GRPO (Group Relative Policy Optimization) is the default algorithm in Relax. It broadcasts the group-relative scalar reward to every token and uses a standard PPO-Clip objective.

Reference: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).

### How It Works

The GRPO objective is the standard PPO-Clip:

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$, and $\hat{A}_t$ is the group-relative advantage (reward minus group mean, normalized by group standard deviation). Gradients are zeroed out when the ratio exceeds the clipping bounds.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator grpo` | default | Enable GRPO |
| `--eps-clip` | `0.2` | Clipping margin (ratio range = `[1-ε, 1+ε]`) |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin; can be set differently for asymmetric clipping |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

GRPO is the default algorithm — no parameter changes needed. Just run the training script directly:

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## GDPO

GDPO extends group-relative optimization to multiple independent rewards. It prevents one reward dimension from
dominating merely because it has a larger scale.

### How It Works

For every prompt group, GDPO standardizes each configured reward independently:

$$z_{i,k} = \frac{r_{i,k} - \mu_{g,k}}{\sigma_{g,k} + 10^{-4}}$$

It then combines the standardized rewards with optional weights and standardizes the resulting scalar once more
across the sequence batch:

$$a_i^\text{group} = \sum_k w_k z_{i,k}, \qquad
\hat{A}_i = \frac{a_i^\text{group} - \mu_\text{batch}}
{\sigma_\text{batch} + 10^{-4}}$$

Both standard deviations use sample statistics. A zero-variance reward dimension contributes zero. The final
sequence advantage is broadcast to its response tokens.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gdpo` | — | Enable GDPO |
| `--reward-keys` | — | Two or more keys returned by the reward function |
| `--reward-weights` | all `1.0` | Optional weights, one per reward key |
| `--reward-key` | first reward key | Primary raw reward used by existing logging and evaluation paths |

GDPO requires `global_batch_size` to be divisible by `n_samples_per_prompt`. The current implementation also
requires `rollout_batch_size * n_samples_per_prompt == global_batch_size`, so one rollout is exactly one batch-wise
normalization window.

### Quick Start

The minimal example returns independent `correctness` and `format` rewards:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash examples/algorithms/run-qwen3-4B-8xgpu-gdpo.sh
```

The essential configuration is:

```bash
--custom-rm-path examples.algorithms.gdpo_reward.reward_func
--advantage-estimator gdpo
--reward-key correctness
--reward-keys correctness format
--reward-weights 1.0 1.0
```

---

## CISPO

CISPO (Clipped Importance-ratio Soft Policy Optimization) preserves gradient signal for out-of-trust-region tokens instead of zeroing it out. It caps gradient magnitude via a stop-gradient'd coefficient while keeping the gradient direction alive.

Reference: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585).

### How It Works

The CISPO objective is:

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

where $\hat{r}_{i,t}(\theta)$ is the clipped importance-sampling weight:

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

and $r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$. Gradients flow **only** through $\log\pi_\theta$; both $\hat{r}_{i,t}$ and $\hat{A}_{i,t}$ are stop-gradient'd.

### Key Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--advantage-estimator cispo` | — | — | Enable CISPO |
| `--eps-clip` | `0.2` | `0.2` | Lower clipping margin (ratio lower bound = `1 - eps_clip`) |
| `--eps-clip-high` | same as `--eps-clip` | `10` | Upper clipping margin (ratio upper bound = `1 + eps_clip_high`). Set to `10` to effectively unclamp the upper side |
| `--kl-loss-coef` | `0.0` | `0.001` | KL loss coefficient. Recommended: `0.001` to add a small KL penalty that constrains policy drift |
| `--use-kl-loss` | off | on | Enable KL loss computation (required for `--kl-loss-coef` to take effect) |
| `--use-tis` | off | on | Token Importance Sampling — recommended to enable with CISPO |
| `--clip-grad` | — | `1.0` | Gradient clipping norm |

### Quick Start

Use any existing GRPO training script and replace `GRPO_ARGS` with `CISPO_ARGS`:

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)
```

---

## GSPO

GSPO (Group-wise Sequence-level Policy Optimization) differs from GRPO in how KL divergence is computed: GSPO uses **sequence-level** KL instead of per-token KL. Every token in a sequence shares the same KL value (the mean over all tokens in that sequence), providing uniform constraint strength within a sequence.

### How It Works

GSPO uses the same PPO-Clip objective as GRPO, but the ratio is computed from sequence-level KL:

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

Every token's ratio is $r_t = \exp(-\text{KL}_\text{seq})$, rather than an independent per-token ratio.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gspo` | — | Enable GSPO |
| `--eps-clip` | `0.2` | Clipping margin |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO (Soft Adaptive Policy Optimization) replaces hard clipping with a smooth sigmoid gate. The gate's steepness is controlled by a temperature parameter, implementing a differentiable trust region constraint.

### How It Works

SAPO's core is a sigmoid gate centered at ratio=1:

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

where $\sigma$ is the sigmoid function, and $\tau$ is selected based on the advantage sign:

- $A > 0$: use $\tau_\text{pos}$ (default 1.0)
- $A \leq 0$: use $\tau_\text{neg}$ (default 1.05, stronger suppression for negative tokens)

SAPO objective: $J_\text{SAPO}(\theta) = f(r) \cdot A$

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator sapo` | — | Enable SAPO |
| `--sapo-tau-pos` | `1.0` | Temperature for positive advantages |
| `--sapo-tau-neg` | `1.05` | Temperature for negative advantages (higher = stronger suppression) |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## Adding a New Algorithm

An algorithm is defined by one `AlgorithmSpec` in `relax/core/registry.py`. The spec registers five lazy hooks:

1. reward extraction;
2. reward post-processing;
3. advantage estimation;
4. policy-ratio construction;
5. policy objective.

Implement the hooks in a lightweight module, describe orchestration requirements with `AlgorithmCapabilities`, and
register the spec with `register_algorithm()`. CLI choices, reward processing, training dispatch, and the Controller's
service view are then derived from the registry. Add a validator when the algorithm has configuration invariants;
do not add a new algorithm-name branch to the training flow.

---

## Algorithm Comparison

| Algorithm | Advantage Computation | Policy Loss | KL Constraint |
|-----------|----------------------|-------------|---------------|
| **GRPO** | Group-relative reward | PPO-Clip (hard clip) | Optional KL loss |
| **GDPO** | Per-reward group normalization, then batch normalization | PPO-Clip (hard clip) | Optional KL loss |
| **CISPO** | Group-relative reward | Stop-gradient coefficient | Recommended KL loss |
| **GSPO** | Group-relative reward | PPO-Clip + sequence-level KL | Sequence-level ratio |
| **SAPO** | Group-relative reward | Sigmoid gate | Temperature-controlled |

## Next Steps

- [Quick Start](./quick-start.md)
- [On-Policy Distillation](./on-policy-distillation.md)
- [Generative Reward Model](./generative-reward-model.md)

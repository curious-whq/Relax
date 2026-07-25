# 算法参考

Relax 支持多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。本文档覆盖所有已集成的算法（OPD 在线策略蒸馏请参阅[单独文档](./on-policy-distillation.md)）。

所有算法共享同一套训练脚本——只需替换脚本中的 `GRPO_ARGS` 为对应算法的参数块即可。

---

## GRPO

GRPO（Group Relative Policy Optimization）是 Relax 的默认算法。将组内标量奖励广播到每个 token，使用 PPO 风格的裁剪目标函数。

参考论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。

### 算法原理

GRPO 目标函数为标准 PPO-Clip：

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$，$\hat{A}_t$ 是组相对 advantage（组内奖励减去组均值，按组标准差归一化）。当 ratio 超出裁剪边界时，梯度被直接置零。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator grpo` | 默认 | 启用 GRPO |
| `--eps-clip` | `0.2` | 裁剪边距（ratio 范围 = `[1-ε, 1+ε]`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距，可设为不同值以构造非对称裁剪 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

GRPO 是默认算法，无需修改参数。直接使用训练脚本即可：

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## GDPO

GDPO 将组相对优化扩展到多个相互独立的奖励维度，避免某个奖励仅因数值尺度更大而主导训练。

### 算法原理

对于每个 prompt 组，GDPO 分别标准化每个配置的奖励：

$$z_{i,k} = \frac{r_{i,k} - \mu_{g,k}}{\sigma_{g,k} + 10^{-4}}$$

然后使用可选权重合并各维度，并对合并后的标量做一次序列 batch 级标准化：

$$a_i^\text{group} = \sum_k w_k z_{i,k}, \qquad
\hat{A}_i = \frac{a_i^\text{group} - \mu_\text{batch}}
{\sigma_\text{batch} + 10^{-4}}$$

两次标准差都采用样本统计量。零方差的奖励维度贡献为零。最终的序列 advantage 会广播到对应 response token。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gdpo` | — | 启用 GDPO |
| `--reward-keys` | — | 奖励函数返回的两个或更多 key |
| `--reward-weights` | 全部为 `1.0` | 可选权重，数量必须与 reward key 一致 |
| `--reward-key` | 第一个 reward key | 供现有日志和评测链路使用的主原始奖励 |

GDPO 要求 `global_batch_size` 可被 `n_samples_per_prompt` 整除。当前实现还要求
`rollout_batch_size * n_samples_per_prompt == global_batch_size`，确保一次 rollout 恰好对应一个 batch-wise
归一化窗口。

### 快速开始

最小示例会分别返回 `correctness` 和 `format` 奖励：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash examples/algorithms/run-qwen3-4B-8xgpu-gdpo.sh
```

核心配置如下：

```bash
--custom-rm-path examples.algorithms.gdpo_reward.reward_func
--advantage-estimator gdpo
--reward-key correctness
--reward-keys correctness format
--reward-weights 1.0 1.0
```

---

## CISPO

CISPO（Clipped Importance-ratio Soft Policy Optimization）对超出信任域的 token 保留梯度信号，而非将其清零。通过 stop-gradient 系数限制梯度幅度，同时保留梯度方向。

参考论文：[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)。

### 算法原理

CISPO 目标函数为：

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

其中 $\hat{r}_{i,t}(\theta)$ 是裁剪后的重要性采样权重：

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

$r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$。梯度**只**流过 $\log\pi_\theta$，$\hat{r}_{i,t}$ 和 $\hat{A}_{i,t}$ 均被 stop-gradient 处理。

### 关键参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--advantage-estimator cispo` | — | — | 启用 CISPO |
| `--eps-clip` | `0.2` | `0.2` | 下方裁剪边距（ratio 下界 = `1 - eps_clip`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | `10` | 上方裁剪边距（ratio 上界 = `1 + eps_clip_high`）。设为 `10` 可近似取消上侧裁剪 |
| `--kl-loss-coef` | `0.0` | `0.001` | KL 损失系数。推荐设为 `0.001`，添加小幅 KL 惩罚以约束策略偏移 |
| `--use-kl-loss` | 关闭 | 开启 | 启用 KL 损失计算（`--kl-loss-coef` 生效的前提） |
| `--use-tis` | 关闭 | 开启 | Token Importance Sampling，推荐与 CISPO 同时开启 |
| `--clip-grad` | — | `1.0` | 梯度裁剪范数 |

### 快速开始

使用任意 GRPO 训练脚本，将 `GRPO_ARGS` 替换为 `CISPO_ARGS`：

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

GSPO（Group-wise Sequence-level Policy Optimization）与 GRPO 的区别在于 KL 散度的计算方式：GSPO 使用**序列级** KL 而非逐 token KL。每个 token 的 KL 值等于该序列所有 token KL 的均值，这为序列内所有 token 提供统一的约束强度。

### 算法原理

GSPO 使用与 GRPO 相同的 PPO-Clip 目标函数，但 ratio 的计算基于序列级 KL：

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

每个 token 的 ratio 均为 $r_t = \exp(-\text{KL}_\text{seq})$，而非各自独立的 token 级 ratio。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gspo` | — | 启用 GSPO |
| `--eps-clip` | `0.2` | 裁剪边距 |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO（Soft Adaptive Policy Optimization）用平滑的 sigmoid 门控替代硬裁剪。通过温度参数控制门控曲线的陡峭程度，实现可微的信任域约束。

### 算法原理

SAPO 的核心是一个以 ratio=1 为中心的 sigmoid 门控函数：

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

其中 $\sigma$ 是 sigmoid 函数，$\tau$ 根据 advantage 的符号选择不同的温度：

- $A > 0$: 使用 $\tau_\text{pos}$（默认 1.0）
- $A \leq 0$: 使用 $\tau_\text{neg}$（默认 1.05，对 negative token 更强的抑制）

SAPO 目标：$J_\text{SAPO}(\theta) = f(r) \cdot A$

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator sapo` | — | 启用 SAPO |
| `--sapo-tau-pos` | `1.0` | positive advantage 的温度参数 |
| `--sapo-tau-neg` | `1.05` | negative advantage 的温度参数（更高 = 更强抑制） |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## 接入新算法

每个算法由 `relax/core/registry.py` 中的一个 `AlgorithmSpec` 描述。该 spec 注册五个懒加载 hook：

1. 奖励提取；
2. 奖励后处理；
3. advantage 计算；
4. 策略 ratio 构造；
5. 策略目标函数。

在轻量模块中实现这些 hook，通过 `AlgorithmCapabilities` 描述编排要求，再调用 `register_algorithm()`
注册。命令行选项、奖励处理、训练分发和 Controller 服务视图都会从注册表派生。算法存在配置不变量时，
应额外注册 validator；不要在训练流程中新增算法名称分支。

---

## 算法对比

| 算法 | Advantage 计算 | 策略损失 | KL 约束方式 |
|------|---------------|---------|-----------|
| **GRPO** | 组相对奖励 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **GDPO** | 各奖励组内归一化，再做 batch 归一化 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **CISPO** | 组相对奖励 | Stop-gradient 系数 | 推荐 KL loss |
| **GSPO** | 组相对奖励 | PPO-Clip + 序列级 KL | 序列级 ratio |
| **SAPO** | 组相对奖励 | Sigmoid 门控 | 温度控制 |

## 下一步

- [快速开始](./quick-start.md)
- [在线策略蒸馏](./on-policy-distillation.md)
- [生成式奖励模型](./generative-reward-model.md)

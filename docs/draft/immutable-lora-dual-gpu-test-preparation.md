# No.7 双 GPU 自动验收

状态：2026-09-23，自动部署与测试代码已接入；本地无 GPU／完整 Ray 环境，目标机运行结果仍待验证。脚本存在不等于验收通过。

## 1. 一条命令

在已安装当前 Relax 和对应 SGLang patch 的容器中执行。先退出之前手动启动的两个测试引擎；脚本遇到所选 GPU 上已有计算进程会拒绝启动，不会替你杀掉其他任务。

```bash
cd /workspace/Relax
python3 -m tests.engine.lora.acceptance \
  --model /workspace/models/Qwen3-1.7B \
  --gpus 2 3 --deterministic \
  --output /workspace/no7-runs/auto-001
```

`2 3` 是容器内 `nvidia-smi` 的物理序号，即第 3、4 张卡。必须显式指定两张不同的空闲卡。输出目录必须不存在；下一次使用 `auto-002` 等新目录。省略 `--output` 则使用带时间和随机后缀的 `test-results/lora-*`。

不需要手工生成 adapter、启动 Ray、启动引擎、填写端口或 baseline URL。程序依次完成：生成并封存 A/B/C → 一步真实训练与已有 checkpoint 导出 → 按任务创建独立 Ray/Serve、真实 RM/SessionShard 和引擎 → 实验 → 保存报告 → 清理自己创建的资源。

每个任务从新的 A 默认版本开始，避免前一任务的缓存、fence 或引用污染后一任务。失败立即退出，已完成证据保留，剩余任务标记 NOT_RUN。退出码非零表示本次运行失败；只跑部分任务时 `full_acceptance=INCOMPLETE`，不能称完整通过。

## 2. 环境和资源

- 使用完整、未量化的本地 Qwen3-1.7B HF 目录，包含 config、全部权重和 tokenizer；此入口不联网下载模型，也不安装依赖。
- 无需下载训练数据集。固定的 32 个诊断输入、容差、KV 位置和性能门槛位于 `tests/engine/lora/acceptance/profile.json`。真实导出任务使用固定短文本做一步 optimizer 更新，目的为验证导出链路。
- 容器需要已经可用的 PyTorch、Ray/Serve、SGLang、Transformers、safetensors、PEFT 和仓库依赖。SGLang 必须应用当前 `docker/patch/sglang/v0.5.17.patch`。仅 `git pull Relax` 不会自动更新已安装的 SGLang patch。
- v0.5.17 tag 对象为 `b6a09f38fcc5e96574324b4acc19d421c539cfc6`，解引用的源码 commit 为 `29481685462732237d80d86076d6563e1f658102`。
- 使用新的本地 Ray 实例，不连接已有训练集群。真实 Agentic 服务会创建多个 ingress/Shard，CPU 资源要求由部署入口检查；Serve 使用现有服务发现约定的 8000 端口，端口占用时明确失败，不替换现有服务。
- 首轮为每个受管 engine TP=PP=DP=1、BF16、Triton LoRA、rank 8、q_proj/v_proj，CUDA graph 保持开启。每进程静态显存份额 0.2、最大 32 个运行请求、8192 KV tokens。适合两张 48GB 卡及此小模型，不代表任意大模型都能共驻留。
- 独立 A/B 基线使用单独的进程且保留 radix cache，以新 `extra_key` 建立零命中的冷前缀；受管引擎和普通性能基线同样保留缓存。性能任务每卡共驻留两个进程（受管和普通参考），测量窗口只驱动当前对照组，不让其他基线同时计算。
- 测试注入仅经 `tests/backends/sglang/lora_test_hooks` 加载到本次子进程，不写进生产 API。不要把该路径加入正常训练的全局 PYTHONPATH。

## 3. 六个可独立运行的任务

每行对应一个完整测试文件，共用部署、进程、fixture、观测和断言均在 `tests/` 下。

| `--tasks` 名称 | 文件（`tests/engine/lora/acceptance/`） | 验证内容 |
| --- | --- | --- |
| `publication` | `test_publication.py` | 单引擎失败及资源清理、旧 ACK、E2 延迟时仍绑定 A、取消后的迟到 prepare、重复请求、摘要冲突、提交后新 B／旧 A 数值 |
| `sessions` | `test_sessions.py` | 真实 Agentic 首次生成、外部工具等待式轮次、两轮续写、实际 pause/abort/resume、旧 A／新 B 的独立 logprob 对照；缺失 adapter 明确失败 |
| `capacity` | `test_capacity.py` | native pin/residency、容量 2 时 C 拒绝且 native prepare 为零、GPU 最后使用及 slot 清理等待、重复取消终态不回退、各 engine 仅卸载一次、C 复用 A slot 后 B/C 混合生成 |
| `numerical_cache` | `test_numerical_cache.py` | A/B 独立冷基线重复性和区分度、实际 K/V 差异、A→B／B→A 缓存隔离、同版命中正对照、错误 KV 负对照、混合 prefill/decode 和真实 graph replay |
| `performance` | `test_performance.py` | 普通 LoRA、受管 A 稳态、发布窗口、A/B 稳态；持续流量进展、发布时间、延迟、吞吐、失败数及资源占用；检查 pause/flush/abort-all 调用 |
| `export` | `test_export.py` | 一步真实训练 → 现有 checkpoint 保存及 PEFT 导出 → 封存 → 双引擎发布 → 带实际版本身份生成 |

例如只重跑缓存任务，仍会自动部署及清理：

```bash
python3 -m tests.engine.lora.acceptance \
  --model /workspace/models/Qwen3-1.7B --gpus 2 3 \
  --deterministic --tasks numerical_cache --output /workspace/no7-runs/cache-001
```

也可以通过 pytest 单独运行某个完整任务：

```bash
RELAX_LORA_MODEL=/workspace/models/Qwen3-1.7B \
RELAX_LORA_GPUS=2,3 RELAX_LORA_DETERMINISTIC=1 \
python3 -m pytest tests/engine/lora/acceptance/test_capacity.py -q -rs
```

不设置这两个环境变量时，pytest 显式 SKIP GPU 任务；普通 CPU 回归不会意外启动 GPU 服务。不要并行运行多个部署任务争用同一组 GPU／8000 端口。

## 4. 报告怎么看

- 根目录 `report.json`：总体状态、选定 GPU、各任务结果、缺失证据和进程清理记录。
- `artifacts/`：A/B/C 封存制品、摘要、publication 配置及固定验证 profile。A/B 是人为构造的数值 fixture，C 与 B 内容相同但版本 ID 不同。
- `training-export.log`、`training-export/export.json`：真实 optimizer/checkpoint/export 的结果及来源。
- `<task>/deployment.log`、`deployment.json`：Ray/Serve 启动日志与实际部署信息；启动失败首先看这里。
- `<task>/baseline-*.log`、`ordinary-*.log`：独立引擎日志；Agent launcher 另存日志。
- `<task>/report.json`、`evidence/`：逐项 PASS/FAIL/NOT_RUN、输入和原始数值、状态观测、故障注入、缓存／graph／slot 证据。

固定 `atol=rtol=1e-3`，KV 同样；fixture 区分度不足、负对照未能检出错误、混合 batch 或 graph 没实际执行，都按失败处理。性能门槛来自 profile，不能看过结果后调宽以获得 PASS。

失败时保留整个目录。脚本正常退出、异常及常规终止会清理自己的子进程／部署；主机断电或 SIGKILL 无法保证 Python 的 finally 执行，下一次 GPU 占用检查仍会拒绝盲目复用残留资源。

## 5. 验收范围

真实导出调用 `relax.backends.fsdp.checkpoint.save_checkpoint` 的 adapter-only 路径及 `export_peft_adapter`，实际 tensor 与更新后的权重逐项比较。这是单设备训练和现有导出路径的完整接入证据，**不是 Megatron 在线训练、分布式 FSDP 或完整 RL Controller 的验收**。

该自动 profile 也不验证 TP=2、PP、多 tokenizer、speculative 或 colocate/offload；这些需要分别安排资源和对照。两卡、两个 TP=1 engine 通过不能外推其他并行配置。Agent 测试使用确定性的外部等待模拟两轮之间的工具间隙，不以小模型是否自行生成 tool-call 作为判据。

旧的 `acceptance --config ... --report ...` 外部部署入口已被上述自动入口替代。底层 CPU 状态机、快照及原生协议回归继续保留，GPU 自动化复用其余既有验收逻辑，不把 mock 成功算作 GPU 通过。

## 6. 两小时性能对照与热点诊断

以下入口独立于六项验收，自动部署和回收引擎：

```bash
python3 -m tests.engine.lora.acceptance.overhead \
  --model /workspace/models/Qwen3-1.7B --gpus 2 3 \
  --deterministic --server-timings \
  --output /workspace/no7-runs/overhead-001
```

默认 12 组、每组四个 120 秒窗口，每窗口另加 30 秒预热；测量和预热合计两小时，启动、收尾与诊断需要额外时间。先做平衡顺序的普通／受管 A 稳态对照，再独立采集绑定、HTTP、原生时序、Python 栈与 CPU/GPU trace。CUDA graph 与缓存保持开启；`--server-timings` 对两组同时增加 metrics 观测成本。仅需定位时加 `--diagnostics-only`，只需正式吞吐时加 `--skip-diagnostics`，两者互斥。

重点查看 `performance/overhead.json`、逐窗口 `overhead-*.json` 与 `performance/diagnostics/report.json`。默认目标 0.05%，报告区间跨过目标时为 INCONCLUSIVE，不等于没有开销。退出码 0 表示达到目标（或仅诊断完成），1 表示超目标，2 表示精度不足或诊断不完整；部署和运行错误也非零退出。缺 `py-spy` 时保留其他证据并报告诊断 INCOMPLETE，脚本不自动安装依赖。

两组使用同一份已打补丁的 SGLang，因此结论只说明受管路径的增量成本，不能解释为整份补丁相对上游零开销。采样可帮助定位成本来源，最终仍需结合具体 trace 判断，不能根据总吞吐差额直接推断某个函数消耗。

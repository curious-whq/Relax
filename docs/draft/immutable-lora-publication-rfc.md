# RFC：不可变 LoRA 在线发布、Session 绑定与安全回收

状态：v2 重构待评审。CPU 状态机、HTTP 故障和持久化协议测试已运行；GPU 数值与性能验收脚本已提供，尚未运行。分支：`feat/immutable-lora-publication`。

## 1. 实现范围

固定基座、两个独立 SGLang 引擎、一个发布网关。旧 Session 持续使用首次生成绑定的 adapter；新版本只有在两个引擎均完成加载、pin 和 GPU 前向后才成为默认版本。实际生成显式携带 `lora_path`；版本缺失、引擎代次变化或后端终态未知时拒绝服务或保留占用，不使用基座或最新版本兜底。

首版提供可单独运行的 HTTP owner，无需 Ray 即可验证；Agentic Session 已接入该 owner。网关是单进程、单 event loop、单 worker，不能配置多个副本。引擎属于独立的推理 cohort，不得同时作为原生训练权重同步的目标。

当前支持 SGLang **0.5.17**、dense Qwen2/Qwen3/Llama、未量化固定基座、每个引擎 TP=DP=PP=1、单 tokenizer worker、**关闭 overlap scheduler**、非分离式推理、非流式单样本 token-ID 输入、`n=1`。禁止 gRPC/Rust server、LoRA overlap loading、层次缓存和推测解码等未经此协议覆盖的路径。Relax 自带的其他 SGLang 补丁可能影响数值或行为，GPU 报告需记录实际镜像/补丁版本。

| 文件 | 职责 |
| --- | --- |
| `relax/engine/lora/snapshot.py` | 已完成 HF adapter 导出的不可变快照、摘要、基座指纹 |
| `artifact.py`、`snapshot.py` | 版本化 PEFT 语义、来源 manifest、字节快照 |
| `execution.py`、`sglang_protocol.py` | scheduler 独立执行账本、FIFO 控制协议和排空证明 |
| `journal.py`、`outbox.py` | 单写者持久化、结果配额、可靠关闭责任 |
| `publication.py` | 默认指针、发布尝试、CAS 提交、容量和两类引用 |
| `control.py`、`http.py` | 引擎端 operation fence、请求去重、终态证据、owner/boot 校验 |
| `sglang_server.py` | 实际 SGLang 加载、pin、GPU 预热、生成和幂等卸载 |
| `gateway.py` | 单 owner HTTP 服务、Session 路由、请求对账和周期回收 |
| `cli.py` | HF 导出发布命令、现有 save-HF 回调、A/B fixture 生成 |
| `verify.py` | 双引擎真实 logprob、缓存隔离、取消续写、容量及流量实验 |
| `relax/agentic/session/service.py` | 首次生成绑定及终态释放 |
| `relax/agentic/pipeline/runtime.py` | 绑定、实际生成、abort/close 的网关传输 |

## 2. 不变量与状态转换

### 版本与文件

版本 ID 限制为路径安全的唯一标识。`AdapterSnapshot` 包含 `version_id`、`digest`、`base_model_digest`、`path`；公开 LoRA 名称为 `relax_policy@<version_id>`，底层内部 ID 为独立 operation UUID，不能原地替换。

```text
<store>/<version_id>/
  adapter_config.json
  adapter_model.safetensors
  producer_manifest.json
  manifest.json
```

SHA-256 覆盖基座指纹、规范化配置、producer manifest 和 safetensors 文件字节。生产者 manifest 声明训练基座指纹、基座配置指纹、导出 step、producer、格式契约及权重摘要；目标只比较来源事实，不代替导出者填写。它是受信任生产者的声明，不是训练过程的密码学证明。

`relax.dense-lora.v1` 只接受普通 `alpha/r` LoRA：显式模块列表、统一 rank、完整层覆盖、bias=none、单一 F16/BF16/F32 dtype。引擎加载前检查精确 tensor 名称、形状、offset 和文件长度。rsLoRA、DoRA、rank/alpha pattern、modules_to_save、额外参数、修改基座的初始化和未知配置字段均拒绝。只有字节摘要正确不构成语义验收。版本 ID 不参与内容摘要，因此 A/C 可以内容相同但身份不同。相同 ID、相同内容幂等；相同 ID、不同内容拒绝。文件序列化字节不同也算不同内容，不做 tensor 语义等价判断。

复制源必须是已经完成、停止修改的一次 HF 导出。临时目录写入后 fsync、rename，再同步父目录；已有版本不覆盖。引擎加载前重新验证摘要。只读文件权限不等于防篡改机制。共享文件系统必须支持这些操作，并保证发布期间不修改快照和基座目录。

基座指纹覆盖本地 HF 目录顶层的模型、配置和 tokenizer 文件；首版不处理任意自定义代码依赖图或在外部目录动态加载的权重。

### 发布

```text
PREPARING -- 全部 pinned + GPU ready --> PUBLISHED
    |                                      |
    | 失败/显式取消                         | 非默认、两类引用均为零
    v                                      v
RETIRING <----------------------------- RETIRING
    |
    +-- fence + absent 全部确认 --> ABORTED / RETIRED
    +-- 任一结果未知 -----------> CLEANUP_FAILED（继续占容量）
```

默认版本赋值是单 event-loop 的 CAS 决策点；成功 ACK 和后续发送均等待日志屏障。binding、默认 epoch 和发布结果在同一 SQLite 事务中持久化。昂贵准备不持有全局提交锁；每次接纳记录 expected_default_epoch，过期尝试不能覆盖较新提交。同版本并发发布共享 owned task；调用方断连或取消等待不会取消已经接受的发布。`cancel_publication` 必须携带 version_id 和 operation_id，迟到旧取消不能影响新尝试，仅在提交之前有效，不能回滚已发布的默认版本。已发布历史 ID 的幂等重试返回原 binding，不把 default 回退。

失败后向全部目标发送 retire，包括没有确认加载的目标。引擎在第一次 `await` 前安装 operation fence，等待原 prepare 收敛，然后卸载；迟到 prepare 不得重新创建资源。成功清理的目标不重复有效卸载，丢失 ACK 的目标允许幂等重试。清理不确定就保留容量。

物理加载额度由远端 operation 的实际完成证明释放，HTTP 超时不释放。scheduler load/unload 使用相关联的请求 ID；丢失 load 回包后，经同一 FIFO 通道查询加载结果。网关日志保存当前额度拥有者，重启先恢复这一观察，再接纳其他加载。没有调用全局 pause、flush cache 或 abort-all。单个 scheduler 的加载仍可能造成局部延迟，是否满足进度/性能标准必须由 GPU 实验判断。

### GPU 就绪与底层保护

SGLang 原始 load ACK 只证明 CPU LoRA 已加载，因此 prepare 还执行一次绑定该 adapter 的单 token GPU 前向。完成权重槽位拷贝和实际推理后才确认 ready。预热失败会撤销发布；预热已提交但结果未知时禁止卸载。

所有已接受版本在加载时设置 `pinned=True`。SGLang 的 CPU registry LRU 和 GPU 槽位淘汰均遵守 pin；逻辑容量 2 要求 `max_loras_per_batch >= 3` 且 `max_loaded_loras >= 3`，因为该版本最多允许 pin `max_loras_per_batch - 1` 个 adapter。

回收释放 adapter 权重和槽位；旧 KV 条目按各自 namespace 自然淘汰。磁盘实体与版本元数据分离：`archive_artifact` 仅删除已完成退休/撤销的实体文件，持久化 digest 和 operation 历史继续拒绝同 ID 冲突。默认实体配额 8 GiB；单 staging 作业在复制前接纳，容量不足返回错误。生产者封存目录也有独立的 8 GiB 配额。

wrapper 在专用 scheduler 中补充按内部 ID 的幂等卸载，清理加载失败留下的配置、权重、槽位和 pending event。只有全部 worker 返回成功，才确认 absent。此实现固定依赖 0.5.17 内部接口，升级 SGLang 必须重新审核。

### Session 与后端在途请求

Session 预创建不绑定。`AgenticSessionShard._chat` 完成输入校验、首次接纳生成时，在 Session 锁内绑定，并保存到 `SessionForest.static_metadata["lora_adapter"]`。工具轮次、重试及 abort/resume 复用 binding；网关再次校验 `expected_adapter`。

Session 引用覆盖工具间隙、park 和排队；物理请求引用独立覆盖实际后端执行。Session 关闭可释放前者，但后者只能由可信终态或“确定未提交”的证据释放。

- 首次放置按接纳负载选引擎，后续保留缓存亲和；前次执行未结算时禁止跨引擎迁移，结算后可迁移到持有同一版本的可用引擎。实际请求携带绑定版本的 `lora_path`。
- 网关在发送前创建 lease；内部 rid 固定长度散列，避免 SGLang 前缀 abort 意外影响其他请求。
- 网关和引擎都拥有 dispatch task，HTTP 客户端断连不销毁后端生命周期。
- 相同 rid、相同内容重试复用结果；内容不同拒绝，不盲重试物理生成。
- abort-before-generate 创建墓碑；未到达的迟到请求不会重新执行。
- abort ACK、HTTP 超时、客户端取消都不能作为后端终态。
- 请求发送情况不明时查询引擎 ledger；“当前 absent”还要安装 abort fence，才能认定迟到请求不会执行。
- 终态一旦结算，迟到 pending、unknown 或重复终态不能覆盖结果或重复减引用。

SGLang 的 TokenizerManager 在 tokenization 前取得真实 native registry 引用。结果返回、HTTP 断线或 coroutine 异常均不释放；只有 scheduler 独立账本的 drain proof 才释放一次。用户生成与 GPU 预热走同一执行账本；取消发布主动取消预热，retire 等待真实排空。

**GPU happens-before 范围**：只支持非 overlap、TP=1 的普通文本前向。scheduler 在 `process_batch_result()` 返回后才把已经发送终态的 execution 标为 drained；排队取消在处理结束后确认。此路径的结果拷贝/CPU token 读取已等待对应前向完成。grammar abort 仅设置 finished 标记不算排空；重复 IPC admission 不得给原请求制造 AbortReq 终态。没有引入全局 CUDA synchronize。这里的源码推导仍需实际 GPU 故障实验验证。

scheduler RID 是 tokenizer epoch + 单调序号，ACK 后压缩为连续水位和区间；一个长期未知请求不会导致后续每个 ACK 常驻内存。版本 fence 的清理依赖单 tokenizer FIFO：所有已发送工作先于 cleanup；上层持久化 operation fence 阻止新的旧版本接纳。

网关维护 active/pending、每 Session 请求索引及待 forget 索引。终态结果、lease inactive、dispatch inactive 原子提交后，才向引擎 ACK。历史去重记录落盘，大结果默认总配额 64 MiB，超出变为 RESULT_EXPIRED，旧 rid 不会重新执行。单 journal 默认上限 512 MiB；元数据耗尽会阻止新接纳，不做不安全 TTL 删除。

Session 关闭先写共享持久 spool，再允许本地业务 waiter 取消。网关验证 spool identity 与规范 URL，并独立消费关闭待办；close ACK 表示关闭意图已持久化，不表示 GPU 已排空。owner 使用 Ray actor ID + 启动 epoch；独立网关只在 Ray State 明确报告该 actor DEAD 后安装单调 fence 并关闭其 Sessions。网络不可达不等于死亡。dead owner 的清理责任在全部 Session close 落盘前保持可恢复。

### 容量和失败行为

| 情况 | 结果 |
| --- | --- |
| B 只有一个目标 ready | default 保持 A，新 Session 仍绑定 A |
| B 任一目标失败 | A 保持默认；清理未发布 B |
| A 有 Session 引用、B 默认，发布 C | HTTP 507，C 不加载 |
| A Session 已关闭，但后端请求未知 | HTTP 507，A 不提前回收 |
| A 全部目标确认卸载 | C 可以发布 |
| 目标缺少绑定版本 | HTTP 503 或明确拒绝，不回退 |
| 相同 ID 内容冲突、rid 内容冲突 | HTTP 409 |
| 确定未提交的非法生成 | HTTP 400，释放该请求引用 |
| owner 或 engine boot 不匹配 | 拒绝，不能空账本接管旧引擎 |

网关使用单写者文件锁和 SQLite WAL/FULL 日志。独立重启复用原 owner，恢复默认版本、准备尝试、Session 和未结算请求；部分 claim 失败也能使用同一 owner 重试。**引擎代次改变仍拒绝接管**：本版不提供远端进程树死亡证明或自动替换引擎，不能把新 boot 或健康检查失败当作旧 GPU 已死亡。需要运维确认旧实例全部停止后，显式建立新的 cohort；旧 Session 不会自动迁移到新代次。动态引擎集合、多网关 HA 不在范围。

控制与数据采用独立 HTTP 连接池；生成还有每引擎并发和 token 预留额度，未知执行继续占额度。默认每引擎 32 个请求、65536 个保守预留 tokens，每请求最多 8192 tokens，body/result 各 4 MiB。调用方必须显式指定 max_new_tokens。Agentic 启动从 cohort 读取实际并发容量，不再使用训练 rollout GPU 数推算。现有原 router program-admission 与 KV Session lifecycle 仍不组合；本协议使用完整 prompt + 版本 namespace 的普通 radix cache。

控制 token 和数据 token 必须不同，生产入口启动时校验。owner/boot 是一致性 fence，Bearer token 才是调用方权限；这是一个受信任租户内的两种操作角色，不提供租户级 Session ACL。公网部署还应在外层终止 TLS。

## 3. 导出与 Agentic 接入

接入现有 `write_hf_peft_adapter` / Megatron checkpoint 的 HF adapter 目录（`adapter_config.json`、`adapter_model.safetensors`）。可以发布直接 adapter 目录或包含 `lora_adapter/` 的 HF 导出目录。

手工导出方先 seal，再提交封存目录（MODEL_PATH 必须是训练时固定基座）：

```bash
python -m relax.engine.lora.cli seal \
  --export-dir "$COMPLETED_EXPORT_DIR" --model-path "$MODEL_PATH" \
  --store "$PRODUCER_STORE" --version-id "$VERSION_ID" \
  --export-step "$EXPORT_STEP" --producer "$EXPERIMENT_ID"
```

然后发布：

```bash
python -m relax.engine.lora.cli publish \
  --gateway-url "$GATEWAY_URL" \
  --export-dir "$PRODUCER_STORE/$VERSION_ID" \
  --version-id "$VERSION_ID"
```

自动导出回调使用现有参数，不修改 checkpoint 导出顺序：

```bash
export RELAX_LORA_PUBLICATION_URL="$GATEWAY_URL"
export RELAX_LORA_VERSION_PREFIX="$EXPERIMENT_ID"
# 由部署的 secret 管理机制注入 RELAX_LORA_CONTROL_TOKEN；不得提交到仓库。
# 在已有训练脚本中设置：
# --save-hf "$HF_EXPORT_ROOT"
# --save-hf-post-hook-path relax.engine.lora.cli.export_hook
```

回调先同步封存当前导出，再把不可变引用交给单后台线程排队，最多保留两个未完成发布，版本 ID 为 `<prefix>-<rollout_id>`；队列满则拒绝本次入队并记录失败，可稍后显式重试。final save 调用 `flush()` 等待队列收敛。现有 save-HF 框架会记录并吞掉回调异常，因此训练继续不代表发布成功，应读取网关 `/state`。HTTP 等待超时也不代表后台发布已撤销。

训练作业正常使用的权重同步引擎必须与本 cohort 分开；此功能没有替换现有固定名称的训练权重更新流程。不要把 managed 引擎注册为原生 updater 的目标，否则其写接口会被拒绝。

Agentic 作业增加：

```text
--use-agentic-rollout
--lora-publication-url <网关 URL>
```

Agentic Shard 另需设置 `RELAX_LORA_DATA_TOKEN` 和 `RELAX_LORA_CLOSE_OUTBOX`（与网关共享的持久目录）。网关必须配置规范 public-url、相同 spool 和 Ray State 地址。其余 agent、tokenizer、数据源和 Ray 配置沿用已有可运行作业。启动生成前先发布 A。不要同时启用 `--agentic-session-lifecycle` 或基于原 router 指标的 `--agentic-program-admission`；参数验证会拒绝这两种组合。这里的版本引用与原生 KV Session hint 独立，完整 prompt 仍然是生成输入。

普通 Session 请求经网关，不经原 router 的 HTTP 重试或分布式 fallback。KV cache 依赖 SGLang 内部独立 LoRA ID 的 namespace；实验必须用真实输出验证，不能只检查版本日志。

## 4. 单机运行与 GPU 验收

**不需要两台机器。推荐单机两张 GPU，每张运行一个 TP=1 引擎。** 一张 GPU 显存够时也可以同时运行两个独立小模型引擎；需要降低各引擎的显存预算，报告明确标注共卡资源竞争。一个引擎无法覆盖两个目标的发布语义。

准备已安装依赖的 SGLang 0.5.17 / Relax 环境、本地固定基座（建议小型 dense Qwen2/Qwen3/Llama）、两引擎可访问的相同绝对路径存储。以下变量由运行者设置，不能直接复用其他集群的配置：

| 变量 | 含义 |
| --- | --- |
| `MODEL_PATH` | 固定 HF 基座目录 |
| `ENGINE_A_STATE`、`ENGINE_B_STATE` | 各引擎独立的持久元数据目录 |
| `GATEWAY_STATE` | 网关主机上的持久日志目录；与共享 adapter store 分离，SQLite WAL 不放在 NFS 上 |
| `CLOSE_OUTBOX`、`RAY_STATE_ADDRESS` | Agentic 与网关共享的持久关闭 spool；明确的 Ray dashboard 地址 |
| `RELAX_LORA_CONTROL_TOKEN`、`RELAX_LORA_DATA_TOKEN` | 从 secret 管理注入的不同 Bearer 凭证 |
| `FIXTURE_DIR`、`SNAPSHOT_STORE` | fixture 输出和不可变快照目录，放在基座目录之外 |
| `GPU_A`、`GPU_B` | 各引擎可见 GPU ID；单卡验证可相同 |
| `ENGINE_HOST`、`ENGINE_A_PORT`、`ENGINE_B_PORT` | 引擎绑定地址与不同端口 |
| `ENGINE_A_URL`、`ENGINE_B_URL` | 对应 HTTP 地址 |
| `MEM_FRACTION` | 按实际显存预先确定的每引擎预算 |
| `GATEWAY_HOST`、`GATEWAY_PORT`、`GATEWAY_URL` | 正常服务模式的网关地址 |
| `ATOL`、`RTOL`、`MAX_PROGRESS_GAP` | 实验前冻结的数值容差和最大全局完成进度间隔（秒） |
| `TOPOLOGY` | GPU 型号、数量、放置方式、镜像与 patch 版本的说明 |
| `REPORT_PATH` | 实验 JSON 输出 |

生成两份有固定随机种子、LoRA B 矩阵符号相反的 fixture（rank 8）：

```bash
python -m relax.engine.lora.cli fixtures \
  --model-path "$MODEL_PATH" --output "$FIXTURE_DIR" --rank 8
```

在两个独立终端启动引擎；通过 wrapper 启动才能获得所需的 fencing 与终态契约：

```bash
CUDA_VISIBLE_DEVICES="$GPU_A" python -m relax.engine.lora.sglang_server \
  --engine-id engine-a --version-capacity 2 --state-dir "$ENGINE_A_STATE" \
  --model-path "$MODEL_PATH" --host "$ENGINE_HOST" --port "$ENGINE_A_PORT" \
  --tp-size 1 --disable-overlap-schedule --enable-lora --max-lora-rank 8 \
  --lora-target-modules q_proj v_proj \
  --max-loras-per-batch 3 --max-loaded-loras 3 \
  --mem-fraction-static "$MEM_FRACTION"
```

```bash
CUDA_VISIBLE_DEVICES="$GPU_B" python -m relax.engine.lora.sglang_server \
  --engine-id engine-b --version-capacity 2 --state-dir "$ENGINE_B_STATE" \
  --model-path "$MODEL_PATH" --host "$ENGINE_HOST" --port "$ENGINE_B_PORT" \
  --tp-size 1 --disable-overlap-schedule --enable-lora --max-lora-rank 8 \
  --lora-target-modules q_proj v_proj \
  --max-loras-per-batch 3 --max-loaded-loras 3 \
  --mem-fraction-static "$MEM_FRACTION"
```

数值验收使用两个**新启动且尚未被其他网关 claim** 的引擎；脚本自身持有 owner，不要同时启动正式 gateway：

```bash
python -m relax.engine.lora.verify \
  --engine-url "$ENGINE_A_URL" --engine-url "$ENGINE_B_URL" \
  --model-path "$MODEL_PATH" --fixtures "$FIXTURE_DIR" --store "$SNAPSHOT_STORE" \
  --report "$REPORT_PATH" --atol "$ATOL" --rtol "$RTOL" \
  --max-progress-gap "$MAX_PROGRESS_GAP" --concurrency 4 \
  --topology "$TOPOLOGY" --timeout-seconds 900
```

容差参数没有默认值。先用独立试运行校准噪声，再固定正式验收容差；不得看到正式结果后放宽。脚本会检查三次重复基线的一致性，以及 A/B 差异大于容差尺度的十倍，否则失败。每轮验证后重新启动两引擎，不能复用旧 owner。

实验包含：

1. 每个引擎分别只加载 A / B，建立固定候选 token 的 logprob 基线；新 operation ID 建立冷 namespace，不发 flush。
2. 持续对两个引擎发送 A 请求；B 只有第一个引擎加载时设置测试门，验证新 Session 仍数值匹配 A。
3. 放开测试门，B 完整发布后验证新 Session 匹配 B、旧 Session 匹配 A。
4. 为同一引擎构造新旧 Session，在同输入上做 A→B、B→A 数值比较，记录 cache hit；没有实际缓存复用则验收失败。
5. 中断一条较长的 A 请求，随后在原 Session 再次生成并比较 A 基线。
6. A 仍有引用时发布 C 必须容量失败；A 全部释放后再发布 C。
7. 记录发布耗时、测试门释放后的耗时、生成完成时间、单 token 请求延迟分位数、失败、cache hit 和版本/引用占用序列；发布期间必须有完成进度且间隔不超过预设阈值。

单 token 非流式请求延迟用于近似观察首 token 延迟，不声称测得流式 TTFT。报告包含测试门等待时间，另列释放后的发布耗时。CPU 测试统计每引擎有效卸载次数；GPU 实验验证数值和容量结果。磁盘加载造成的性能退化不会被隐藏，未达到预设进度标准会报失败。

也可把上述命令参数保存为 JSON（key 用下划线，`engine_url` 为两项列表），然后运行：

```bash
RELAX_LORA_GPU_CONFIG="$GPU_CONFIG_JSON" python -m pytest tests/engine/lora/test_gpu_publication.py -q
```

正式服务模式需另起一批新引擎，然后启动 gateway：

```bash
python -m relax.engine.lora.gateway \
  --engine-url "$ENGINE_A_URL" --engine-url "$ENGINE_B_URL" \
  --model-path "$MODEL_PATH" --store "$SNAPSHOT_STORE" --capacity 2 \
  --state-dir "$GATEWAY_STATE" \
  --max-lora-rank 8 --target-module q_proj --target-module v_proj \
  --public-url "$GATEWAY_URL" --close-outbox-dir "$CLOSE_OUTBOX" \
  --ray-state-address "$RAY_STATE_ADDRESS" \
  --host "$GATEWAY_HOST" --port "$GATEWAY_PORT"
```

再通过 CLI 发布 A，启动已配置 `--lora-publication-url` 的 Agentic 作业，后续发布 B。`GET /state` 返回 default、epoch、版本占用、两类引用、ready/cleanup 目标和请求进度；`POST /collect` 主动对账，后台也每秒对账。控制命令还有 `bind_session`、`abort_request`、`close_session`、`cancel_publication`；这些命令不是 OpenAI 兼容服务，外部 Agent 仍通过 Relax Session API。

## 5. 测试状态与后续边界

```bash
python -m pytest tests/engine/lora -q
python -m pytest tests/test_agentic_rollout.py -q
```

第一组可在 CPU 上运行状态机、HTTP 丢包/故障、SGLang 分发边界、容量、导出回调测试；其中真实 GPU 实验在未配置环境时 skip，现有 exporter 的实际格式测试在没有 PyTorch/safetensors 时 skip。fake 权重和 fake backend 不代表 GPU 推理通过。

另有验收脚本自身的 CPU 编排测试：正常输出能够完成实验，注入跨版本错误数值必须失败；这用于检查脚本，不替代真实 GPU 证据。

第二组依赖完整 Relax 的 Ray/PyTorch 等运行环境，包含 Agentic 网关传输和终态 close 的回归测试。本次本机环境缺少这些依赖，未运行完整 Agentic/Ray 集成测试，也没有运行多节点训练或 GPU 性能实验。不能将提交的脚本视为已通过验收的实验报告。

仍未验证：实际 GPU 的数值、缓存与吞吐；完整 Ray/Agentic 进程死亡和网络故障实验。仍未实现：引擎代次变更后的可信死亡证明与自动重建、动态 cohort、跨网关 HA、更多并行模式、把 CPU adapter 解析完全移出 scheduler。非 overlap 限制可能降低吞吐，必须报告与原生 overlap 的差异，不能把安全性修正表述为无性能代价。

## 参考

- [Miles #3127](https://github.com/radixark/miles/pull/3127)：staged adapter publication 的参考实现。
- [SGLang LoRA 文档](https://docs.sglang.io/docs/advanced_features/lora)：动态加载、pin 和容量约束。
- [SGLang v0.5.17 TokenizerManager](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/managers/tokenizer_manager.py)：请求分发与终态路径。
- [SGLang v0.5.17 LoRA memory pool](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/lora/mem_pool.py)：GPU 驻留和淘汰保护。

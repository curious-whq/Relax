# 不可变 LoRA 发布：人工 Review 记录

原始审查基准：`621a59a4ff134b2187db5f0c1eb35dfb3ba8696a`（`feat/immutable-lora-publication`）。各轮结果按时间追加；当前状态与本次拆分见文末 R008/R009。

设计参考：[RFC 与验证边界](./immutable-lora-publication-redesign-rfc.md)。

## 记录规则

- 用户通过提问逐步审查，助手依据实际代码解释行为、理由和局限。
- 用户明确指出需要修改或不合理的地方时，当轮更新本文，保留其意见原意；普通提问不自动视为修改决定。
- 每项使用稳定编号，记录文件及函数、具体问题、用户要求的修改方向和状态。尚未确定的方案写明“待讨论”，不替用户作决定。
- 同一问题的后续讨论更新原条目；用户改变决定时保留简短说明，避免遗失上下文。
- 记录意见不等于修改代码。收到实施要求后再改；完成时补充实际改动和验证结果，不把“已记录”标成“已解决”。

## 修改清单

### R001：将 verify.py 的验收逻辑去重并整合到 tests

- **位置**：`relax/engine/lora/verify.py`，尤其是近千行的 `verify()`；关联 `tests/engine/lora/` 和 `tests/backends/sglang/` 中的测试。
- **用户意见**：“第一件事情，需要将 veriy 这个文件看看和那些 test 文件有重合的，做到 test 里面，而不是像现在一样。”此处指 `verify.py`。
- **问题与影响**：该文件承担测试和验收职责，却放在业务代码目录中；大量实验与辅助逻辑集中在一个函数里，难以单独审查和运行。与现有测试的具体重合范围尚待逐项核对。
- **修改方向**：
  1. 对照现有测试，梳理 `verify.py` 各项检查的覆盖内容、依赖及重复部分。
  2. 将测试和验收逻辑整合到 `tests/` 的合适位置，合并重复用例及公共辅助逻辑，拆开当前过大的函数。
  3. 保留必要的真实引擎、GPU 数值及性能验证；不能因 CPU 单元测试名称或场景相似就删除不同层级的验证。
  4. 同步处理原模块的调用方、测试入口和文档命令。是否还需要保留薄的命令行入口，待梳理后讨论。
- **状态**：代码已调整，等待整体 review。
- **落实与验证**：2026-09-20 将验收逻辑移到 `tests/engine/lora/acceptance/`，分离数值断言、实际请求辅助、故障／缓存场景和性能测量；删除生产 `verify.py` 并更新调用方。CPU 数值与 CLI 回归通过，GPU 未运行。CPU 状态机与 GPU 同名场景承担不同证据责任，均保留。

### R002：增加性能测试，发布模式必须保留 CUDA graph

- **位置**：`relax/backends/sglang/sglang_engine.py` 的 publication 启动配置；`docker/patch/sglang/v0.5.17.patch` 的原生配置限制及 graph 执行／排空路径；关联性能测试、示例配置和 RFC。
- **用户意见**：“需要增加性能测试，同时 cuda graph 一定不能关掉。”
- **问题与影响**：当前 publication 分支强制关闭 decode／prefill CUDA graph，原生补丁也要求关闭；这可能影响没有发布操作时的日常推理性能，实际影响尚未测量。当前关闭 graph 的首版取舍不再符合用户要求。
- **修改方向**：
  1. 发布功能不得通过关闭 CUDA graph 来规避实现或验证问题；移除强制关闭配置及对应原生限制，补齐 graph 开启时的正确性和安全回收支持，不能只删除检查。
  2. 覆盖 graph replay 下的 A/B 混合请求、LoRA 索引及 rank/scaling 更新、A 退休后 C 复用槽位，以及最后一次 GPU 使用完成后才允许回收的时序。
  3. 在 `tests/` 中增加或整合性能对照测试，与 R001 的整理统一进行。比较原有配置基线、开启发布功能但无发布操作的稳态，以及持续流量下的实际发布窗口；CUDA graph 保持开启，硬件、输入、并发及其他配置可比。
  4. 记录吞吐、延迟、生成进度、发布耗时、失败数和资源占用，区分功能常驻开销与发布期间的额外开销。具体性能门槛待讨论，不能用估计值冒充实测结果。
  5. 同步更新 RFC、运行限制与实验配置，不再将“关闭 CUDA graph”列为支持该功能的前提。
- **状态**：graph 配置与测试接线已调整；真实 graph／性能结果待用户运行。
- **落实与验证**：移除强制关闭 decode/prefill graph 的参数及 native 拒绝项，保留 replay 之后的 forward-stream event。测试要求每台目标实际 replay，增加普通 LoRA 服务与受管稳态的性能对照及独立引擎停顿检查。性能门槛采用 profile 预登记值；CPU 事件测试不作为 GPU 通过证据。

### R003：cli.py 不允许包含测试逻辑

- **位置**：`relax/engine/lora/cli.py`，包括 `make_fixtures()`、`fixtures` 子命令的参数注册和执行分支；关联测试调用方及文档示例。
- **用户意见**：“第三条，测试不允许放到 cli.py 中。”
- **问题与影响**：当前 LoRA 管理 CLI 混入测试权重生成逻辑，日常操作入口与测试工具职责混杂。
- **修改方向**：
  1. 将测试 fixture 生成及其他测试专用逻辑移出 `cli.py`，整合到 `tests/` 的合适位置，与 R001 的测试整理统一进行。
  2. 移除管理 CLI 中的 `fixtures` 子命令及对应分支；保留实际导出、封存、发布、查询、取消和回收等管理命令。
  3. 同步更新测试入口、调用方和文档命令，保留生成可复现 A/B 测试制品的能力。
- **状态**：代码已调整，等待整体 review。
- **落实与验证**：fixture 生成移到 `tests/engine/lora/acceptance/fixtures.py`，管理 CLI 不再包含该子命令；生成器只计算一次基座指纹，更新了文档命令。


### R004：扩展训练与资源部署模式

- **用户要求**：支持同步、hybrid、colocate、train offload 和 rollout offload。
- **当前改动**：同步／hybrid／fully-async 在已有 all-rank RPC 中接收导出意图，worker 完成 optimizer 更新后、sleep 前封存；bootstrap 同样在首次让出训练 GPU 前导出，服务连接 RM 后只提交已封存制品。训练 RPC 期间到达的手动导出留到下个边界；全 rank 结果不一致不发布。publication 分支跳过旧 rollout 权重通信，保留 actor/ref 同步与备份。
- **内存移交**：rollout suspend/resume 复用既有管理入口，携带固定 cohort/boot 与单调序号。先封闭新绑定和发布、等待已接纳的控制与实际执行收敛；LoRA pool 纳入 weights CPU-backup，Session binding/ref/pin/容量保持。显式 offload 在释放 KV 显存前重置缓存元数据；恢复后用完整 token 前缀重算，不重新绑定版本。正常发布不调用这条路径。
- **容错**：旧 HTTP/IPC 移交序号不能在恢复后再次 offload；每个 memory tag 前后汇合 rank 结果，局部异常不让其他 rank 继续进入单边 barrier。未知/部分失败保持准入关闭与资源占用。分阶段 resume 只在全部目标、全部 tags 完成后开放；健康监控识别计划内 suspension。
- **配置与边界**：放开 colocate 和 rollout offload；共享训练/推理 GPU 必须同时启用 train/rollout offload，并遵循已有资源交接 barrier。独立推理 GPU 可在下一轮生成前恢复。colocate 本身按步骤交替占用 GPU，不能把该部署说成训练与生成始终同时执行。
- **验证**：CPU 覆盖迟到旧移交、waiter 取消、分 tag 恢复、跨 rank 阶段异常、悬挂 Session close 和引用保持。真实 Actor/Megatron 类接线回归因完整依赖缺失跳过；尚需真实训练、memory-saver、graph/LoRA 恢复数值实验，不能仅凭放开参数声称运行验收通过。

### R005：保留原生 LoRA 兼容的 GPU／并行能力

- **用户要求**：单卡多引擎、多卡单引擎、更多引擎，以及 TP/CP/PP/SP/DCP/PCP、CUDA graph、spec 等原有 LoRA 能力。
- **当前改动**：固定目标数放宽到至少两个，状态机测试覆盖 2/3/8 目标；保留 CUDA graph 和原生支持的 NGRAM speculative 分支。TP 的所有执行 rank 经 ProbeBoots→CommitOwner 建立共同身份，load／retire／Session close／request drain 必须汇合全部 rank 的匹配回执；专用完成通道不改变原有 leader token 输出。跨节点通道使用已有 rendezvous host。迟到旧 commit 在重启 worker 上不能安装旧 owner。
- **单卡多引擎**：publication YAML 新增 `engines_per_gpu`，默认 1；设为 2 时，一张推理 GPU 可部署两个独立 TP=1 引擎。同步调整 PG 物理槽索引、Ray 资源份额、端口布局和 Agentic permit 总量。必须显式设置每引擎 `--sglang-mem-fraction-static < 1 / engines_per_gpu`；这不是模型必然装得下的保证。多卡单引擎继续使用既有 `--rollout-num-gpus-per-engine`。
- **TP 故障边界**：正常缺版本在接纳／全 rank prepare 时明确失败；已接纳且 pinned 的实例若在某 rank 的 batch 前意外丢失，视为引擎内部不变量破坏，触发原生进程故障监督并保留协调者 UNKNOWN，占用不提前释放。不能在单 rank 局部过滤 batch，使其他 rank 进入不一致的 collective；正常路径也不新增每步 CPU all-reduce。TP 当前不承诺这种内部损坏的单请求隔离恢复。
- **PP／异步加载**：PP 微批使用实际 forward event，排空扫描所有 microbatch；内存操作先沿 PP 传递，再在各 stage 内汇合 TP 就绪，避免接收 handler 阻断后续 stage。worker 身份与完成回执覆盖 TP×PP。非 overlap scheduler 和原生 LoRA H2D overlap loading 不再被额外关闭；独立保存 copy event，避免原生事件索引提前删除导致误报 READY。
- **DP attention**：当前只接通 Qwen3 attention projections（实际 wrapped 模块限 qkv/o、Triton LoRA）。Session 固定 DP route；每个 DP 组分别真实预热；请求排空等待参与的 TP/CP 子组及全部 PP stage，版本与 Session 控制仍等待整个 engine。idle CUDA graph 前清零静态 LoRA rank，避免读取上一批旧 slot；下一活跃 batch 正常刷新 metadata。
- **明确未实现的范围**：prefill CP/PCP 的 token 重排后 adapter metadata 重建、DP-gather 后 MLP/lm_head LoRA、其他模型／量化／MoE 的制品契约没有在本轮补齐。这些是生产实现缺口，不能仅标为“GPU 未测试”。受管模式明确拒绝相应组合；DCP 分组也不能跨业务 DP 组。上游存在参数不等于它与 LoRA 的全部组合正确。
- **验证边界**：新增 PP 队列、DP 组身份／迟到 ACK、逐组 warmup、idle graph metadata、H2D copy 事件与内存移交用例；真实多进程 GPU 数值／性能未运行，不外推所有组合已支持。
- **上游边界**：固定 v0.5.17 原生 `check_lora_server_args` 仅允许 NGRAM/None；EAGLE 等组合原本不支持。CP/SP/DCP/PCP 按具体模型、后端和原生实现核对，不因框架存在参数就声称 LoRA 支持。

### R006：扩缩容不能破坏绑定与回收

- **用户要求**：扩容／缩容需要对应处理。
- **确定语义**：扩容成员先追赶全部在用版本，再接纳 Session；缩容先挡新绑定，再等待旧 Session（含工具等待）与后端排空。发布操作的目标集不能随健康发现或缩容缩小。
- **当前接线**：已有 scale_out/scale_in 管理路径调用同一个 manager。加入者在全部在用版本 READY 后原子加入 serving 集；原发布目标集保持不变，回收目标则包含后续持有该版本的成员。退出者先停止新绑定，旧会话保持原路由；全部原生实例 ABSENT、子进程退出均有证据后才停止 Ray owner 并释放空 group 的 PG。历史记录不再持有运行句柄；必要退出证明只保留到 actor 停止成功，未知结果可重试。force/短 timeout 不能跳过这些条件。
- **未知结果**：新引擎 init 或清理未知时保留 DRAINING group、Ray owner 和 PG；回包迟到继续处理原实例。自动换机和外部无进程所有权 URL 接管仍未实现，不以缩小健康列表代替清理。
- **并发额度**：复用 placement 0 的 Shard limiter，增加绝对容量 + topology epoch 的更新。旧 HELD permit 保留到真正排空；缩容后可暂时 in_use 大于新容量，此时不再发新许可。迟到扩容不能覆盖新缩容；后台只在 epoch 改变时同步，停进程前必须确认调整。用户配置的活跃 Agent Group 上限继续独立生效，扩容性能实验须同时给足上游需求。
- **验证**：CPU manager 测试覆盖 join 部分失败、旧 Session 缩容等待、取消 waiter、offload 与 drain 并发；实际 RM/Shards 接线测试已增加但因依赖缺失跳过。未运行真实扩缩容。

### R007：清理冗余与长期运行开销

- **用户要求**：无死代码、过度设计或不必要的性能损失，控制总代码量。
- **当前改动**：混合 batch 实验证据移出生产 request/IPC 记录，改由隔离测试 hook 采集；首次绑定按当前引擎引用数选择路由，后台回收只遍历待关闭 Session 与占槽操作，并避免无调用者的全量状态序列化。历史身份仍保留以抵御迟到消息。
- **验证**：新增历史记录不可进入绑定／后台扫描路径的回归，验证重复关闭、引擎均衡和容量；状态机 CPU 回归通过。新版本覆盖 A slot 的错误权重负对照已接线，需 GPU 执行验证检错能力。

本次代码仍有 R005 列出的生产兼容性缺口，不能认定用户提出的“所有 LoRA 相关能力均可用”已完成。已接线项目也尚未通过真实 colocate/offload、多进程 GPU 或扩缩容运行验收。没有提交、push 或运行远程/GPU 作业。

## 本地验证记录（2026-09-20）

- TP 接线后的核心回归：213 passed、51 skipped；skip 对应缺少 Ray／Megatron／完整 SGLang／GPU 环境，未用 mock 冒充真实集成。
- 后续单卡共享布局与配置回归：74 passed、39 skipped；其中真实 EngineGroup 接线测试因 Ray／SGLang 依赖缺失而跳过。
- 多 rank 协议用例覆盖 2/4 rank：最后一个 rank 的加载或 slot clear 未完成时不报告完成，重复／旧 boot ACK 不能补足额度；请求引用和 Session 关闭同样等待所有 rank。
- GPU 测试脚本按 rank 检查 pin、resident、物理 slot 复用、有效卸载次数和 graph replay；混合 batch 证据按 rank 分文件，避免多个进程覆盖同一个证据文件。
- 本轮训练边界与内存移交后的核心回归：285 passed、54 skipped；与上述旧计数重叠，不累计相加。额外 memory 协议针对性回归 10 项通过，包含两 rank 线程屏障模拟，不能代替真实设备实验。
- 修复 offload 核查发现的 KV 元数据失效、LoRA pool 缺 CPU backup、迟到旧移交及 rank-local barrier 问题。固定 v0.5.17 干净源码重新应用 patch 并逐字比较 51 个文件，Python 语法解析通过。
- 全仓 pre-commit（含 gitleaks）通过；未提交或 push。仍在继续实现扩缩容与剩余并行兼容性。

### 本轮追加验证

- 完整本地选择集：**313 passed、64 skipped**，包含新 PP/DP、训练导出边界、内存移交、成员变更与 permit 测试；skip 明确对应完整 Ray/Agentic/Megatron/SGLang 或 GPU 缺失。与上面的历史数字重叠，不求和。
- 从固定 v0.5.17 的全新源码应用当前补丁，53 个改动文件逐字匹配工作源码，Python AST 解析通过。
- 主验收入口现在按原生返回的 DP route 覆盖各引擎的所有 attention-DP 组，检查相同组 A/B 混合 batch、cache 正负对照及对应 worker 的 drain；不是假设每个请求都由整个 engine 执行。
- 历史方案曾提供 `memory_handoff: true` 辅助。2026-09-23 审计确认六项自动入口没有调用它，R019 已删除死分支；不能将此项视作已接通的 GPU 验收。内存移交 CPU 协议回归仍保留。

最终全仓及新增文件 pre-commit（含 gitleaks）通过，`git diff --check` 通过。没有提交／push。

## 本轮：提交重组、RFC 重写与测试精简（2026-09-20）

### R008：两个提交一起按功能拆分

- **用户要求**：将 `621a59a` 与 `5fb059b` 一起重新拆分，便于逐项 review；RFC 要完整、详细。
- **处理**：以 `4cb07d8` 为基线组织七个提交：不可变制品、原生执行协议、发布协调、训练导出、框架接入、验收工具、RFC/review。每个实现提交携带相应测试；不把测试全部压到一个独立大提交。
- **历史保护**：拆分前建立本地 `backup/immutable-lora-before-resplit-5fb059b`，保留已推送的原提交。此次只重建本地历史，远端仍保留旧分支历史；尚未执行 force push。
- **RFC**：重写为当前实现说明，按训练保存→快照→prepare/commit→Session→原生排空→回收→offload→扩缩容→配置与验收展开，替代历次提案和追加修订互相矛盾的正文。配置默认值、API、状态名对照当前代码。
- **新核对出的限制**：native attention-DP 有局部协议接线，但 `relax/utils/arguments.py` 的总校验仍要求 adapter mode 的 `sglang_dp_size=1`，因此 Relax 端到端 DP 尚未接通。已修正文档，不能仅标为“GPU 未测试”；本轮没有更改生产参数行为。

### R009：删除重复测试、减少文件与样板代码

- **用户要求**：删除不必要的测试，解决文件过多、内容冗余。
- **已删除的重复覆盖**：108 种独立配置开关的笛卡尔积改为 9 组代表性配置；删除单 tensor writer smoke，由已有完整 writer/provenance/snapshot 用例承接。
- **已合并文件**：CLI 与单独 placement 校验合入 configuration；fixture/GPU 入口/验证器合入 test_acceptance；permit 合入 Agentic transport；communicator 合入 native execution。
- **已精简写法**：91 处逐测试嵌套 coroutine + asyncio.run 改用仓库已有 pytest-asyncio。没有增加项目依赖，本机缺少的既有测试依赖仅安装到 `/tmp`。
- **保留理由**：manager 的 fake READY、原生协议中的假事件、真实 tokenizer/pool 接线、GPU 数值分别证明不同层级，不能相互替代。GPU 验收辅助保留场景／性能／HTTP 分工，不重新合成千行函数。
- **规模口径**：统计本分支涉及的测试 Python 文件（包括其中原有内容）由 32 个／9381 行变为 26 个／9047 行，减少 6 个文件、334 行；`test_*.py` 从 23 个变为 17 个。测试函数 273→272，主要去掉的是重复配置实例和样板，不以删故障场景换行数。
- **整理后本地选择集**：214 passed、65 skipped；不同于此前 313/64 的主要原因是配置组合精简及 optional-runtime 测试收集粒度变化，不能用总数下降推导安全覆盖被删除。GPU 与完整分布式运行仍未执行。

## 本轮：旧评审仍适用问题的修复（2026-09-21）

### R010：生成不能耗尽取消／排空查询的连接额度

- **用户要求**：实现本轮核对后仍成立的连接池隔离与快照 staging 预算修复。
- **改动**：`SGLangBackendAdapter` 的 generate 保留数据客户端；attempt cancel/status 改用独立控制客户端，最多 16 个连接、8 个 keep-alive，5 秒超时。Session bind/close 仍走 RM Ray 通道。shutdown 关闭两者，控制连接失败不释放原生引用或 permit。
- **测试**：在现有 transport 测试文件补充真实本地 HTTP 连接池压力场景：把数据池限制为一个连接，保持生成未返回，取消仍须完成；生成结束后查询可信 drain，最后检查两个客户端关闭。沿用实际 runtime 类，未用 AST 抽取或伪造 Ray 模块绕过依赖。
- **验证边界**：本机缺少 Ray／完整 Agentic 运行环境，该压力场景跳过，尚未取得实际运行通过证据。

### R011：复制快照前预留空间，失败后释放

- **改动**：在复制权重前，以文件长度、规范化配置和 manifest 大小计算完整预留；沿用 `artifact_max_bytes` 和进程间文件锁，将正式制品、各快照预留及未受管 staging 当前可见字节合并记账。复制在锁外执行，写入不超过预留文件长度；提交在锁内将预留转换为正式占用，失败只清理自己的 staging。
- **幂等与故障**：已有版本先核对摘要，同内容重试不再复制、不要求额外额度；不同内容仍拒绝。异常退出留下的预留继续占额，不按 TTL 释放；rename 后同步失败继续保留可见制品和 UNKNOWN 语义。
- **范围**：修复的是 snapshot 复制接纳。训练导出的源文件写入尚无预留协议，只统计 store 内当前可见字节；外部源目录、文件系统元数据不受此逻辑额度约束。没有新增数据库、后台回收服务或自动删除正式制品。
- **测试**：在原有 snapshot 测试文件覆盖复制前容量拒绝、并发预留竞争、失败归还、满额重试、遗留预留保留和复制期间源文件增长。不增加测试文件。


## 本轮：按实例精简 SGLang 排空（2026-09-21）

### R012：复用原生生命周期，删除每请求 × 每 rank 账本

> 后续 R013 发现本次精简留下 PP 悬空调用，且原生职责与历史记录仍未收敛。本条的测试通过记录不能作为 PP 可运行或架构精简已经完成的结论。

- **用户要求**：TP 使用原生多 rank 控制；PP 只需每 stage 的实例排空。删除 shadow scheduler，不实现 worker 重启后的局部请求恢复。
- **实现**：删除 `SchedulerExecution`、scheduler attempt 状态／墓碑／pending 表和 tokenizer 的 `drained_ranks`。原生请求复用 `rid/lora_id`，worker 只补 boot、kind、Session ID；逻辑完成由原生输出 leader 通知一次。leader 的活动索引仅引用真实 Req，完成后删除。
- **实例回收**：扫描原生队列／PP microbatch；每实例保留 launching 与最后 GPU event，沿用 load/fence/cleanup fan-out。所有 worker 的清理完成仍是 ABSENT 的必要条件；旧操作不能影响同名新实例。KV 沿用原生内部 lora_id 隔离，不新增 cache key 系统。
- **明确调整的语义**：REQUEST_FINISHED 是逻辑请求完成，用于 tokenizer 引用与 fleet permit；不是全 rank GPU 完成。实例事件独立阻止旧权重卸载和 slot 复用。Session close 仍等各 worker 的固定 stream 完成点及 KV close；不以 HTTP 退出证明完成。协议升为 v2，旧版客户端／引擎不混用。
- **删除的旁路**：删除缺 slot 后逐 rank 手工过滤 batch／释放 KV 的恢复逻辑及相关 invariant 扩展。已接纳实例丢失按原生 engine 故障处理，不让一个 rank 私自改写 collective 的请求集合。
- **测试调整**：移除逐请求 rank ACK 汇合及 shadow quarantine 测试；用真实 Req 容器引用、单 leader、实例事件、原生低频控制测试替代。GPU 脚本改为 A 最后 GPU 使用与 slot clear 分别延迟，要求 B 前进、C 容量拒绝；保留混合 batch、cache 双向与负对照、graph 和多 worker 回收。
- **范围**：没有新增依赖或扩展并行支持矩阵；不关闭 CUDA graph。GPU／完整训练验证仍由用户 review 后执行，不将 CPU 事件替身记为 GPU 通过。

- **本地结果**：相关选择集 215 passed、51 skipped；完整 pre-commit（含 gitleaks）与 `git diff --check` 通过。固定 v0.5.17 新副本应用补丁后逐文件匹配工作源码、Python AST 解析通过。原生补丁只改动 7 个已有涉及文件，其中 invariant_checker 的旧扩展整体移除，其余补丁段落逐字保留。
- **规模**：整个 SGLang patch 7229→6914 行，`version_control.py` 2088→1958 行；减少的是逐请求 worker 状态和手工 batch 修复。保留加载、pin、制品校验、Session close、graph、offload 与低频控制关联，这些仍有各自职责；不把整份 patch 都算成 TP/PP drain 代码。未 commit／push 本轮修改。

## R013：SGLang 补丁的完整职责审查（2026-09-21）

**用户要求**：认为近 5000 行 SGLang 新增代码存在过度设计，要求更仔细审查。**审查结论：Request Changes。** 本轮只审查、记录，不修改运行代码；下面的修复与收敛项均未实施。

审查范围为 No.7 之前的 `4cb07d8` 到当前工作区，包含 `1a0653b` 之后尚未提交的 R012 精简。两份 Relax patch 分别应用到相同的 SGLang v0.5.17 原始源码，再比较应用后的源码。没有把原本的多模态、采样、PP 定制补丁算成本题新增，也没有只审查最近减少的几百行。

### 1. 实际规模与分布

当前整个 patch 是 6914 行；**No.7 实际新增源码 3587 行、删除 47 行，涉及 14 个原生文件**。以下使用逐文件行序列差分统计，包含新增注释、空行，不包含测试，也不包含 patch 上下文；相近内容可能因 diff 算法产生少量计数差异。

| SGLang 原生位置（均位于 `python/sglang/srt/`） | 新增／删除 | 新增职责 |
| --- | ---: | --- |
| `lora/version_control.py` | 1958／0 | 身份与消息、启动校验、tokenizer 请求所有权、版本加载／准备／退休、Session close、health、内存移交、scheduler 排空 |
| `managers/tokenizer_control_mixin.py` | 618／3 | 新控制通道、boot 握手、Relax 制品接纳、管理 API、请求绑定、预热、旧入口限制 |
| `managers/scheduler.py` | 397／3 | 消息处理、受管请求接入、取消、调度循环 hook、内存移交、KV close、入口限制 |
| `lora/lora_manager.py` | 142／1 | 精确实例清理、驻留准备、batch 校验、部分加载失败处理 |
| `entrypoints/http_server.py` | 127／0 | 管理路由、受管初始化、独立健康探测 |
| `managers/tokenizer_manager.py` | 102／2 | 请求 acquire／发送／终态接线、响应来源、取消与 consumer 分离 |
| `managers/communicator.py` | 71／16 | owned RPC、关联回执、sender 去重、取消隔离 |
| `lora/mem_pool.py` | 67／0 | CLEARING 期间保留槽位，清理 event 完成后才释放 |
| `lora/lora_registry.py` | 24／0 | 按原始 LoRARef 精确 acquire／unregister |
| `model_executor/model_runner.py` | 23／14 | 将 LoRA pool 纳入 memory-saver 的权重备份区域 |
| `managers/scheduler_pp_mixin.py` | 20／2 | PP batch event 与回收推进；含未清干净的调用 |
| `managers/io_struct.py` | 16／3 | binding、私有请求字段、memory 控制字段与错误结果 |
| `lora/backend/triton_backend.py` | 13／0 | DP 空闲 graph 路径清除旧 LoRA rank 元数据 |
| `managers/scheduler_components/weight_updater.py` | 9／3 | 受管内存移交中的 KV reset 与 collective 顺序 |

前三个文件占新增行数约 83%。`LoRAVersionControl` 类本身有 811 行，同时管理请求、版本、Session、health、offload；1958 行的模块并非全部在实现 TP/PP drain。上次主要删除逐请求 rank 账本，没有收敛这些职责，所以只减少几百行并不意外，也不能据此宣称完成精简。

### 2. 确定的问题

#### P1-01：删除方法后，普通 PP 循环仍调用它

- **位置**：[patch](../../docker/patch/sglang/v0.5.17.patch#L6858)，`SchedulerPPMixin.event_loop_pp`，应用后原生文件第 119 行。
- `_reject_invalid_lora_requests()` 已从 `Scheduler` 删除，但 PP 循环仍无条件调用。当前完整原生 Python 源码中只有这一个调用，没有定义或注入；R012 之前存在定义，Relax 基线没有此调用。
- **结果**：常规 PP 循环一旦执行到这里会触发 `AttributeError`；调用没有 publication mode 判断，普通 PP 也受影响。这是本次精简引入的确定集成回归，不能归类为“只差 GPU 性能验证”。
- **修改**：删除残留调用，沿用已经选定的实例校验／engine 故障边界，不把删掉的 batch 修复系统重新补回来。相关 PP 实际运行覆盖必须补上；语法解析与 helper 单元测试无法发现这种错误。

#### P1-02：成功的健康探测会耗尽业务接纳额度

- **位置**：[health_probe](../../docker/patch/sglang/v0.5.17.patch#L5538)、[close_session](../../docker/patch/sglang/v0.5.17.patch#L5860)、`_session/acquire/_release`；以及 [monitor_boots](../../relax/distributed/ray/rollout.py#L1085)。
- 每次新的内部 probe 使用新 Session ID、rid，进入完整业务请求生命周期。即使已收到 `SESSION_DRAINED`，`records`、`rids`、`sessions` 都不删除；scheduler 的已关闭 Session 也保留完成 event 等完整对象。
- **本轮 CPU 复现**：直接导入应用后的完整原生模块，控制 probe 成功与 Session drain 回执，额度设为 20。第 1／5／10 次成功后占用分别为 2／10／20，活动请求始终为 0；第 11 次报 `LIFECYCLE_CAPACITY_EXCEEDED`，`accepting=false`。没有模拟 GPU 数值或声称启动了真实引擎。
- 这不是无限制内存增长，而是**把历史预算变成了健康检查可耗尽的服务寿命预算**。默认额度 100000，仅每约 10 秒一个成功 probe 就约 5.8 天耗尽；这是忽略探测耗时、初始化与其他流量的估算，业务流量会进一步消耗额度。
- [现有测试](../../tests/backends/sglang/test_native_lora_versions.py#L1024)明确断言两个成功 probe 后第三个耗尽额度，说明测试在固化当前设计，不能证明该行为合理。
- **修改**：内部 probe 用有界的内部身份／序号和原生请求完成路径。取消未排空时仍保留 owner；可信完成后释放完整记录，不向永久业务 Session 历史收费。旧 probe 消息必须因实例／序号不匹配被拒，不能用不安全 TTL 代替这个条件。

#### P2-01：结束一个 Session，扫描整个 engine 的历史请求

- **位置**：[close_session](../../docker/patch/sglang/v0.5.17.patch#L5860)、`retire_version`、[change_memory](../../docker/patch/sglang/v0.5.17.patch#L5405)。
- 关闭通过 `self.records.values()` 筛选当前 Session；版本退休也扫全表；offload 枚举全表等待 completion。已结束记录从不压缩，仍带多个 task、Event、状态字段，失败 task 还可能保留异常上下文。
- 设累计请求数为 N，关闭一次的筛选成本为 O(N)，而非当前 Session 的请求数。若每个 Session 一个请求，持续创建／关闭的累计筛选次数为 O(N²)。这是代码可确定的工作量增长；本轮没有测量它对 GPU 吞吐的百分比影响。
- **修改**：当前执行按 Session／instance 关联，完成后移除活动记录；Session close fence 完成后压缩其 attempt 历史，保留需要拒绝迟到请求的最小证据。考虑使用不复用的内部 rid，避免为了输出路由安全永久保留完整执行对象。不能直接删除所有墓碑，让迟到请求重新被接纳。

### 3. 主要过度设计位置

#### P1-03：原生扩展仍是多种管理职责的集中实现

- **位置**：[TokenizerVersion](../../docker/patch/sglang/v0.5.17.patch#L4957)、[LoRAVersionControl](../../docker/patch/sglang/v0.5.17.patch#L5325)、[configure_lora_publication](../../docker/patch/sglang/v0.5.17.patch#L2774)。
- 原生 tokenizer 直接导入 Relax 的 `ModelContract/AdapterSnapshot`，了解 store 布局、来源 step、封存确认；同一 owner 又管理 DP Session 路由、请求交付状态、health 和 memory sequence。代码中的隔离不变量跨越太多职责，修改某一项很容易遗漏其他入口，PP 残留调用就是实际例子。
- `_lora_publication_operations` 与 `control.versions` 以相同 operation ID 保存**同一对象**；这不是两套独立事实，但确实是无必要的重复索引与接口穿透。mixin 直接调用 control 的 `_version/_command`，还替它持有 `artifact_task/observation_task`。
- 单版本对象有 artifact／prepare／load／retire／observation 五类 task；再叠加 communicator owned task。它们并非全都无用，尤其 fence 不能被未完成 load 阻塞；但 artifact→prepare→load 的直线阶段没有必要分别成为长期保留的业务 task 层。
- **修改**：Relax 保有发布意图、制品策略、默认指针和 Session binding。引擎侧保留实际加载实例及一个准备操作 owner，顺序完成核验／load／warmup；退休意图能够独立安装 fence，随后推进同一实例清理。存储格式适配集中在 engine 边界，scheduler 只接收精确实例与资源操作。引擎读取制品的完整性核验仍须保留，不能简单把校验搬到远端后假设引擎读到的一定是同一份文件。
- 只把 1958 行拆成几个文件、或者把同样状态机搬回 Relax，不算完成该项。必须减少状态拥有者、重复索引和交接关系。

#### P2-02：通过旁表和队列巡检重新推断原生请求完成

- **位置**：[native_requests](../../docker/patch/sglang/v0.5.17.patch#L6152)、[poll_request_completion](../../docker/patch/sglang/v0.5.17.patch#L6298)、[自定义取消](../../docker/patch/sglang/v0.5.17.patch#L2211)。
- 现在已没有“每请求 × 每 rank”状态表；leader 的 `rid → Req` 只是原生对象索引，不能继续把旧问题描述为仍原样存在。但正常循环仍扫描多种 batch／队列、建立临时容器，来判断 Req 是否离开；取消也走一个单独实现。以后原生新增容器时，两处都要同步维护。
- 当前受管入口通过拒绝 grammar、PD、部分 spec 等路径限制巡检范围。未发现当前允许配置中的 grammar 漏算：scheduler 入口显式拒绝了该请求，不能把它误报为已证实提前回收。不过这种做法也说明它没有自然继承全部原生请求能力。
- **修改**：在原生实际接纳／最终退出路径增加最小 instance 所有权 hook，覆盖排队取消、重排、chunked、PP buffered work。不能只挂 `req.finished()` 或 KV release 就自称完成；需明确“不可再调度”与“GPU 最后使用”两个边界。正常 token 路径更新活动量与最后使用依赖；仅有待退休实例时推进 retirement。实例 GPU event 和 slot clear 仍保留。
- 若暂时需要容器扫描，应共用一个清晰的原生所有权遍历入口，并只遍历活动状态；不要让多个 LoRA poll 函数各自重建相同视图。是否能彻底移除逐循环扫描，需要 hook 覆盖证明，不是现在直接删掉扫描。

#### P2-03：请求热路径存在无充分用途的工作

- [请求 fingerprint](../../docker/patch/sglang/v0.5.17.patch#L3054)对 dataclass 字段（含完整 token 输入）重新 JSON 序列化并 SHA-256。当前重复 attempt 无论内容相同或不同都拒绝，不恢复输出；fingerprint 主要用于区分两个拒绝错误。**制品内容摘要必须保留；请求全文摘要不是同一回事。** 如果统一重复请求错误契约，可删除请求全文散列与对应字段。
- [驻留校验](../../docker/patch/sglang/v0.5.17.patch#L4776)在 batch 路径检查静态 dtype／rank／scaling／target modules 契约；`begin_execution_batch`、fetch 和 prepare 又有相关检查。应在 prepare 确定不可变配置，热路径保留必要的实例、slot、CLEARING 校验，避免重复规范化静态配置。不能把 slot 校验也删掉。
- `finish_execution_batch`在非 PP 路径新增 batch event；PP 已复用原生 event。需审核可否复用其他路径已存在、确实覆盖 LoRA 读取的完成依赖，不能仅凭 event 名字相似合并。本轮只指出需要测量／审核，没有证据宣称 event 已造成某个比例的性能下降。

### 4. 哪些应保留，不能为了行数删除

| 能力 | 审查结论 |
| --- | --- |
| 唯一内部 LoRA ID 与精确实例比较 | 保留。旧 op 的迟到 retire 不能按公开名卸掉新实例；KV 已有内部 lora_id 隔离，本题无需另造 cache key 系统 |
| tokenizer 与 scheduler 的接纳 fence | 保留必要消息顺序。发送中的请求、HTTP 退出与后端执行不是同一时刻 |
| pin 与 CPU/GPU LRU 保护 | 原生能力复用；容量拒绝不能靠卸载仍有引用版本来解决 |
| 最后 GPU 使用与 CLEARING event | 保留，尤其 graph／PP／H2D 路径；这部分是题目要求的物理安全，不是纯防御性样板 |
| 控制回执 correlation 与取消后的 task 所有权 | 保留。原 communicator 的单 waiter 不能自动解决迟到 ACK 误配 |
| 低频多 worker 退休完成证据 | 保留。它不同于每请求每 worker 账本；worker 故障应让 engine UNKNOWN，不实现局部透明接管 |
| Session close fence 与 KV 保护释放 | 保留必要语义，压缩终态记录；不能用 HTTP close 200 代替 scheduler 完成 |
| offload／colocate、graph、并行接线 | 用户明确要求，不能为减少补丁直接删功能或关闭 graph；应复用 weight updater／memory saver，仅增加实例排空与恢复核验 |

新增独立 ZMQ 回执通道和四个 owned communicator 是可审查的收敛目标，但**本轮不能断言直接删除就安全**：已有 tokenizer 回执路径主要按 DP/leader 汇集，并不自动给出所有 TP/PP worker 的退休完成证据。需要先在原生控制路径完成实例级聚合，再减少通道；独立 fence 必须仍能越过未收敛 load。不能用每轮全局 barrier 换掉这些代码，导致其他版本停止生成。

### 5. 清理顺序与可判定的完成条件

| 优先级 | 清理／替换项 | 判定方式 |
| --- | --- | --- |
| 立即修复 | PP 悬空调用 | 全树无悬空引用；普通／受管 PP 的真实调度入口运行，不只测试 helper |
| 可直接收敛的候选 | 同对象重复 operation 字典；仅测试调用的 `metadata_usage()`；从无调用者传 false 的 `require_slot` 参数 | 全树调用检索、状态查询不变；不删制品 digest，不删必要 slot 断言 |
| 先修终态记录 | probe 不消耗永久业务历史；Session drain 后压缩 attempt 对象 | 重复成功 probe 超过原额度仍健康；已结束会话数增长时，活动对象/task/event 不线性积累；迟到请求仍被拒 |
| 再收敛准备操作 | 一个准备 owner 代替 artifact→prepare→load 多层长期 task | load 超时／迟到 ACK／取消先到仍安全，同 ID 重复不重复加载，cleanup 未知不释放槽 |
| 再复用原生请求生命周期 | 消除自建完成判定和取消的重复维护 | waiting／running／chunked／PP／断连／取消竞态；A 退休不要求 B 停止；逻辑完成与 GPU drain 分别证明 |
| 最后缩减原生控制接口 | 实例级控制聚合、共用已有通道与 memory updater | TP/PP 全目标 ABSENT 后才释放容量；offload 和 graph 恢复数值正确；旧 ACK 不能完成新命令 |

请求 fingerprint 删除需同步其错误码及测试预期；它是有调用但用途过小的逻辑，不冒称死代码。CP／DP 的未接通分支也不能简单算死代码删掉来“满足”用户的并行支持要求：Relax 总参数校验仍限制 LoRA adapter mode 的 DP=1，native 还拒绝 CP 等组合，兼容性缺口需要单独如实处理。

### 6. 证据与审查边界

- 比较了上表 14 个应用后的源码差分，重点追踪原生 load/unload、registry、请求消费与取消、调度循环、PP、memory pool、graph reset，以及 Relax 对应 engine／RM 调用方。
- 新增证据是 PP 调用与定义核对、完整原生模块的健康历史 CPU 复现、历史扫描与任务／索引调用关系核对。没有重新跑全套旧测试，也没有把之前的 `215 passed / 51 skipped` 当成本轮结论。
- PP 残留调用与健康历史耗尽应阻止将当前版本称为完成；职责重构和热路径精简也尚未完成。
- 未运行真实 SGLang server、TP/PP GPU、graph、训练、数值或性能实验。因此不承诺一个未经验证的“最终只需几百行”，也不估算删掉多少行就一定无性能损失。

本次审查要求的改变是：**减少原生层承担的业务管理职责、长期历史对象和重复生命周期逻辑，同时保留精确实例与可信 drain。** 单纯拆文件、删测试、删兼容性或重复缩短代码，都不能作为 R013 完成的依据。

## R014：先修复确定错误与持续运行开销（2026-09-21）

- **用户问题**：能否处理审查问题，尤其是性能问题。确定的实现错误和冗余开销可以修复；真实吞吐／延迟改善必须由用户 review 后的 GPU 实验验证。本条不将整个 R013 标为完成。
- **PP**：移除已删除方法的残留调用。在原有 native hooks 测试中增加普通／受管 PP 实际循环入口到 batch selection 的回归，完整原生环境缺失时显式跳过；它不是 TP/PP GPU 运行验收。
- **health**：保留自身请求排空、GPU event、取消隔离；完成后释放完整 probe 记录，用每 engine 单调序号拒绝迟到消息。公共入口拒绝内部身份前缀。probe 没有打开 KV Session，不再调用无必要的原生 KV close。CPU 回归将额度设为 4，连续执行 20 次成功探测，每次结束后两侧 probe Session／attempt 表为空，旧 submit 拒绝、旧 close 幂等。
- **Session 历史**：以 Session 自身的 attempt 索引完成关闭，删除其完整执行记录；清空终态 task／回执引用与 scheduler completion event。关闭 fence 和最小 rid 索引保留并继续占元数据预算，不能让晚回包落到新请求。回归禁止 close 枚举全局 records，同时验证其他 Session 的引用未被释放以及旧 rid 不能复用。
- **请求开销**：删除请求正文额外序列化与 SHA-256、相关字段和原测试假设；同 attempt 重发统一拒绝。制品 SHA-256 不变。静态 LoRA 配置校验留在 load/READY，batch 只检查动态实例／pin／slot／CLEARING；保留 graph、实例最后使用与 slot 清理事件。
- **小型冗余**：删除指向同对象的第二份 operation 字典、仅测试调用的 metadata 汇总方法、无调用者使用的可选 slot 校验参数。没有新建服务、后台扫描器或测试文件。
- **仍需收敛**：准备操作的多层 owned task、原生业务管理耦合、常规循环的 Req 容器巡检与自定义取消。本轮没有未经证明地替换它们，不宣称已经实现最终 minimal native contract。
- **性能证据范围**：CPU 回归可以证明记录不随成功探测次数积累、close 不遍历全局历史，以及请求正文／静态配置不重复处理；不能证明 GPU 吞吐提升比例。已有验收入口保留普通 LoRA 基线、受管稳态、发布流量、graph 与逐 engine 进度／延迟采集；真实实验仍未运行。
- **本轮验证**：固定 v0.5.17 原生模块协议测试及可选 hooks 为 `113 passed / 30 skipped`；`tests/engine/lora/` 与验收注入 hooks 为 `103 passed / 9 skipped`。跳过项需要完整 SGLang／Ray／GPU 或显式验收环境，不能据此声明 PP、graph 或性能验收通过。发布管理器测试曾在沙箱内等待线程回调，解除沙箱限制后同一源码的 21 项测试通过，未为此修改管理器代码。补丁在原始固定版本上重新应用成功，应用结果与编辑源码一致；`pre-commit run --all-files` 全部通过，包括 gitleaks。

## R015：原生层精简设计（2026-09-21）

- **用户要求**：明确否定“剩下每一行都必要”的判断，要求进行精简设计，而非继续零散删行。
- **设计位置**：[RFC 第 21 节](./immutable-lora-publication-redesign-rfc.md#21-sglang-精简设计与实施边界)。第 1–20 节仍记录当前代码；第 21 节是待实施的替换方案，未获得 GPU／性能验收结果。
- **确定取舍**：制品契约移回目标机器上的现有 SGLangEngine；原生采用不可复用实例 ID；每实例一个业务 work task；ReqState/Req 作为唯一完整请求 owner；原生精确 abort 和终结通知替代专用取消器／每轮 request polling；普通 status 不启动观察任务；四个 owned communicator 收敛为 mutation/fence 两个。
- **不为减行破坏安全**：保留独立 fence、当前单个低频全 worker 回执 socket、pin、slot CLEARING、实例最后 GPU 使用和 Session close fence。暂不增加每 rank active_count；pending 控制时复用原生当前容器检查，正常生成不扫描历史或轮询所有请求的完成状态。若该阶段实测仍有瓶颈，再评估聚合计数。
- **性能与范围**：保留 graph／TP／PP／offload 接线及完整验收要求；同步 loader 的局部停顿、尚缺的并行能力不能靠本次设计自动解决。说明冷／热路径成本、删除前提、逐文件改动、协议迁移和三阶段实施；移动代码不计净删除，测试不按行数裁剪。
- **状态**：完成设计草案，本轮仅修改 RFC 与本记录；没有把上一轮测试结果当成新设计已经通过的证据，也未修改或提交运行代码。


## R016：执行 R015 的原生层精简（2026-09-21）

- **用户授权**：“请你完成”。本轮实施第 21 节三个代码阶段，沿用当前分支并保留此前未提交的 R012/R014 改动；未提交、未 push。
- **实际删除**：原生 Relax 制品解析、`TokenizerExecution` 完整对象表、scheduler `lora_requests` 和 `poll_request_completion`、LoRA 专用取消器、五个分层 task 字段、重复 native UID 索引、两个多余 communicator。没有以另一个类重建同样的请求账本。
- **替换接线**：目标 `SGLangEngine` 独立 artifact 并发组校验；RM 预分配实例；native instance/boot 两字段；唯一 work task；ReqState 的原有 dispatch/abort 字段与一次性 finalizer；终态输出／排队 abort hook；原生 exact abort；pending 控制共享容器视图。health/memory 回归既有入口，状态与任务责任仍受控。
- **协议和样本**：capability v3；请求 `lora_path` 使用实际 native ID。原生返回 instance/digest/boot，经 Relax 校验后映射业务版本和来源 step。CPU 测试、可选原生 hooks、GPU 测试注入与诊断请求均同步迁移。
- **保留保障**：Session ref、pin、未确定控制的所有权、实例最后 GPU 使用、slot CLEARING、全 worker 的低频 ACK 和 KV 身份隔离保留。graph／TP／PP／offload 既有接线保留；不承诺未实现或未验收的组合已支持。
- **去重与测试**：旧协议测试改为真实原生终态／ReqState 契约；一个 `conftest.py` 集中 CPU 请求状态和依赖 double。导入完整 native control mixin，没有抽取方法 AST 后伪装成集成测试。新增状态只读、迟到校验、单任务和共享 pending view 回归；修复目标校验错误受限于 5 秒状态查询预算的问题，并验证较慢校验仍可发布。
- **验证**：精确源 patch 应用／结果一致性通过；原生协议及注入 hooks `123 passed / 30 skipped`，LoRA 目录 `97 passed / 9 skipped`。完整原生 hooks、Ray／PyTorch 与 GPU 缺失导致 skip，不能据此宣称 graph／多卡／性能验收通过。完整 `pre-commit run --all-files` 通过，新增测试辅助文件另行通过 Ruff 检查。
- **规模诚实报告**：同框架基线，HEAD 原生净增 3770 行，当前 3453 行，累计减少 317 行；R014 之后本轮再净减少 102 行。结构精简不等于补丁已经很小，移动代码未计净删除，测试删行未冒充实现精简。
- **状态**：代码替换可 review；完整原生与 GPU／性能验收待运行，R013 的同步 loader 停顿及并行兼容性缺口没有被标为解决。

## R017：双 GPU 验收准备（2026-09-21）

- **用户要求**：准备在两张 6000 系列 GPU 上测试，明确模型／数据需求。
- **制品准备**：新增测试目录的 `acceptance.prepare`，复用现有 fixture writer 和 snapshot；生成 A/B、B 内容的新 ID C、摘要、发布配置及含 32 条固定 prompt 的部署配置草稿。保留 `relax.fixture` 来源，不覆盖已有实验，不启动远程任务。
- **发现并修复遗漏**：性能脚本仍发送旧版 lora_path／业务字段，status 缺 native ID；现在使用 v3 实例协议并核对流式 digest/boot。graph hook 原先读取已移除的 native engine_id，现在按实际 forward batch 的 instance/rank 记录并验证 A/B graph replay。
- **验证**：相关验收及观测 hook 回归 `21 passed / 2 skipped`；跳过真实 PyTorch fixture 生成和 GPU deployment。prepare 的命令帮助、本轮完整 pre-commit 检查通过；未运行 GPU。
- **部署边界**：准确卡型／显存、镜像、模型目录、Ray/Serve 地址和进程显存份额尚待确认；现有完整 runner 依赖独立基线与实际 Actor 导出记录，不宣称只有两个 engine 即可直接运行全套。详见[双卡准备说明](./immutable-lora-dual-gpu-test-preparation.md)。


## R018：全分支清理与重新组织提交（2026-09-23）

- **用户要求**：再次整理所有代码，删除不必要的测试，并重组课题提交；不以减少文件数代替去除冗余。
- **删除依据**：`acceptance/diagnose.py` 属于旧数值路径探索，其正式替代是缓存重放与确定性 profile；删除该文件及四个专属测试函数。重复 Agent 只保留 `tests/engine/lora/acceptance/agent.py`，删除 examples 下相同副本；旧手工部署 JSON 没有调用者，自动部署 profile 和用户发布 YAML 保留。
- **小范围实现去重**：snapshot 复用 artifact 的 canonical JSON 编码，序列化选项及摘要不变；删除独立基线无人使用的禁用缓存参数和分支，不更改当前验收配置。
- **保留依据**：版本状态机、实际 Session transport、原生消息协议、pool 清理与完整原生 hook 测试覆盖不同边界；保留全部六项 GPU 任务及两小时开销／诊断入口。未删掉尚缺真实环境的测试以减少 SKIP，也未降低数值或性能判据。
- **闲置接口**：当时保留无调用者的 `RayTrainGroup.export_lora_adapter` 和仅供测试使用的原生 `LoRAVersionControl.prepare_version`；用户确认后的删除／迁移见 R020。
- **回归**：固定 v0.5.17 patch 应用成功；相关本地套件 `280 passed / 90 skipped`。跳过项需要 GPU、Ray、PyTorch 或完整 SGLang/Megatron 环境，没有据此补报远程验收结果。
- **历史整理**：以 `4cb07d8` 为基线，按制品、原生协议、发布协调、训练导出、框架接入、自动验收、文档七个职责合并 20 个演进提交。本地备份 `backup/immutable-lora-before-cleanup-dfceac7` 保留原历史。重组应与清理后的最终文件树逐项一致；中间提交用于 review，不保证逐个独立部署。


## R019：按行为价值删除冗余测试（2026-09-23）

- **用户纠正**：R018 只清旧脚本不足以解决接近两万行的功能增量，不能以“每个文件各有职责”为由保留全部测试。本轮直接减少低价值局部覆盖，不只参数化或缩写。
- **验收器自测**：`test_acceptance.py` 从 1015 行降至 552 行。删除 traffic/overhead 模拟编排、窗口文件数量、runner preflight/报告 mock、profile 转发、任务布局和 fixture 重复性自检；两套 Session 数值样本合并。保留固定容差、nonfinite、token fork、mask、缓存重放、错误 slot 防假阳性、统计不确定性与子进程/profiler/session 清理。
- **原生协议**：`test_native_lora_versions.py` 从 1691 行降至 1388 行。共享准备与取消 fixture，合并 GPU 排空和 waiter 取消、Session 提交前/在途关闭、错误 warmup proof、overlap copy；删除额外 health 缺失字段矩阵、私有 kind 组合、probe 重复计数和扫描次数断言。真实 retirement 测试保留部分资源清理，协议层不再重复完整模拟。
- **其他删除**：配置测试从 259 行降至 166 行，移除另建 OPD/参数解析替身及简单 CLI 响应自检；删除 pickle/字段存在性、仅 fake event 的 graph 配置自检、注入工具的开关/透传/stream 选择单测；合并两个真实取消入口和 warmup/probe HTTP payload 测试。
- **死辅助**：删除未调用的 `scenarios.memory_handoff` 及不可达的可选 evidence 分支，修正文档。现有六任务不因此少跑任何步骤；不再暗示 JSON 开关能启用该 GPU 场景。
- **规模与取舍**：本轮测试及辅助净减少 1038 行；没有修改生产实现，也没有把 patch 上下文当实现删行。局部 CPU 分支覆盖确实减少；真实六项 GPU 验收、容差、负对照、性能门槛与两小时诊断保留。
- **验证**：完整相关选择集 `222 passed / 91 skipped`；缺 GPU、PyTorch、Ray 或完整 SGLang/Megatron 环境的测试显式跳过。这里没有重跑或补报远程 GPU 结论。
- **提交整理**：本轮测试改动归入原生协议、框架接入、自动验收对应提交，文档归入末尾；最终仍为七个职责提交。本轮前历史保存在 `backup/immutable-lora-before-test-prune-2f36929`。

## R020：接口与验收部署去重（2026-09-23）

- **已获确认的接口清理**：删除无生产调用的 `RayTrainGroup.export_lora_adapter`；训练边界导出、Megatron actor 和 checkpoint 导出保留。原生 `LoRAVersionControl.prepare_version` 移入测试辅助，生产 HTTP prepare 不变；移动不计作净删行。
- **按任务分配验收资源**：capacity 只启动 B 参考引擎，export 不启动参考引擎，performance 保留同卡两组 A 基线，其余任务保留 A/B。只在 publication／sessions 连接 eval runtime，只有 sessions 连接 train runtime。
- **部署结构**：共用一份参考／受管引擎配置；用现有 `ExitStack` 加 `contextmanager` 管理生命周期，删除 `Cluster` 包装层及 prepare 未使用的全局 control 文件。任务自己的故障注入文件、失败报告、资源清理均保留。
- **取舍**：保留 Session 请求索引，避免关闭会话时扫描其他会话请求。manager 的两个重复集合去重是另行待确认方案，本轮未应用；未改变发布锁、默认切换或退休判据。
- **规模**：本轮代码净减少 34 行，其中生产实现减少 43 行；其余包含测试辅助迁移及公共配置展开。没有据此宣称已大幅压缩整个实现，也没有增加新的常驻服务或测试文件。
- **验证**：固定版本补丁应用成功；相关本地选择集 `222 passed / 91 skipped`。跳过原因仍为缺少 GPU、PyTorch、Ray 或完整 SGLang/Megatron 环境；未重跑远端 GPU，不给出性能提升比例。
- **提交整理**：R020 清理归回原生协议、训练导出和验收工具对应提交，保持七个职责提交；整理前历史保存在 `backup/immutable-lora-before-structure-9084fe1`，完整清理文件树保存在 `backup/immutable-lora-structure-tree-20260923`。未应用待确认的 manager 记录去重方案。

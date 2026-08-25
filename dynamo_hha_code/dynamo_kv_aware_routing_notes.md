# Dynamo KV-aware Routing Notes

这份笔记整理了目前对 Dynamo KV-aware routing 的理解，重点回答两个问题：

1. Dynamo router 到底如何根据 KV cache 做路由？
2. Router 怎么知道 SGLang/vLLM/TRT-LLM worker 里有哪些 KV cache？

## 1. 一句话结论

Dynamo 的 KV-aware routing 不是简单选“缓存命中最多”的 worker，而是对每个候选 worker 估算：

```text
把这个请求放过去以后：
  还需要做多少 prefill 计算
+ 会增加多少 active decode / KV block 负载
= 综合 cost
```

然后选择 cost 最低的 worker。

Router 自己不能直接读取 SGLang/vLLM/TRT-LLM 进程里的 KV cache。它依赖 backend integration 把 KV cache lifecycle events 发布出来，Dynamo 的 indexer 根据这些事件维护一个“每个 worker 缓存了哪些 block prefix”的索引。

## 2. Dynamo 在整个推理栈里的位置

Dynamo 不是模型推理引擎本身。vLLM、SGLang、TensorRT-LLM 负责真正执行模型 forward、管理 engine 内部 KV cache。Dynamo 在它们上面做系统层编排：

- OpenAI/KServe frontend
- worker discovery
- request routing
- KV-aware routing
- prefill/decode disaggregation
- active request/load tracking
- Kubernetes deployment/operator
- Planner autoscaling
- KVBM/NIXL 相关 KV transfer/offload 能力

对 KV-aware routing 来说，Dynamo 的关键职责是：

- 维护 worker/cache 状态的外部索引
- 根据请求 token block 与 worker cache 的 prefix overlap 打分
- 同时考虑当前 worker 负载，避免 cache-rich worker 被打爆
- 把请求直接发给选中的 worker

## 3. Router 能知道 KV cache 状态吗？

能，但不是自己读内存，而是靠事件。

以 SGLang 为例，链路是：

```text
SGLang scheduler / RadixAttention / KV cache
  -> SGLang KV event ZMQ publisher
  -> Dynamo SGLang adapter 订阅 ZMQ event socket
  -> Dynamo KvEventPublisher 转成 Dynamo RouterEvent
  -> Dynamo event plane / worker local indexer
  -> Router-side KV indexer
  -> Router 查询 indexer 得到 per-worker prefix overlap
```

所以更准确地说：

- KV cache 的真实状态在 SGLang/vLLM/TRT-LLM 内部。
- 引擎或引擎适配层在 block 存入、移除、清空时发布事件。
- Dynamo 不保存 KV tensor，只保存 block hash / prefix / worker 映射。
- Router 根据这个索引判断“某个请求的前 N 个 block 是否已经在某个 worker 上”。

## 4. SGLang 这边是怎么接入的？

Dynamo 的 SGLang adapter 直接使用 SGLang 的 KV event publisher：

```python
from sglang.srt.disaggregation.kv_events import ZmqEventPublisher
```

代码入口：

- `components/src/dynamo/sglang/publisher.py`
- `DynamoSglangPublisher.init_kv_event_publish`

核心逻辑：

```python
if self.server_args.kv_events_config:
    kv_events = json.loads(self.server_args.kv_events_config)
    base_ep = kv_events.get("endpoint")

    for dp_rank in dp_ranks:
        zmq_ep = ZmqEventPublisher.offset_endpoint_port(base_ep, dp_rank)

        publisher = KvEventPublisher(
            endpoint=self.generate_endpoint,
            worker_id=self.kv_worker_id,
            kv_block_size=self.server_args.page_size,
            zmq_endpoint=zmq_ep,
            enable_local_indexer=self.dynamo_args.enable_local_indexer,
            dp_rank=dp_rank,
        )
```

也就是说，SGLang 通过 `kv_events_config` 开启自己的 ZMQ KV events。Dynamo adapter 订阅这个 socket，然后用 `dynamo.llm.KvEventPublisher` 转发到 Dynamo 的事件/indexer 系统。

如果 `kv_events_config` 没开，Dynamo 不会得到真实的 engine KV cache events。

## 5. KV event 里面大概是什么？

核心事件类型：

```text
Stored(blocks...)
Removed(block_hashes...)
Cleared
```

它们表达的是：

- 某个 worker / dp_rank 新存了哪些 KV blocks
- 某些 KV blocks 被 eviction/free
- 某个 worker 的 cache 被清空

注意：事件里不是 KV tensor 内容，而是 block 的身份信息，例如 token hash / block hash / parent hash / dp_rank / worker_id。

Dynamo indexer 看到事件以后，维护一棵 radix tree，用来回答：

```text
请求 block 序列: [A, B, C, D, E]

worker 1: prefix 命中 2 blocks
worker 2: prefix 命中 5 blocks
worker 3: prefix 命中 0 blocks
```

## 6. Indexer 为什么用 radix tree？

LLM KV cache 复用主要是 prefix reuse，而不是任意子串 reuse。

典型场景：

- system prompt 相同
- few-shot prefix 相同
- 长对话历史前缀相同
- agent/session 中大量重复上下文

Radix tree 很适合做“最长公共前缀”查询。

Dynamo 的 router indexer 会把 worker-reported KV events 组织成 prefix tree。查询时，把请求 token/block hash 序列放进去找最长 prefix match，并返回每个 worker 的 matched block count。

相关代码：

- `lib/kv-router/src/indexer/radix_tree.rs`
- `RadixTree::find_match_details`
- `RadixTree::apply_event`

## 7. 一条请求的 routing 流程

以 frontend-embedded KV router 或 standalone selection service 为例：

```text
client request
  -> frontend tokenize / normalize
  -> 按 block_size 切成 blocks
  -> 计算 block hashes
  -> indexer.find_tiered_matches(block_hashes)
  -> OverlapAnalysis 转成 router overlap signals
  -> scheduler 查询 active load projection
  -> selector 对每个 worker 算 cost
  -> 选 best worker
  -> tracked 模式下先 book active request
  -> 返回/转发给 worker
```

核心代码路径：

- `lib/kv-router/src/services/selection/core/mod.rs`
  - `prepare_selection_inputs`
  - `schedule_selection`
- `lib/kv-router/src/scheduling/queue.rs`
  - `admit_one`
- `lib/kv-router/src/scheduling/selector.rs`
  - `DefaultWorkerSelector::worker_logit`
  - `DefaultWorkerSelector::select_worker`

## 8. 核心 cost 公式

文档版：

```text
raw_prefill_blocks = active_prefill_blocks + incoming_prompt_blocks
adjusted_prefill_blocks = max(raw_prefill_blocks - overlap_credit_blocks, 0)
decode_blocks = active_decode_blocks + incoming_active_blocks
cost = prefill_load_scale * adjusted_prefill_blocks + decode_blocks
```

更贴近代码版：

```text
raw_prefill_tokens =
    active_prefill_tokens
  + uncached_tokens_for_this_request
  + cached_tokens_for_this_request

raw_prefill_blocks = raw_prefill_tokens / block_size

overlap_credit_blocks =
    overlap_score_credit * overlap_credit_decay * device_overlap_blocks
  + host_cache_hit_weight * host_overlap_blocks
  + disk_cache_hit_weight * disk_overlap_blocks
  + shared_cache_multiplier * shared_cache_blocks_beyond_device

adjusted_prefill_blocks = max(raw_prefill_blocks - overlap_credit_blocks, 0)

prefill_cost_blocks = prefill_load_scale * adjusted_prefill_blocks

decode_cost_blocks =
    active_decode_blocks
  + additional_active_blocks_if_request_goes_to_this_worker

cost = prefill_cost_blocks + decode_cost_blocks
```

代码位置：

- `lib/kv-router/src/scheduling/selector.rs`
- `DefaultWorkerSelector::worker_logit`

## 9. 为什么不是“谁命中最多就选谁”

例子：

```text
request = 10 blocks

worker A:
  device overlap = 8 blocks
  active decode load = 20 blocks
  cost = (10 - 8) + 20 = 22

worker B:
  device overlap = 2 blocks
  active decode load = 2 blocks
  cost = (10 - 2) + 2 = 10
```

虽然 A cache 命中更多，但 A 已经很忙。Dynamo 会选 B。

这就是 Dynamo KV-aware routing 的关键：cache locality 和 load balancing 同时考虑。

## 10. Active load 是怎么来的？

Router 不只看 engine-reported cache events，还维护自己的 active request tracker。

这部分状态表示：

- 这个 router 已经把哪些请求分配给了哪些 worker
- 这些请求是否还在 prefill 阶段
- 它们占用了多少 active decode blocks
- 请求结束后什么时候释放

关键代码：

- `lib/kv-router/src/sequences/multi_worker.rs`
- `lib/kv-router/src/sequences/single.rs`
- `lib/kv-router/src/sequences/block_tracker.rs`
- `lib/kv-router/src/sequences/prompt_registry.rs`

调度前：

```rust
request.worker_loads = self
    .slots
    .project_worker_loads(request.token_seq.as_deref(), decay_now);
```

这个 projection 对每个 worker 算：

- `active_prefill_tokens`
- `active_decode_blocks`
- `additional_active_blocks`

`additional_active_blocks` 是“如果这个请求放到该 worker 上，会新增多少 active blocks”。如果新请求和该 worker 当前 active requests 共享 prefix blocks，那么这些 shared blocks 不重复计算。

## 11. Decode load 为什么按 block 算，不按请求数算？

因为 KV cache 压力和 decode 阶段的资源占用更接近 block 数，而不是 request 数。

更重要的是，多个请求可能共享同一段 prefix KV blocks。Dynamo 的 `BlockTracker` 用 block hash 去重：

```text
request 1: [A, B, C]
request 2: [A, B, C, D]

active unique blocks = [A, B, C, D] = 4
不是 3 + 4 = 7
```

这样 router 不会把共享系统 prompt 的场景误判成过重。

## 12. 请求生命周期如何更新 active load？

真实路由时，Dynamo 会先 book request：

```text
select worker
  -> slots.add_request(...)
  -> respond selected worker
```

后续生命周期：

```text
add_request:
  增加 active prefill tokens
  增加 active prompt blocks

mark_prefill_completed:
  prefill 完成，active prefill tokens 清掉

add_output_block:
  可选，生成时增加 output block 负载

free:
  请求结束，释放 active blocks
```

多 router 副本时，会同步这些 active sequence events：

- `AddRequest`
- `MarkPrefillCompleted`
- `Free`

这样不同 router replica 不至于各自为政，把流量同时压向同一批 worker。

## 13. Event transport / recovery

Dynamo 支持几种 KV event 模式：

### 13.1 Local indexer + event plane

常见默认路径。Worker 本地维护 indexer/event buffer，router 订阅 live events。

事件 transport 可以是：

- ZMQ
- NATS Core

如果 router 发现 event id 有 gap，可以向 worker local indexer 请求 replay。如果 gap 太旧，worker local indexer 可以 dump radix tree state 给 router 恢复。

### 13.2 JetStream durable events

NATS JetStream 持久化事件流。更偏 production consistency，但文档里有 deprecated 趋势。

### 13.3 Approximate mode

如果不开真实 KV events：

```text
--no-router-kv-events
```

Router 会根据自己的 routing decisions 预测“哪些 block 可能已经在 worker 上”，并用 TTL 过期。

这比真实 events 不准，但部署简单。适合没有 backend event support 或只是想做近似 load/cache routing 的场景。

## 14. SGLang patch / upstream 关系

这里要分两件事。

### 14.1 普通文本 KV events

Dynamo 当前直接依赖 SGLang 包里的：

```python
sglang.srt.disaggregation.kv_events.ZmqEventPublisher
```

说明普通 KV event publisher 已经是 SGLang 侧能力。Dynamo adapter 只是接入它。

当前 Dynamo `pyproject.toml` 里 pin 了：

```toml
sglang[diffusion]==0.5.14
```

### 14.2 多模态 KV routing 的 mm_hash 协议

多模态场景有额外问题：router 侧计算的 routing token view 必须和 SGLang 内部 RadixAttention cache key 对齐。

文档里提到 SGLang 需要 `GenerateReqInput.mm_hashes` 字段，并引用了 upstream PR：

```text
sgl-project/sglang#25300
```

Dynamo 文档也写到 Dynamo SGLang image 里可能 vendored 了对应 patch。

所以：

- 普通 KV events：SGLang 已经提供，Dynamo 使用。
- 某些 Dynamo 精确对齐需求，尤其多模态 mm_hash：可能依赖 SGLang PR/patch 或 Dynamo runtime image 内置 patch。

## 15. 如果 backend 不告诉 Dynamo，会怎样？

Router 不能凭空知道真实 KV cache。

如果 backend 不发 KV events：

- KV indexer 没有真实 cache 状态
- prefix overlap 可能为空或不准
- 可以用 approximate mode
- 或者把 `overlap_score_credit` 调低/设 0
- 或者直接使用 `round-robin` / `least-loaded`

这时 routing 更接近 load-aware routing，而不是准确 KV-aware routing。

## 16. 关键配置

Frontend/router 侧常见：

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

重要参数：

```text
--router-mode kv
--kv-cache-block-size
--router-kv-overlap-score-credit
--router-prefill-load-scale
--router-temperature
--router-kv-events / --no-router-kv-events
--router-track-prefill-tokens / --no-router-track-prefill-tokens
--router-track-output-blocks
```

SGLang worker 侧关键是要有 KV events config，例如 launcher 里设置 `--kv-events-config`。

Dynamo SGLang adapter 会根据 `server_args.kv_events_config` 判断是否启用 KV event forwarding。

## 17. 代码索引

### Router / Scheduling

- `lib/kv-router/src/scheduling/selector.rs`
  - worker cost 计算
  - deterministic min-cost selection
  - temperature softmax sampling

- `lib/kv-router/src/scheduling/queue.rs`
  - 请求进入调度队列
  - 计算 projected worker loads
  - 调用 selector
  - tracked request booking

- `lib/kv-router/src/scheduling/overlap.rs`
  - tiered matches -> overlap signals
  - device/host/disk/shared-cache credit 计算

### Indexer

- `lib/kv-router/src/indexer/radix_tree.rs`
  - KV event apply
  - prefix match query

- `lib/kv-router/src/indexer/kv_indexer.rs`
  - async indexer wrapper
  - event queue
  - match query
  - approximate routing decision record

- `lib/kv-router/src/indexer/local.rs`
  - worker local indexer / replay / tree dump recovery

### Active Load Tracking

- `lib/kv-router/src/sequences/multi_worker.rs`
  - per-worker active sequence manager
  - add/free/prefill complete
  - replica sync hooks

- `lib/kv-router/src/sequences/single.rs`
  - per-worker active request/block tracking

- `lib/kv-router/src/sequences/block_tracker.rs`
  - active unique block ref-count-ish tracking

- `lib/kv-router/src/sequences/prompt_registry.rs`
  - fast projection of active prefill/decode load

### SGLang Integration

- `components/src/dynamo/sglang/publisher.py`
  - subscribes to SGLang ZMQ KV events
  - creates Dynamo `KvEventPublisher`

- `components/src/dynamo/sglang/args.py`
  - derives `use_kv_events` from `kv_events_config`

- `components/src/dynamo/sglang/init_llm.py`
  - setup SGLang engine and publisher

### Python Bindings

- `lib/bindings/python/src/dynamo/_core.pyi`
  - `KvEventPublisher`
  - `KvIndexer`
  - `KvRouter`
  - selection service APIs

## 18. Mental model

可以把 Dynamo KV-aware routing 想成三张表：

### 18.1 Engine cache table

来自 backend KV events：

```text
worker_id, dp_rank -> cached block prefix set
```

这是 indexer/radix tree 维护的。

### 18.2 Active load table

来自 router 自己的 request lifecycle：

```text
worker_id, dp_rank -> active_prefill_tokens, active_decode_blocks
```

这是 active sequence tracker 维护的。

### 18.3 Request projection

对每个新请求临时计算：

```text
如果放到 worker X：
  overlap = ?
  uncached prefill = ?
  additional active blocks = ?
  total projected cost = ?
```

Selector 最终选 projected cost 最低的 worker。

## 19. 常见误解

### 误解 1：Router 直接知道 SGLang 内部 KV cache

不对。Router 只知道事件投影出来的 block hash/indexer 状态。

### 误解 2：KV-aware routing 就是选命中最多的 worker

不对。它还考虑 active prefill 和 decode load。

### 误解 3：不开 KV events 也能准确 KV-aware

不对。不开真实 events 只能 approximate。

### 误解 4：KV events 传的是 KV tensor

不对。传的是 block lifecycle metadata，不是 tensor 内容。

### 误解 5：所有 cache hit 价值一样

不对。device、host pinned、disk/external、shared cache 有不同权重。

## 20. 后续可继续深挖的问题

- SGLang `kv_events_config` 的具体格式和 launch 脚本怎么设置。
- vLLM 的 KV events 接入和 SGLang 有什么差异。
- `router_temperature` softmax sampling 对热点分散的实际影响。
- `overlap_score_credit_decay` 如何在 worker prefill backlog 高时削弱 cache locality 偏好。
- disaggregated serving 下 prefill worker 和 decode worker 的 routing 是否使用同一套 overlap/load signal。
- 多 router replica 的 `AddRequest/MarkPrefillCompleted/Free` 同步一致性边界。
- approximate mode 的 TTL 和真实 cache eviction 不一致时会产生什么误判。

## 21. Hit rate 的几种口径

讨论 prefix cache hit rate 时，要先确认分母是什么。不同系统和 dashboard 可能不是同一个口径。

### 21.1 Request-level hit rate

```text
有任意 prefix 命中的请求数 / 总请求数
```

这个口径最粗。一个请求命中 1 个 block 和命中 30k tokens 都算一次 hit，所以它不适合估算 serving 成本节省。

### 21.2 Per-request token/block hit rate

```text
cached_prefix_tokens / input_tokens
```

或者按 block：

```text
overlap_blocks / isl_blocks
```

例如输入 32k tokens，命中 15k tokens，单请求 hit rate 约等于：

```text
15k / 32k = 46.9%
```

Dynamo 文档里 `avg_kv_hit_rate` 的 replay 定义就是 per-request `overlap_blocks / isl_blocks` 的算术平均。

### 21.3 Aggregate arithmetic mean

```text
mean_i(cached_tokens_i / input_tokens_i)
```

这个考虑了每条请求内部的 token 数，但每条请求在平均时权重一样。

例子：

```text
request A: 1k input, 1k cached -> 100%
request B: 32k input, 0 cached -> 0%

arithmetic mean hit rate = 50%
```

但真实省掉的 prefill tokens 是：

```text
1k / 33k = 3.0%
```

所以它可能高估长请求 workload 下的收益。

### 21.4 Global token-weighted hit rate

```text
sum(cached_prefix_tokens) / sum(input_tokens)
```

这个更接近“省掉了多少 prefill token work”。对固定资源下的 agentic RL serving 优化，它通常比 request-level hit rate 或 arithmetic mean 更有解释力。

更进一步，最该看的不是 hit rate 本身，而是：

```text
saved_prefill_tokens_per_second
saved_prefill_tokens_per_gpu_hour
effective_prefill_tokens = input_tokens - cached_prefix_tokens
```

因为服务成本和吞吐最终跟省掉的 prefill work 相关。

### 21.5 仍然要和 worker load 一起看

即使用 token-weighted hit rate，也不能单独判断系统好坏。

如果为了提高 hit rate，把大量请求都打到 cache-rich worker：

```text
prefix hit rate 上升
worker hotspot 上升
decode queue 上升
ITL / total latency 变差
```

所以 serving 侧比较合理的三元组是：

```text
token-weighted prefix hit rate
saved prefill tokens / effective prefill tokens
worker prefill/decode load balance
```

## 22. SGLang-native router 和 Dynamo router 的边界修正

前面容易产生一个误解：好像如果 KV-aware routing 放在 SGLang router 里，它就能直接读 SGLang worker 进程里的 KV cache；而放在 Dynamo router 里才需要订阅 events。

这个说法不严谨。

如果 SGLang router 也是一个独立组件，而不是和每个 SGLang worker 共进程，那么它同样需要某种 worker cache-state signal：

```text
SGLang worker
  -> KV cache events / local index query
  -> SGLang router
  -> router-side prefix/cache index
  -> routing decision
```

也就是说，独立 router 天然都需要“worker 告诉 router cache 发生了什么”。真正的差异不是是否订阅 events，而是下面这些边界：

### 21.1 谁定义 cache event 语义

SGLang-native router 可以直接使用 SGLang 内部的 KV event schema、RadixAttention cache key、page/block 语义、multimodal hash 规则、eviction 语义。

Dynamo router 则需要把 SGLang/vLLM/TRT-LLM 各自的事件统一成 Dynamo 的 `RouterEvent` / `OverlapSignals` / `TieredMatchDetails`。

所以 Dynamo 多了一层 adapter/normalization，但换来跨 backend 的统一调度。

### 21.2 谁维护 router-side index

SGLang router 也需要维护自己的 prefix index，除非它把每次 routing 都 RPC 到 worker 查询实时 cache 状态。前者是常见设计，后者延迟和扩展性通常不好。

Dynamo 维护的是 engine-agnostic radix indexer；SGLang router 可以维护更贴近 SGLang RadixAttention 的 indexer。

### 21.3 谁维护 active load / booking

KV cache overlap 只解决“哪里有缓存”。独立 router 还必须解决“哪里现在忙不忙”。

所以 SGLang router 如果要达到 Dynamo KV router 的效果，也需要类似：

```text
AddRequest
MarkPrefillCompleted
Free
active prefill tokens
active decode blocks
request booking before response
multi-router replica sync
```

Dynamo 已经把这套 active-load accounting 放在 `lib/kv-router/src/sequences/*` 和 `lib/kv-router/src/scheduling/*`。

### 21.4 谁负责集群级控制面

SGLang-native router 更适合 SGLang-only serving stack：

- event schema 更直接
- cache key 对齐更简单
- 不需要跨 engine abstraction
- 对 SGLang 版本/内部语义可以更激进地优化

Dynamo router 更适合统一推理平台：

- 同一路由层支持 SGLang/vLLM/TRT-LLM
- 与 Dynamo frontend、worker discovery、Planner、Kubernetes CRD、Gateway API、disaggregated serving、KVBM 等集成
- 能在不同 backend 上复用同一套 active load、priority queue、taint/filter、DP rank、overload 逻辑

### 21.5 更准确的判断

如果只讨论 SGLang-only 部署，那么“这个功能放在 SGLang router 里”是完全合理的设计，而且它仍然会订阅 SGLang KV events。

如果讨论 Dynamo 的目标，即做跨 backend、跨节点、Kubernetes-native 的 inference orchestration，那么把 KV-aware routing 放在 Dynamo router 里也合理。它不是因为 Dynamo 能比 SGLang 更懂 SGLang cache，而是因为 Dynamo 想把“cache locality + active load + policy + discovery + scaling”做成统一的系统层能力。

## 23. 开放性问题：agentic RL serving 的核心瓶颈是什么？

结合前面的 KV-aware routing 讨论，可以把问题放宽到 agentic RL infra。

如果暂时不讨论算法、reward 稀疏、环境稳定性，而是只看固定资源下 serving/rollout 的吞吐，那么一个核心假设是：

```text
agentic RL rollout 的 serving 瓶颈，很大程度上取决于有效 prefix/KV cache reuse。
```

Agentic workload 天然有大量重复前缀：

```text
system prompt
tool schema
environment instruction
role/chat template
few-shot examples
session history prefix
memory/context prefix
```

如果每个 agent step 都重新 prefill 这些内容，固定 GPU 资源下会浪费大量 prefill compute。

### 23.1 但优化目标不是 raw hit rate

简单说“提高 prefix cache hit rate”方向是对的，但 raw hit rate 不一定足够解释 serving 效率。

更合理的 serving 目标是：

```text
最大化 saved prefill compute / GPU-hour
```

或者用更可观测的代理指标：

```text
saved_prefill_tokens / second
saved_prefill_tokens / GPU-hour
effective_prefill_tokens = input_tokens - cached_prefix_tokens
```

### 23.2 至少需要同时看三个维度

#### 1. Prefix cache hit rate

单请求口径通常是：

```text
cached_prefix_tokens / input_tokens
```

或按 block：

```text
overlap_blocks / input_blocks
```

这回答的是“这个请求有多少比例的输入前缀可以复用”。

#### 2. 命中的 token/block 数

命中率本身可能误导。

例如：

```text
request A: input 1k, cached 1k  -> 100%
request B: input 32k, cached 0  -> 0%
```

算术平均 hit rate 是 50%，但真实省掉的 token 只有：

```text
1k / 33k = 3%
```

所以系统级指标更应该看：

```text
sum(cached_prefix_tokens) / sum(input_tokens)
```

也就是 token-weighted hit rate。

#### 3. Worker 负载

高 cache hit 不一定等于高吞吐。

如果所有请求都被路由到 cache-rich worker：

```text
prefix hit rate 上升
worker hotspot 上升
decode queue 上升
ITL / total latency 变差
```

所以 routing 不能只最大化 overlap，还要考虑：

```text
active_prefill_tokens
active_decode_blocks
additional_active_blocks
queue depth
worker overload
```

Dynamo KV router 的 cost function 本质上就是把 cache overlap 视为收益，把 active prefill/decode load 视为代价。

### 23.3 还需要看 KV retention / eviction

Agentic RL 多轮 session 很长，prefix cache 能否命中不仅取决于 router，也取决于 KV cache 是否留得住。

这里要区分两个概念：

```text
潜在 reuse：workload 里确实有重复 prefix
实际 hit：下一次相关请求到来时，这个 prefix 仍然在 KV cache 里
```

如果可复用的 prefix 在下一次相关请求到来前已经被 eviction，那么这部分潜在 reuse 不会转化为实际 cache hit，也就不会节省 prefill。最终统计 hit rate 时，被 eviction 的 prefix 就是不命中。

需要关注：

```text
KV cache residency time
eviction rate
recomputed prefix tokens
cache capacity pressure
session locality
```

所以 prefix cache 优化不只取决于 workload 里有没有重复前缀，还取决于这些前缀能否在 KV cache 中保留到下一次复用发生。

### 23.4 暂定总结

可以把 serving 侧优化问题表述成：

```text
在固定资源下，agentic RL rollout serving 的核心优化目标，
是提高 token-weighted prefix/KV cache reuse，
并在此基础上保持 worker prefill/decode 负载均衡，
避免 cache-locality 造成热点和 decode 侧退化。
```

更短一点：

```text
目标不是最高 hit rate，
而是最多 saved prefill tokens / GPU-hour，
同时不牺牲 worker load balance 和 decode efficiency。
```

这个问题后续可以继续展开：

- agentic RL workload 下，token-weighted hit rate 和 throughput 的相关性有多强？
- session-affinity routing 和 global KV-aware routing 哪个更适合多轮 agent？
- tool schema/system prompt 是否应该独立做 persistent prefix cache？
- KV retention policy 是否应该感知 rollout/session 优先级？
- 长 session 的 KV cache 占用和短 session 的吞吐如何权衡？
- routing cost 里 prefill saving 和 decode load 的权重如何自动调？

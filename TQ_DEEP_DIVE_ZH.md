# TransferQueue 深入解析：从用户使用、核心流程到 Mooncake 后端原理

> 本文基于当前工作区中的 TransferQueue `d58019a`（版本线为
> `0.1.11.dev0`）、verl `549e5bfbf` 和 Miles `778227d6d` 阅读整理。
> 它描述的是这些代码在本文撰写时的实际行为，而不是只复述 README 中的愿景。

## 1. 先用一句话理解 TQ

TransferQueue（下文简称 TQ）是一个面向后训练数据流的“**样本状态表 + 可插拔分布式数据存储**”。

它解决的核心问题不是“如何把一个对象放进远端内存”，而是：

1. 一个训练样本会在 rollout、reward、log-prob、advantage、actor update 等阶段逐步长出新字段；
2. 不同阶段只应该在自己需要的字段已经产生后开始；
3. 不同消费者要有各自独立的消费进度；
4. 调度者只传递轻量元数据，真正的 Tensor 不再经过单一 trainer/driver 中转；
5. 底层数据可以由 SimpleStorage、MooncakeStore、Yuanrong 等不同后端搬运。

因此，与其把 TQ 想成传统 FIFO queue，不如把它想成一张会动态扩展的二维表：

```text
                         字段（pipeline 各阶段的输入/输出）
                 prompts   responses   rm_score   old_log_prob   advantage
样本 global_index=0    ready      ready       ready          ready       ready
样本 global_index=1    ready      ready       ready          ready       ready
样本 global_index=2    ready      ready       empty          empty       empty
样本 global_index=3    ready      empty       empty          empty       empty
```

Controller 只记录“哪一格已经 ready、哪个任务已经消费哪一行”；Storage backend 才保存和传输格子里的真实数据。

这是全文最重要的边界：

```text
控制面：Controller + metadata + sampler
        决定数据是否可读、由谁读、哪些样本算已消费

数据面：StorageManager + StorageClient + backend
        保存、传输、读取、删除 Tensor 和 Python 对象
```

## 2. 为什么后训练系统需要它

### 2.1 单 controller 搬大对象的问题

传统 verl 一类单 controller 架构很容易形成如下路径：

```text
rollout workers
      │ 大 DataProto / TensorDict
      ▼
RayPPOTrainer / driver
      │ 再切分、再分发大对象
      ├────────► reward workers
      ├────────► reference workers
      ├────────► critic workers
      └────────► actor workers
```

这里的 driver 同时承担算法编排和大对象中转，会带来四个问题：

- driver 的网络、CPU 序列化和 host memory 成为单点；
- 同一批数据会被反复收集、切分、发送；
- rollout 与训练阶段被大对象 RPC 的完成时机紧耦合；
- 多模态、长序列、变长 Tensor 会进一步放大内存峰值和传输成本。

TQ 将路径改成：

```text
                         轻量 key / BatchMeta
driver / trainer  ──────────────────────────────────► workers
      │                                                  │
      │ 只负责流程编排                                    │ 按需读取字段
      │                                                  ▼
      └────────────────── Controller              Storage backend
                            控制面                   真实 payload
                                                       ▲
                                                       │ 直接写入
                                                 rollout workers
```

driver 仍然是算法控制者，但不必成为所有 payload 的必经之路。

### 2.2 TQ 优化的是“逐步产生、按需消费”的数据流

后训练数据不是一次性构造完成的静态 batch。例如同一批 trajectory 可能按下面顺序补齐：

```text
prompt
  → rollout 写 responses / rollout_log_probs
  → reward 写 rm_scores
  → actor inference 写 old_log_probs / entropy
  → reference 写 ref_log_prob
  → critic 写 values
  → controller 计算后写 advantages / returns
  → actor/critic 读取所需字段进行更新
  → 清理
```

TQ 的“列式追加”正好对应这个过程。每个阶段可以只读自己需要的列，再将输出作为新列写回同一批样本。

## 3. 用户先要认识的六个概念

### 3.1 Partition：逻辑命名空间

`partition_id` 隔离不同数据集合，例如 `train`、`val`、`step_100`。同一个 key 或同一个字段名可以在不同 partition 中独立存在。

Partition 不是 Mooncake 的物理 segment，也不固定对应某个节点；它首先是 Controller 中的逻辑容器。

### 3.2 Global index：TQ 内部的样本地址

Controller 为每个样本分配进程间一致的整数 `global_index`。底层存储、production status 和 consumption status 都以它为稳定地址。

索引由 `PartitionIndexManager` 全局分配，清理后可以进入 reusable pool 被后续样本复用。Partition 自己记录哪些 global indexes 属于它。

### 3.3 Key：给 KV API 用户看的名字

高层 KV API 允许用户使用 `sample_123`、`uid_session_id` 一类字符串 key。Controller 内部维护：

```text
user key  ⇄  global_index
```

所以 KV API 并没有绕过 `BatchMeta`；它只是先把用户 key 翻译为 TQ 内部索引，再复用同一套 put/get/clear 数据路径。

### 3.4 Field：样本的一列数据

Field 是 `prompts`、`responses`、`rm_scores` 等列名。一个样本可以分多次追加字段，Controller 会动态扩展 production-status 矩阵。

字段 schema 包含：

- dtype；
- 统一 shape，或 nested tensor 的逐样本 shape；
- 是否 nested tensor；
- 是否非 Tensor；
- backend 私有元数据，例如 Mooncake 非 Tensor 的 `packed_size`、超大 GDR Tensor 的 `n_chunks`。

### 3.5 Production status：某个字段是否真正写完

Controller 为 partition 维护二维 `int8` 矩阵：

```text
production_status[global_index, field_column]
0 = 还没有生产完成
1 = 已写入 storage，可以读取
```

一个消费者请求多个字段时，只有某行的所有目标列都为 1，该样本才进入 ready 集合。

### 3.6 Consumption status：某个任务是否取过该样本

每个 `task_name` 都有独立的一维消费向量：

```text
consumption_status[task_name][global_index]
0 = 该任务尚未消费
1 = 该任务已经消费
```

因此同一个样本可以被 `reward`、`ref_log_prob`、`update_actor` 分别消费一次，互不影响。

需要注意：native `get_meta(mode="fetch")` 在 Controller 发出 metadata 时就按 sampler 的第二个返回值标记 consumed，而不是等 `get_data()` 成功后再标记。如果之后的数据读取失败，该任务的消费位仍然是 1，调用者需要通过 `reset_consumption()` 或自己的恢复逻辑处理。这更接近“调度时领取”而不是事务型消息队列的 ACK 语义。

## 4. 三套用户接口怎么选

### 4.1 KV API：最容易接入，也是当前 verl V1 的主要选择

主要接口是：

```python
tq.kv_put(...)
tq.kv_batch_put(...)
tq.kv_batch_get(...)
tq.kv_list(...)
tq.kv_clear(...)
```

适合：

- 已经有 ReplayBuffer 或单 controller 决定采样哪些 key；
- 需要按 key 更新 tag；
- 需要按字段局部读写；
- 希望用最小改造替换大对象传递。

KV API 的 tag 保存在 Controller 的 `custom_meta`，payload 保存在 storage backend。`kv_list()` 只列 key 和 tag，不会把大字段读回 driver。

它的限制也要说清楚：KV API 本身不使用 TQ 的 sampler 和 task-specific consumption tracking。用户通常像 verl 一样，在外部 ReplayBuffer 中解释 tag、选择 keys、处理 off-policy 和失败样本。

### 4.2 Native `TransferQueueClient`：完整控制生产/消费语义

典型流程是：

```python
meta = client.put(data, partition_id="train")

ready_meta = client.get_meta(
    data_fields=["prompts", "responses"],
    batch_size=8,
    partition_id="train",
    task_name="reward",
)
batch = client.get_data(ready_meta)

client.put(reward_output, metadata=ready_meta)
client.clear_samples(ready_meta)
```

适合：

- 想直接使用 production/consumption status；
- 需要自定义 sampler；
- 需要明确控制领取、读取、追加字段和清理；
- 希望构建完全流式 pipeline。

`mode` 的实际含义：

| mode | 行为 |
|---|---|
| `insert` | 创建/领取新 global indexes，返回未 ready 的 metadata，供首次写入 |
| `fetch` | 只扫描“目标字段全 ready 且该 task 未消费”的样本，再交给 sampler |
| `force_fetch` | 返回 partition 的全部已分配 indexes，绕过 sampler 和消费过滤；每行 ready 状态按实际字段计算 |

### 4.3 `StreamingDataset` / `StreamingDataLoader`：训练 rank 主动拉取

它将 `get_meta → get_data → micro-batch 切分` 包装成 PyTorch `IterableDataset`：

```text
DataLoader worker
  → 用 ZMQ 直接创建 TQ client
  → get_meta(batch_size)
  → get_data()
  → 缓存在 dataset.buffer
  → 切成 micro_batch_size
  → yield (TensorDict, BatchMeta)
```

它有两个工作模式：

- 默认无限流模式：没有数据就轮询等待；
- 有限数据集模式：`should_check_consumption_status=True`，确认 partition 全部消费且 buffer 已吐完后结束。

多 rank 时通常配合 `RankAwareSampler`。Sampler 以 `(partition_id, task_name, dp_rank, batch_index)` 缓存抽样结果，让同一数据副本组内的 ranks 得到相同 global indexes。

使用多进程 DataLoader 时，它不会在 fork 后调用依赖 Ray actor 查询的 `tq.init()`，而是用配置中的 ZMQ 地址直接创建 client。这是为了避免 fork 后使用 Ray runtime 的不安全路径。

## 5. 系统启动：`tq.init()` 实际做了什么

第一次初始化按下面顺序发生：

```text
1. 合并内置 config.yaml 和用户配置
2. 解析并实例化 sampler
3. 创建命名 Ray actor：TransferQueueController
4. 从 Controller 取得 ZMQ 地址
5. 调用所选 backend 的 bootstrap provider
6. 将最终配置保存到 Controller
7. 可选启动 metrics exporter
8. 在当前进程创建 TransferQueueClient
9. Client 创建对应 StorageManager / StorageClient
10. StorageManager 与 Controller 做 ZMQ handshake
```

其他进程再次调用 `tq.init()` 时：

```text
1. 通过 Ray namespace 找到已有 Controller
2. 从 Controller 取得第一次初始化时保存的配置
3. 在本进程创建自己的 Client 和 StorageManager/StorageClient
```

这意味着：

- Controller 是集群共享的；
- Python module 中的 `_TQ_CLIENT` 是进程内 singleton，不跨进程共享；
- 每个需要访问数据的 Ray actor/process 都要调用 `tq.init()`；
- 第一个 initializer 的 backend 配置决定整个 TQ 实例，后来进程传入的新配置会被忽略。

同步 `TransferQueueClient` 内部维护一个后台 asyncio event loop，把同步接口桥接到异步 client。Controller RPC 使用 ZMQ，而不是让每次 metadata 操作都经过 Ray method invocation。

## 6. 一批数据的完整生命周期

下面先不区分具体 backend，按真实先后顺序说明。

### 6.1 首次写入：先分配地址，再写 payload，最后发布 ready

用户调用：

```python
meta = client.put(data, partition_id="train")
```

内部顺序是：

```text
Client
  │
  ├─ 1. GET_META(mode="insert") ───────────────► Controller
  │                                               创建 partition（如需要）
  │                                               激活预分配 index 或分配新 index
  │                                               返回 production_status=0 的 BatchMeta
  │
  ├─ 2. StorageManager.put_data(data, meta) ─────► Storage backend
  │                                               写入真实 payload
  │
  ├─ 3. NOTIFY_DATA_UPDATE ─────────────────────► Controller
  │                                               登记 schema/backend meta
  │                                               将相应 field cells 置为 ready
  │
  ├─ 4. SET_CUSTOM_META（如果有 tag）────────────► Controller
  │
  └─ 5. 本地返回补齐字段后的 BatchMeta
```

最重要的正确性原则是：**payload 写成功后才发布 ready**。因此正常写入时，消费者不会看到 Controller 已声明 ready、storage 中却还没有数据的中间状态。

### 6.2 追加字段：复用同一批样本地址

某阶段收到 metadata 后写输出：

```python
client.put(output, metadata=meta)
```

这次不分配新 global indexes。StorageManager 使用原 indexes 写入新字段，成功后 Controller 只把新字段对应的列置为 1。

所以 TQ 的自然用法不是“每个 PPO 阶段建一个新对象”，而是让同一组样本逐步增加列。

### 6.3 Native 读取：先调度 metadata，再直接取 payload

```text
consumer
  │
  ├─ GET_META(fields, batch_size, task_name) ───► Controller
  │                                               过滤未 ready / 已消费样本
  │                                               sampler 选择 sampled indexes
  │                                               标记 consumed indexes
  │                                               返回 BatchMeta
  │
  └─ StorageManager.get_data(meta) ─────────────► Storage backend
                                                  只读取 meta 中指定字段
                                                  按 meta 顺序重建 TensorDict
```

Controller 不接触 payload；它只返回 global indexes、partition、字段 schema、状态和 backend metadata。

### 6.4 KV 读取：key 先翻译成 metadata

```text
kv_batch_get(keys, partition, select_fields)
  → Controller: keys → global indexes
  → 找出这些样本共同已经具备的字段
  → select_fields 进一步裁剪
  → 检查所有目标字段是否 ready
  → StorageManager.get_data(BatchMeta)
```

如果若干 keys 的字段集合不同，不指定 `select_fields` 时只会得到这些样本共同具备的列。这是高层 KV API 保证 batch 能统一重建的方式。

### 6.5 清理：先阻止新读，再删 payload，最后释放 metadata

`clear_samples()` 和 `kv_clear()` 的顺序是：

```text
1. Controller 将目标行 production_status 清零（mark clearing）
2. StorageManager 删除真实 payload
3. Controller 删除字段状态、tag、key mapping
4. IndexManager 释放 global indexes 供后续复用
```

这个顺序关闭了“storage 正在删除时又有新 consumer 领取数据”的窗口。

消费状态和数据删除是两回事：样本被某个 task 标记 consumed 后不会自动清理。业务侧仍需在所有需要它的阶段完成后明确调用 clear，否则 storage 会持续增长。

## 7. Controller 为什么这样设计

### 7.1 状态矩阵让依赖判断变成局部扫描

对一个请求 `(partition, fields, task_name)`，Controller 做的事情可以概括为：

```text
row_mask = 当前 task 尚未消费
col_mask = 请求的字段列
ready = production_status[row_mask, col_mask] 每行是否全为 1
```

它不理解 reward、actor、critic 的业务含义，也不维护硬编码 pipeline DAG。调用者只需声明“我需要哪些字段”，ready 条件就自然形成。

这就是 TQ 能复用于 PPO、GRPO、蒸馏、数据预处理甚至 KV cache/resharding 的原因：Controller 管理的是字段可用性，而不是算法阶段名。

### 7.2 Sampler 把“可读”与“选谁”分开

Controller 先得到 `ready_indexes`，再调用 sampler：

```python
sampled_indexes, consumed_indexes = sampler.sample(
    ready_indexes,
    batch_size,
    ...,
)
```

两个返回值可以不同：

- `sampled_indexes`：本次真正返回给调用者；
- `consumed_indexes`：本次之后永远不应再被当前 task 领取的样本。

这为 replacement、group sampling 和分布式 rank 协调留出了空间，而不需要把策略塞回 Controller 主逻辑。

内置实现包括：

- `SequentialSampler`：按 ready 顺序取前 N 个，并全部标记 consumed；
- `GRPOGroupNSampler`：按连续 N 条 group 组织样本；
- `RankAwareSampler`：缓存 `(dp_rank, batch_index)` 对应的样本；
- `SeqlenBalancedSampler`：按序列工作量做 DP balance，可返回不同 rank 的变长 batch。

### 7.3 预分配索引解决“生产尚未开始，消费者如何判断结束”

`TQ_PRE_ALLOC_SAMPLE_NUM` 会在 partition 创建时预留预期样本数。它主要服务有限 StreamingDataLoader：即使 producer 尚未写完，consumer 也能看到应有的消费状态长度。

如果完全流式场景需要准确判断一个 global batch 是否消费完，应将它设为已知的 global batch size；否则默认值 1 只够最小动态场景，不代表完整数据规模。

### 7.4 Polling mode 决定“数据不足时谁负责等待”

- `polling_mode=True`：Controller 返回空 `BatchMeta`，调用者自行 sleep/retry；适合 StreamingDataLoader 和 async producer-consumer。
- `polling_mode=False`：Controller 在有限超时内等待，随后抛出 `TimeoutError`；适合调用者认为数据理应已经就绪的阶段。

## 8. Storage 抽象与控制面/数据面的接口

所有 backend 都实现三个核心异步方法：

```python
put_data(data: TensorDict, metadata: BatchMeta)
get_data(metadata: BatchMeta) -> TensorDict
clear_data(metadata: BatchMeta)
```

`StorageManager` 还统一负责：

- 与 Controller 建立 handshake；
- payload 写成功后发送 `NOTIFY_DATA_UPDATE`；
- 使用独立 event loop/thread 发送通知并等待 ACK；
- 生命周期结束时清理 ZMQ 资源。

### 8.1 SimpleStorage 路径

SimpleStorage 由 TQ 自己创建多个 Ray storage-unit actors。Manager 按：

```text
target unit = global_index % num_storage_units
```

将样本稳定路由到不同单元，并并发 put/get/clear。它开箱即用，适合功能验证和普通 CPU memory 存储；但 storage-unit 数量变化会改变 hash 路由，当前没有动态扩容后的数据迁移机制。

### 8.2 KVStorageManager 路径

Mooncake、Yuanrong 和 RayStore 复用 `KVStorageManager`。它把二维数据展平成 field-major KV 列表：

```text
global indexes = [7, 9]
fields         = ["advantage", "responses"]

keys = [
  "7@advantage", "9@advantage",
  "7@responses", "9@responses",
]
```

值列表使用完全相同的 field-major 顺序。GET 后再按 metadata 将扁平 values 合并回 TensorDict。Nested Tensor 会按样本 unbind，普通 Tensor 也按第 0 维拆成每样本值，非 Tensor 列则恢复为 `NonTensorStack`。

这个中间层非常关键：Controller 不绑定 Mooncake，Mooncake client 也不理解 partition、task 或 sampler。两边只通过 `BatchMeta` 和扁平 KV 合同连接。

## 9. Mooncake 后端：先理解部署模型

### 9.1 TQ 中涉及的组件

```text
                    控制/位置查询
TQ process ─────────────────────────► mooncake_master
    │                                      │
    │ MooncakeDistributedStore client      │ 管理对象 metadata、placement、容量
    │                                      │ 不承载 TQ payload RPC 中转
    │
    ├── global segment：本 client 向 Store 贡献的 host-memory 容量
    ├── local buffer：Mooncake client 的本地传输缓冲
    └── 可选 GDR staging：TQ 自己创建并长期注册的 CUDA buffer
```

Mooncake master 是控制面，不是所有 Tensor 的中转站。真实数据由 Mooncake clients 与分布式 Store segments 通过 TCP/RDMA 传输。

`metadata_server` 和 `master_server_address` 也不是同一个概念：

- `metadata_server` 用于 Transfer Engine peer metadata；可使用 HTTP 地址，也可设为 `P2PHANDSHAKE`；
- `master_server_address` 是 Mooncake Store master RPC 地址，用于对象 metadata 和 placement。

### 9.2 TQ 的每个 client process 都会执行 Mooncake setup

每个调用 `tq.init()` 的进程都会创建自己的 `MooncakeStoreClient`，并使用同一份配置调用：

```python
MooncakeDistributedStore.setup(
    local_hostname,
    metadata_server,
    global_segment_size,
    local_buffer_size,
    protocol,
    device_name,
    master_server_address,
)
```

因此容量规划必须按 **Mooncake client 进程数** 考虑 `global_segment_size` 和 `local_buffer_size`，不能只看物理节点数。当前 TQ 没有像 Miles 那样按 driver、producer、trainer 角色自动把部分 client 设为 `global_segment_size=0`。

如果 verl 中 driver、AgentLoopWorkers 和各训练 workers 都调用 `tq.init()`，它们都会创建进程本地 Mooncake client。大规模部署前必须用真实进程拓扑核算内存注册与 segment 总量。

### 9.3 `auto_init` 的实际行为

`auto_init=True` 时，第一个 TQ initializer 会：

1. 查找并尝试终止已有的 `mooncake_master` 进程；
2. 启动新的 master；
3. 非 P2P 模式下同时开启 HTTP metadata server；
4. 如果启用 SSD offload，再启动一个独立 `mooncake_client` offload 进程。

这适合受控测试环境，但共享机器上有明显运维风险：它按进程名处理已有 master。生产环境更稳妥的做法通常是平台先管理好 Mooncake 服务，再设 `auto_init=False`。

另一个容易误解的点是：`tq.close()` 不会终止 master，只会警告用户手工处理；若 TQ 启动了 offload client，则会终止 offload client。初始化者还会尽力调用 `remove_all()` 清空 Store keys。

## 10. Mooncake PUT：一格数据是怎样写进去的

以 verl rollout worker 调用：

```python
await tq.async_kv_batch_put(
    keys=["uid0_0_0", "uid1_0_0"],
    partition_id="train",
    fields=trajectory_td,
    tags=trajectory_tags,
)
```

为例，完整路径如下。

### 10.1 key 注册和 metadata 准备

```text
KV API
  → Controller 将 user keys 映射成 global indexes，例如 [17, 18]
  → 返回 BatchMeta
  → tag 写入 BatchMeta.custom_meta
```

### 10.2 TQ 将二维字段展平

假设实际字段为 `responses` 和 `reward_score`：

```text
Mooncake keys:
17@responses
18@responses
17@reward_score
18@reward_score
```

这里使用的是 global index，不是原 user key。User key 只存在于 Controller mapping 中。

### 10.3 Tensor 与非 Tensor 分流

Mooncake client 将 values 分成：

- Tensor：走 raw pointer 的 `batch_upsert_from`；
- 非 Tensor：先编码成 bytes，再走相同的 pointer-based upsert。

普通非 GDR Tensor PUT：

```text
CUDA Tensor（如有）先拷到 CPU
  → contiguous
  → 合并可连续注册的 memory regions
  → register_buffer(ptr, size)
  → batch_upsert_from(keys, ptrs, sizes)
  → unregister_buffer
```

所以 `protocol=rdma` 不等于 GPU Tensor 自动走 GPUDirect。未启用 GDR 时，CUDA Tensor 仍先经过 CPU。

非 Tensor PUT：

```text
Python value
  → msgpack 编码；不支持的类型回退 pickle/cloudpickle
  → metadata frame + auxiliary buffers 打包进连续 uint8 buffer
  → register_buffer
  → batch_upsert_from
  → 返回 packed_size
```

`packed_size` 会写入 Controller 的 backend metadata，GET 时用它准确分配接收 buffer。

### 10.4 成功后才通知 Controller

Mooncake upsert 对失败的 keys 最多重试 3 次。所有写入完成后，`KVStorageManager` 才提取字段 schema，并通知 Controller：

```text
partition_id
global_indexes
field_schema
per-sample/per-field custom_backend_meta
```

Controller 收到 ACK 前不会把这些 cells 视为 ready。这就是数据面写入和控制面发布之间的提交边界。

## 11. Mooncake GET：为什么 Controller 不参与大数据传输

### 11.1 Controller 只返回重建所需的信息

KV/native 查询得到的 `BatchMeta` 包含：

- global indexes 和 partition；
- 目标字段；
- 每个字段的 dtype、shape 或逐样本 shape；
- 非 Tensor 的 `packed_size`；
- GDR 超大 Tensor 的 `n_chunks`。

随后 client 本地生成和 PUT 完全一致的 `global_index@field` keys，直接调用 Mooncake GET。

### 11.2 普通 Tensor GET

```text
根据 dtype/shape 分配最终 CPU Tensor
  → 尽量让多个结果共享连续 allocation
  → register_buffer(最终 Tensor 的 memory region)
  → batch_get_into 让 Mooncake 直接写入 Tensor backing memory
  → unregister_buffer
  → 按字段恢复 TensorDict / nested tensor
```

这里的“零拷贝”边界应该准确描述为：Mooncake 直接写入调用者预分配的最终 PyTorch Tensor buffer，避免先返回 Python bytes 再复制进 Tensor；网络传输本身当然仍然存在。

### 11.3 非 Tensor GET

```text
根据 packed_size 分配 uint8 buffers
  → batch_get_into
  → unpack metadata/auxiliary frames
  → msgpack/pickle decode
  → 重建 Python 对象、Tensor 或 ndarray view
```

解码出的 Tensor/ndarray 可以直接 view 接收 buffer；Python 引用链负责让底层 buffer 在结果仍存活时不被回收。

### 11.4 GET 失败和重试

`batch_get_into` 同样只重试失败 keys，最多 3 次。永久失败会抛异常，不会返回一个部分成功的 TensorDict。

## 12. Mooncake GDR：准确理解 GPU 数据路径

### 12.1 启用条件

至少需要：

```yaml
protocol: rdma
use_gdr: true
gdr_staging_buffer_mb: 1024
```

并且进程在创建 TQ client 前已经初始化 CUDA context。常见正确顺序是：

```python
torch.cuda.set_device(local_rank)
# 执行能确保当前进程 CUDA context 已初始化的操作
tq.init(config)
```

实际判断条件是：

```text
use_gdr=True
AND protocol="rdma"
AND gdr_staging_buffer_mb > 0
AND torch.cuda.is_initialized()
```

如果 CUDA context 尚未初始化，TQ 会静默使用普通 CPU RDMA 路径；如果已经进入 GDR lazy init、但硬件报告不支持 GDR，则首个传输会抛错。

### 12.2 GDR staging buffer 的设计

每个符合条件的进程创建一个 `GdrStaging` 对象，第一次使用时：

```text
cudaMalloc 固定大小 CUDA buffer
  → 向 Mooncake register_buffer 一次
  → 创建专用 CUDA stream
```

这个 buffer 在 client 生命周期内一直注册，并由一个 mutex 串行保护 PUT/GET。它不是跨进程共享池，也不是多 buffer 并发池。

选择单 buffer 的理由是 RL worker 内部的传输通常阶段性发生，能用有限 HBM 避免反复 cudaMalloc/register。代价是同一进程中的并发 GDR 调用会在锁上串行；高并发数据服务需要不同设计。

### 12.3 GPU Tensor PUT

```text
原 GPU Tensor
  → D2D copy 到已注册 staging buffer
  → RDMA 从 staging buffer 写入远端 Store segment
```

它消除了 `GPU → CPU → RDMA` 的 CPU bounce，但仍有一次 GPU 内的 D2D copy。因此更准确的说法是“GPUDirect 数据路径”，不是“从原 Tensor 到远端绝对零拷贝”。

### 12.4 CPU Tensor PUT

当 GDR 已启用时，CPU Tensor 也统一走：

```text
CPU Tensor
  → H2D copy 到 staging buffer
  → RDMA 写入 Store
```

这便于同一 client 使用一个注册路径，但不一定意味着 CPU-origin 数据总比普通 host RDMA 更快，实际应按 workload 测量。

### 12.5 Tensor GET

```text
Store
  → RDMA 直接写入 GPU staging buffer
  → D2D copy 到新分配的最终 CUDA Tensor
```

GDR GET 的输出 Tensor 创建在当前 CUDA device 上。即使生产者写入的是 CPU Tensor，只要消费者的 GDR 路径生效，读取结果也是 CUDA Tensor。业务代码不能假设“输出 device 与生产者输入 device 相同”。

### 12.6 多 Tensor packing 与 256-byte 对齐

小 Tensor 按大小排序后分组，每个 Tensor 在 staging buffer 中使用 256-byte aligned offset。只要一个 group 的对齐后总量不超过 buffer，就可以一次 batch upsert/get。

排序的目的是让小对象尽量装在一起，减少大 Tensor 造成的碎片和调用次数。

### 12.7 单个 Tensor 大于 staging buffer

超大 Tensor 会拆成：

```text
原 key: 17@responses

物理 keys:
17@responses:c0
17@responses:c1
...
```

Controller 为原 cell 记录 `{"n_chunks": N}`。GET 按 chunk 逐段写入 staging，再 D2D 拼到最终 CUDA Tensor；clear 也依赖 `n_chunks` 展开并删除所有子 key。

所以 backend metadata 不是可丢弃的调试信息，而是重建和完整清理数据所需的合同。

### 12.8 非 Tensor 不走 GDR

Python scalar、dict、list 等仍走 CPU bytes 编解码和 host RDMA。`use_gdr=True` 只改变 Tensor 路径，不会把任意 Python 对象变成 GPU-native 数据。

## 13. Mooncake SSD offload 与 pinning

TQ 的 offload 配置会启动一个独立 `mooncake_client`，在第一个 `tq.init()` 所在节点提供集中式 NVMe pool：

```text
任意节点的 Store DRAM segment
      │ master 触发 eviction
      ▼
首个初始化节点上的 standalone offload client
      ▼
file_storage_path 指定的 SSD
```

当前不是每节点一个 offload client，因此要注意：

- `local_hostname` 必须对所有节点可达；
- 该节点和 SSD 会成为集中式资源；
- 大集群的带宽和故障域需要单独评估。

Mooncake 的 hard-pinned 对象不能被 eviction。TQ 的默认策略是：

- offload 关闭：自动 `hard_pin=True`；
- offload 开启：自动 `hard_pin=False`；
- 用户显式设置 `hard_pin` 时覆盖自动行为。

关闭 offload 时，TQ 启动的 master 将 eviction watermark 设为不主动驱逐，并给对象很长的 lease/soft-pin TTL。因此**显式 clear 是容量闭环的一部分**，不能依赖短 TTL 自动回收。

## 14. Mooncake 容量和内存应该怎样估算

### 14.1 先数 client processes，而不是只数节点

TQ 将同一 `global_segment_size` 和 `local_buffer_size` 配置传给每个 Mooncake client process。粗略的配置基线应从下式开始：

```text
Σ 所有 client 的 global_segment_size
+ Σ 所有 client 的 local_buffer_size
+ Σ GDR client 的 gdr_staging_buffer_size（GPU HBM）
```

真实峰值还包括：

- ordinary GET 分配的最终 CPU Tensor；
- GDR GET 分配的最终 CUDA Tensor；
- PUT 中 CUDA→CPU 或 CPU/GPU→staging 的中间数据；
- 非 Tensor 编解码 buffer；
- Mooncake metadata、allocator 和 registration 开销；
- verl 后续 padding、nested-to-dense、DataProto 和训练 microbatch 临时数据。

### 14.2 Store 中保存的是“每样本、每字段”对象

TQ 当前没有 Miles structured-object backend 的 bundle/manifest 聚合层。一个 batch 的对象数近似为：

```text
样本数 × 字段数
```

GDR 超大 Tensor 还会乘 chunk 数。

这使 TQ 很适合字段级增量写入和选择性读取，但大量极小 cells 也会增加 KV metadata 和批处理固定开销。设计 field 粒度时不应把每个无意义的小 scalar 都拆成独立列。

### 14.3 GDR staging size 不等于数据集容量

staging buffer 只是每次传输的工作区。大 Tensor 可以分 chunk，因此它不需要容纳整个 partition；但 buffer 过小会增加子 key、RDMA 调用、Controller backend metadata 和 clear 工作量。

### 14.4 当前没有 TQ 级 replica 参数

TQ 创建 `ReplicateConfig` 并配置 hard pin，但当前公开 Mooncake 配置没有像 Miles 的 `--mooncake-replica-num` 一样暴露副本数。不要把 Miles 文档中的 replica 配置直接套到 TQ。

## 15. Mooncake 清理与生命周期

### 15.1 删除单批样本

```text
kv_clear(keys)
  → keys 映射为 global indexes
  → Controller mark_clearing
  → KVStorageManager 生成 global_index@field keys
  → MooncakeStoreClient.batch_remove(force=True)
  → GDR 大对象根据 n_chunks 删除 :cN 子 keys
  → Controller 清除 metadata/key mapping 并释放 index
```

Mooncake 返回“key 不存在”时会当作幂等成功。其他 remove error 当前主要记录日志，`clear()` 不向上抛出；因此异常清理场景应结合 Store 指标或独立检查，防止 Controller 已释放 metadata、远端却残留 orphan keys。

### 15.2 `tq.close()` 不是业务数据生命周期 API

正常训练应逐 batch/partition clear。`tq.close()` 的作用是结束进程内 client、清理连接和销毁 Controller；初始化进程对 Mooncake 还会尝试 `remove_all()`。

不能把所有数据都留到 `close()` 再回收：长任务会先耗尽 segment，而且进程异常退出时也未必能执行 close。

## 16. verl V1 中的真实 TQ 流程

当前 verl V1 是理解 TQ 用户价值的最好例子。它主要使用高层 KV API，并在 verl 自己的 `ReplayBuffer` 中实现业务采样策略。

### 16.1 启动

`TaskRunnerV1.run()` 强制启用 TQ，并调用：

```python
tq.init(config.transfer_queue)
```

Driver 创建共享 Controller。AgentLoopWorkers 和接收 `KVBatchMeta` 的计算 workers 会在各自进程中再次 `tq.init()`，连接已有实例。

### 16.2 提交 prompt

Trainer 从 dataloader 取 prompt，为每条生成 uid，然后先只写 tag：

```text
key = uid
tag = {
  is_prompt: true,
  status: pending,
  global_steps: current_step,
}
```

同步 trainer 只写 tag；async trainer 还会保存 prompt fields，供 checkpoint 后重发 in-flight prompts。

随后 Trainer 只把 prompt batch dispatch 给 AgentLoopManager，不等待所有 trajectory 完成。

### 16.3 AgentLoopWorker 更新 group 状态并写 trajectory

一个 prompt group 的 tag 状态依次为：

```text
pending → running → finished
                  ↘ failure
```

每个 rollout session/output 使用独立 trajectory key：

```text
{uid}_{session_id}_{output_index}
```

AgentLoopWorker 将 `prompts`、`responses`、`input_ids`、`position_ids`、`loss_mask`、log-prob、reward extra info、多模态预处理结果等组织成 TensorDict，一次 `async_kv_batch_put()` 写入。大 payload 直接进入 TQ storage，driver 只需要通过 `kv_list()` 看 tag。

### 16.4 verl ReplayBuffer 在控制面采样

ReplayBuffer 轮询：

```python
tq.kv_list()
```

并只根据 Controller 中的 tag 构建：

- pending/running/finished/failure prompt sets；
- trajectory key/tag snapshot；
- global step 和 off-policy staleness；
- DAPO group filter 结果。

当 terminal prompt groups 足够时，它选择 uids，删除不再需要的 prompt marker，再返回只含 trajectory keys/tags 的 `KVBatchMeta`。

也就是说，verl 使用的是：

```text
TQ Controller tags 作为轻量 replay index
+ TQ storage fields 作为 trajectory payload
+ verl ReplayBuffer 作为领域采样器
```

它没有使用 native `BaseSampler`，这是 KV API 与 native streaming API 职责差异的直接体现。

### 16.5 PPO 各阶段只读所需字段并写回新字段

一批 `KVBatchMeta` 进入 PPO step 后，典型顺序是：

```text
ReplayBuffer sample keys
  → reward 阶段 GET prompts/responses/raw_prompt，PUT rm_scores
  → balance 只重排 KVBatchMeta 的 keys/tags
  → actor inference GET 输入，PUT log_probs/entropy
  → controller 转换并 PUT old_log_probs
  → reference GET 输入，PUT ref_log_prob
  → critic GET 输入，PUT values
  → advantage 阶段只取所需 fields，PUT advantages/returns
  → critic/actor update 按 metadata 在 worker 本地取数据
  → metrics/dump 按需读少量 fields
  → step 完成后 kv_clear trajectory keys
```

`tqbridge` decorator 是旧有 worker API 与 TQ 之间的桥：

```text
KVBatchMeta
  → 在 worker 进程中转换为 BatchMeta
  → worker 本地调用 TQ storage GET
  → 原计算函数看到 TensorDict
  → 输出 TensorDict 在 worker 本地 PUT 回同一 global indexes
  → Ray 返回轻量 KVBatchMeta
```

因此从 driver 看，原 worker 方法仍像是在接收和返回 batch；真正的大 Tensor 已经不通过 driver/Ray 返回值来回搬运。

### 16.6 换成 Mooncake 后，哪些层改变、哪些不变

不变的层：

- verl 的 prompt/trajectory key 设计；
- tag 状态机和 ReplayBuffer；
- PPO 阶段先后顺序；
- `KVBatchMeta` 在 Ray 控制流中的传递；
- 每阶段 `select_fields` 和写回字段。

改变的层：

```text
KVStorageManager 下方
SimpleStorage ZMQ storage units
        ↓ 替换为
MooncakeDistributedStore + TCP/RDMA/GDR + segments
```

这就是可插拔 backend 的真正价值：算法控制流不需要理解 RDMA buffer registration、GDR chunk 或 Mooncake placement。

## 17. 与 Miles Mooncake rollout backend 的对照

Miles 文档非常适合帮助理解“控制引用与 payload 分离”的共同原则，但不能把它的实现细节直接当成 TQ 行为。

### 17.1 共同点

两者都遵循：

```text
调度系统只传轻量 ref/metadata
真实 rollout payload 进入专门数据面
consumer 在自己进程本地读取
所有使用者完成后明确清理远端对象
```

两者都必须关注：

- master/metadata 服务只管理控制信息，不搬运大 payload；
- TCP 与 RDMA 是数据 transport 选择，不改变训练算法；
- local buffer、registered memory、Store segment 要分开做容量规划；
- 异步 pipeline 不等于单次 PUT/GET 是 lazy future；
- “零拷贝”必须说明省掉了哪一段 copy，而不是宣称数据从未搬动。

### 17.2 核心差异

| 维度 | TQ + Mooncake | Miles + Mooncake structured object store |
|---|---|---|
| 上层对象 | 动态二维样本表 | 一轮 rollout 的完整 dict/bundle |
| 物理 key 粒度 | 每个 `(global_index, field)` 一个 key | manifest + field payload/chunks |
| 发布点 | payload 写完后通知 TQ Controller，将 cell 置 ready | 所有 payload 完成后最后发布 manifest |
| 部分读取 | TQ 按 fields 和 sample keys 天然选择 | structured layer 支持 field/row/range，但 Miles 当前常用完整 eager GET |
| 变长数据 | 每样本 Tensor，GET 后重建 nested tensor | typed-ragged 连续 data + offsets/shapes/views |
| 本地结果 | TQ 预分配最终 PyTorch Tensor，Mooncake `get_into` 写入 | BufferPool lease 上创建 ndarray/ragged views |
| 生命周期 | clear cells/keys，释放 Controller index | trainer release local result；driver remove remote bundle，两层所有权 |
| segment 角色 | 当前 TQ 对所有 client 使用同一配置 | Miles 按角色决定哪些 client 贡献 segment |
| 副本配置 | 当前 TQ 未暴露 replica num | Miles 暴露 `--mooncake-replica-num` |

### 17.3 两种设计分别擅长什么

TQ 的 cell 模型擅长：

- rollout 完成一条就能写一条；
- pipeline 逐字段追加；
- 按 keys/fields 细粒度读取；
- Controller 统一看见每个样本的状态；
- 不同 task 独立消费。

Miles bundle 模型擅长：

- 一次交接一个已经完成的大 rollout dict；
- 利用显式 schema 把大量碎片化 ndarray rows 聚合为少量大 I/O；
- manifest 统一发布和回滚一个完整逻辑对象；
- typed-ragged GET 返回连续 buffer 上的 views。

二者的设计目标有交集但不相同。不能因为 Miles 对大 ragged bundle 的 benchmark 很好，就自动推出 TQ 的“每样本×每字段 KV”布局在相同 payload 上具有同样的对象数和吞吐；反过来，也不能要求 Miles bundle 天然提供 TQ 的逐 cell readiness 和 task consumption。

### 17.4 Miles 给 TQ 部署的直接启示

Miles 文档中最值得迁移到 TQ 运维实践的经验是：

1. 先用 TCP 验证功能，再切 RDMA/GDR；
2. 按实际 client 数计算 buffer/registration，而不是只按节点；
3. Store 远端副本、consumer 本地结果和 GPU Tensor 会同时存在；
4. 大对象“能分 chunk 传”不代表 consumer 可以用有界内存训练；
5. 对超长上下文、多模态和 R3，必须测完整 step 的 host/GPU 峰值和清理时延，而不是只测单次 GET 带宽；
6. 最终消费粒度若是 microbatch/PP layer/CP range，最好让读取粒度尽早接近它，避免整批 materialize 后再切。

## 18. 一致性、失败语义和恢复边界

### 18.1 PUT 的一致性边界

TQ 保证正常路径是 storage first、metadata second，所以不会主动发布尚未写完的数据。但它不是跨 Mooncake keys 与 Controller 的分布式事务：

- storage 写成功、notify Controller 失败时，可能存在尚未被 Controller 看见的 orphan payload；
- Controller notify 成功后 client 崩溃，数据仍可被其他 client 读取；
- 多字段 upsert 的底层失败会按 key 重试，永久失败则整个 put 抛错。

业务要通过幂等 key、partition 清理和监控处理极端故障，而不能假设有两阶段事务回滚。

### 18.2 GET 的消费语义

Native fetch 在发出 metadata 时标记 consumed，因此 GET 失败不会自动重新入队。KV API 不使用该 consumption bit，通常由 tag/ReplayBuffer 自己决定是否重试。

这也是 verl V1 选择 KV + 自定义 ReplayBuffer 的原因之一：它需要 pending/running/finished/failure、off-policy 和 group refill 等比通用 sampler 更丰富的业务状态。

### 18.3 Checkpoint

Controller checkpoint 包含：

- partitions 和 production/consumption/custom metadata；
- global-index allocator；
- sampler cache/state。

SimpleStorage 还支持保存所有 storage units，保存顺序为 Controller first、storage second，以保证 checkpoint 中 Controller 声称 ready 的数据已经进入随后保存的 storage snapshot。

Mooncake manager 当前没有实现 TQ storage checkpoint。`tq.save_checkpoint()` 会记录警告并退化为 controller-only checkpoint。只有当外部 Mooncake 数据在重启后仍以完全相同 keys 可用时，这种恢复才自洽；如果 master/segments 一起丢失，单独恢复 Controller 会得到指向不存在 payload 的 metadata。不要把 controller-only snapshot 当成完整 Mooncake 容灾方案。

### 18.4 Close 和 owner

只有首次 initializer 持有 bootstrap resources 并负责集群级清理。其他进程的 `tq.close()` 主要关闭自己的 client。应用应由拥有全局生命周期的 driver 最后关闭共享 TQ，worker 退出时不要抢先销毁 Controller。

## 19. 可观测性：用户应该看什么

开启 metrics 后，Controller 提供：

- partition 数和每 partition 样本数；
- production/consumption progress；
- allocated/reusable global indexes；
- Controller 请求吞吐、延迟、错误和样本数；
- Controller RSS。

SimpleStorage 还提供每 storage unit 的容量、active keys、utilization、延迟和 RSS。当前这套 storage-unit metrics 注册只针对 SimpleStorage；Mooncake backend 主要能看到 TQ Controller 侧状态，Mooncake segment/local-buffer/RDMA/offload 的容量与传输指标需要从 Mooncake 自身补充。

最有用的泄漏判断是：

```text
NOTIFY_DATA_UPDATE samples/s
vs
CLEAR_META samples/s
```

以及 active keys 是否长期单调上升。Mooncake 场景还应至少观察：

- 每个 client/segment 的 allocated/live bytes；
- RDMA registration、memlock 和失败码；
- PUT/GET latency 与 retry；
- GDR staging HBM；
- offload queue、SSD 使用量和 eviction；
- verl step 结束到 `kv_clear` 完成的时间。

## 20. Mooncake 最小配置与推荐验证顺序

### 20.1 先用外部 master + TCP

生产环境建议先由平台启动并管理 master，然后配置：

```yaml
controller:
  polling_mode: true

backend:
  storage_backend: MooncakeStore
  MooncakeStore:
    auto_init: false
    metadata_server: "<reachable-host>:50050"
    master_server_address: "<reachable-host>:50051"
    local_hostname: ""          # 使用 Ray node IP 自动发现；确认它是数据网地址
    protocol: tcp
    global_segment_size: 4294967296
    local_buffer_size: 1073741824
    device_name: ""
    hard_pin: null
    use_gdr: false
    gdr_staging_buffer_mb: 1024
```

`local_hostname` 必须是其他节点可路由的地址。跨节点不能使用 `localhost` 或错误的管理网 IP。

### 20.2 验证顺序

建议严格按下面顺序推进：

1. 单进程、单节点、TCP，验证 KV tensor/non-tensor round trip；
2. 多进程、单节点，核对每进程内存与清理；
3. 多节点 TCP，确认 hostname、master 和 metadata server；
4. RDMA 但不启用 GDR，确认 NIC、驱动、端口、GID、memlock；
5. 设置 CUDA device 并初始化 context 后启用 GDR；
6. 验证 GPU Tensor、CPU Tensor、nested Tensor、非 Tensor 和超 staging 大 Tensor；
7. 运行 verl 小规模完整 step，确认 trajectory 在 step 后清理；
8. 扩大 batch/context，记录 Controller、Mooncake、host 和 GPU 全链路峰值；
9. 最后再考虑 SSD offload、长稳和故障注入。

### 20.3 对应测试入口

仓库已经提供可作为阅读和验证依据的测试：

```bash
TQ_TEST_BACKEND=MooncakeStore \
  python -m pytest -q tests/e2e/test_kv_interface_e2e.py

TQ_TEST_BACKEND=MooncakeStore \
  python -m pytest -q tests/e2e/test_e2e_lifecycle_consistency.py

TQ_TEST_BACKEND=MooncakeStore_GDR \
  python -m pytest -q tests/e2e/test_e2e_lifecycle_consistency.py

python -m pytest -q tests/test_mooncake_utils.py
```

GDR E2E 覆盖 GPU Tensor round trip 和 CPU Tensor 经 H2D staging 的 round trip；工具单测覆盖 256-byte alignment、分组、超大 Tensor 子 key 和 clear 展开。

## 21. 常见误解和结论

### 误解 1：TQ 就是一个 FIFO 队列

不是。它是动态的样本×字段状态表，sampler 决定取样策略，partition 和 task 又提供多维隔离。

### 误解 2：Controller 保存或转发 Tensor

不是。Controller 保存 metadata 和状态；payload 由 storage data plane 直接移动。

### 误解 3：`kv_put` 绕过了 Controller/native path

不是。KV key 会先映射为 global index，然后仍走 BatchMeta、StorageManager、notify-ready 这套路径。

### 误解 4：`protocol=rdma` 就是 GPU Direct

不是。还必须启用 `use_gdr`、设置非零 staging buffer、提前初始化 CUDA context，并具备 GDR 硬件能力。

### 误解 5：GDR 完全没有 copy

不是。TQ 当前使用持久 staging：PUT 有 D2D/H2D copy，GET 有 staging 到最终 CUDA Tensor 的 D2D copy；它消除的是 CPU bounce 和重复注册开销。

### 误解 6：消费后会自动删除

不会。Consumption bit 只影响 sampler 是否再次发放。业务必须在所有阶段结束后 clear。

### 误解 7：Miles 的 Mooncake bundle 机制就是 TQ Mooncake 实现

不是。Miles 使用 structured bundle/manifest/typed-ragged/BufferPool；TQ 当前使用每样本每字段 KV、BatchMeta schema 和 caller-allocated Tensor GET。两者共享分离控制面/数据面的原则，但物理布局、生命周期和内存模型不同。

### 误解 8：启用 Mooncake 后 checkpoint 已经完整容灾

不是。当前 TQ 只会可靠保存 Controller metadata；Mooncake storage checkpoint 未由 TQ 实现，外部 Store 数据是否可恢复需要独立保证。

## 22. 最终设计评价

TQ 最有价值的设计不是某一种通信库，而是三层解耦：

```text
业务层：verl ReplayBuffer / PPO pipeline
        只定义 key、tag、需要的 fields 和阶段顺序

控制层：Controller / BatchMeta / Sampler
        管理 partition、global index、ready 和 consumption

数据层：StorageManager / Mooncake client
        管理物理 key、buffer、TCP/RDMA/GDR、offload 和删除
```

这三层让后训练系统可以保留清晰的算法控制流，同时把大数据从单 controller 路径中移走。Mooncake backend 又进一步把数据面扩展到分布式 segments、RDMA、GDR 和 SSD offload。

它当前最需要用户认真管理的边界也同样清楚：

- KV 模式的业务状态和采样由上层负责；
- native fetch 是领取即 consumed，不是 get-success ACK；
- 每个 TQ process 都会创建 Mooncake client，内存要按进程核算；
- TQ Mooncake 是 cell-level KV，不是大 bundle 自动聚合器；
- GDR 使用单 staging buffer，换取有界 HBM和低注册开销，但同进程传输串行；
- clear、监控和完整故障恢复仍需应用与部署系统共同闭环。

只要把这些边界纳入设计，TQ 就不只是“更快地传 Tensor”，而是一个能让 rollout、reward、inference 和 training 围绕共享数据状态独立演进的数据协调层。

## 23. 主要代码与文档索引

TransferQueue：

- [`transfer_queue/interface.py`](transfer_queue/interface.py)：初始化、KV API、checkpoint API
- [`transfer_queue/client.py`](transfer_queue/client.py)：同步/异步 client、put/get/clear 顺序
- [`transfer_queue/controller.py`](transfer_queue/controller.py)：partition、状态矩阵、sampling、key mapping
- [`transfer_queue/metadata.py`](transfer_queue/metadata.py)：`BatchMeta`、`KVBatchMeta` 和字段 schema
- [`transfer_queue/sampler/`](transfer_queue/sampler/)：内置采样器
- [`transfer_queue/storage/managers/base.py`](transfer_queue/storage/managers/base.py)：storage 抽象与 KV flatten/rebuild
- [`transfer_queue/storage/clients/mooncake_client.py`](transfer_queue/storage/clients/mooncake_client.py)：Mooncake CPU RDMA/GDR PUT/GET/clear
- [`transfer_queue/utils/mooncake_utils.py`](transfer_queue/utils/mooncake_utils.py)：GDR staging、alignment、chunking
- [`transfer_queue/storage/bootstrap/mooncake_bootstrap.py`](transfer_queue/storage/bootstrap/mooncake_bootstrap.py)：master 与 SSD offload 启动
- [`transfer_queue/config.yaml`](transfer_queue/config.yaml)：默认配置
- [`docs/storage_backends/mooncake_gdr.md`](docs/storage_backends/mooncake_gdr.md)：GDR 使用说明
- [`docs/metrics.md`](docs/metrics.md)：指标和 Grafana
- [`docs/checkpoint.md`](docs/checkpoint.md)：checkpoint 设计与限制

verl 对照：

- [`verl/trainer/main_ppo.py`](/mnt/shared-storage-user/huanghaian/code/verl/verl/trainer/main_ppo.py)：V1 TQ 初始化
- [`verl/trainer/ppo/v1/agent_loop_tq.py`](/mnt/shared-storage-user/huanghaian/code/verl/verl/trainer/ppo/v1/agent_loop_tq.py)：rollout 状态和 trajectory 写入
- [`verl/trainer/ppo/v1/replay_buffer.py`](/mnt/shared-storage-user/huanghaian/code/verl/verl/trainer/ppo/v1/replay_buffer.py)：tag 同步、group sampling、淘汰与 refill
- [`verl/trainer/ppo/v1/trainer_base.py`](/mnt/shared-storage-user/huanghaian/code/verl/verl/trainer/ppo/v1/trainer_base.py)：PPO 各阶段字段读写与 step 清理
- [`verl/utils/transferqueue_utils.py`](/mnt/shared-storage-user/huanghaian/code/verl/verl/utils/transferqueue_utils.py)：`tqbridge` metadata/data 转换
- [`verl/trainer/config/transfer_queue/transfer_queue.yaml`](/mnt/shared-storage-user/huanghaian/code/verl/verl/trainer/config/transfer_queue/transfer_queue.yaml)：verl 配置入口

Miles 对照：

- [`miles_mooncake_rollout_data_transfer.md`](/mnt/shared-storage-user/huanghaian/code/slime_package/miles/hha_code/miles_mooncake_rollout_data_transfer.md)：Mooncake structured rollout bundle、内存与生命周期分析


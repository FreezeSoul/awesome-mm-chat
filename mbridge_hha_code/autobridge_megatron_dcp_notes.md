# AutoBridge、Megatron Checkpoint 和 DCP Reshard 关系梳理

更新时间：2026-07-12

## 1. Megatron-Bridge 和 AutoModel 的核心区别

Megatron-Bridge 和 AutoModel 都属于 NeMo 体系里的模型训练/适配路线，但底层并行抽象不同：

| 库 | 底层路线 | 主要定位 |
| --- | --- | --- |
| Megatron-Bridge | Megatron-Core / Megatron-LM | HF 模型到 Megatron-Core 的 bridge、checkpoint 转换、训练 recipe、SFT/LoRA |
| AutoModel | PyTorch DTensor | 更 PyTorch-native / DTensor-native 的预训练路线 |
| NeMo-RL | Post-training / RL | 可接 Megatron 或 AutoModel backend |

Megatron-Bridge 的关键点是：它不是单纯模型实现库，而是把 Hugging Face 模型接到 Megatron-Core 高性能训练体系上的转换层和训练层。

## 2. AutoBridge 可以独立用吗

可以独立于 Megatron-Bridge 的训练 loop 使用，但不能脱离 Megatron-Core 生态。

AutoBridge 可以单独做：

- 判断 HF 模型是否支持：`AutoBridge.can_handle(...)`
- 只用 HF config 创建 Megatron provider：`AutoBridge.from_hf_config(...)`
- HF 权重导入 Megatron 模型：`from_hf_pretrained(...).to_megatron_model(...)`
- Megatron checkpoint 导出回 HF：`export_ckpt(...)` / `save_hf_pretrained(...)`
- 流式导出权重：`export_hf_weights(...)`
- 导出 LoRA/adapter：`save_hf_adapter(...)`

但 AutoBridge 生成的是 Megatron-Core provider/model，转换逻辑依赖：

- `megatron-bridge`
- `megatron-core`
- 已注册的模型 bridge / provider / mapping
- Megatron 的分布式初始化和 checkpoint 体系

所以它不是一个纯 Hugging Face 工具，而是 Megatron-Core 的适配入口。

## 3. HF 模型接 Megatron 训练的典型链路

典型流程：

```text
Hugging Face checkpoint
  -- AutoBridge: HF -> Megatron 权重名/布局映射 -->
Megatron distributed checkpoint
  -- Megatron-Core / Megatron-LM 训练 -->
Megatron distributed checkpoint
  -- AutoBridge: Megatron -> HF 权重名/布局映射 -->
Hugging Face checkpoint
```

AutoBridge 负责的事情：

- 读取 HF config，生成 Megatron-Core provider/config
- 做 HF 参数名和 Megatron 参数名映射
- 处理 QKV fusion / split
- 处理 gated MLP 的 gate/up fusion / split
- 处理 TP/PP/EP 下 Megatron 参数布局
- 保存为 Megatron-format checkpoint
- 训练后反向导出为 HF checkpoint

注意：

- 如果从 HF 预训练权重继续训练，需要发生 HF -> Megatron 权重转换。
- 如果从零预训练，只需要 HF config/架构，不需要转换权重，可用 `load_weights=False`。
- 如果外部不用 Bridge 训练，而是需要 Megatron checkpoint，通常需要先用 AutoBridge/mbridge 落盘一份 Megatron checkpoint。

## 4. AutoBridge 落盘的 Megatron checkpoint 是否绑定固定并行策略

结论：模型权重一般不强绑定固定 TP/PP/DP。保存时会按某个并行配置实例化并保存，但如果保存格式是 Megatron-Core 的 distributed checkpoint，例如 `torch_dist`，加载时通常可以换 TP/PP/DP 后重新 reshard。

更准确地说：

```text
绑定的是模型结构、参数 key、checkpoint schema 和 sharding metadata；
通常不绑定某一组永久固定的 TP/PP/DP rank 文件布局。
```

适用场景：

- HF -> Megatron model-only checkpoint：通常转换一次即可，多组实验可以用不同 TP/PP 加载。
- 完整训练 checkpoint resume：如果包含 optimizer/RNG/dataloader/scheduler 状态，换并行度限制更多。
- DP 变化一般最安全。
- TP/PP 变化通常依赖 `torch_dist` distributed checkpoint 的 reshard 能力。
- EP/MoE 更依赖模型实现、provider、expert 参数命名和 checkpoint mapping 是否一致。

## 5. Megatron-LM / Megatron-Core 如何实现 reshard

核心不是按文件名猜测，而是每个参数都带 sharding metadata。

保存时：

```text
当前 Megatron 模型
  -> model[i].sharded_state_dict()
  -> 每个参数包装成 ShardedTensor
  -> dist_checkpointing.save(...)
```

`ShardedTensor` 里包含：

- `key`：全局参数名
- `global_shape`：完整参数形状
- `local_shape`：当前 shard 形状
- `global_offset`：当前 shard 在完整 tensor 里的偏移
- `axis_fragmentations`：每个轴被切成几份
- `replica_id`：副本信息

加载时：

```text
当前新并行配置下的 Megatron 模型
  -> 生成目标 sharded_state_dict()
  -> dist_checkpointing.load(target_sharded_state_dict, checkpoint_dir, ...)
  -> loader 根据 key + shape + offset 读取旧 checkpoint 中对应切片
  -> 填入当前 rank 的参数
```

所以 TP=4 保存、TP=2 加载时，不是旧 TP rank 和新 TP rank 通信。旧进程已经不存在。实际发生的是：

```text
checkpoint storage: 保存了 TP=4 的 4 份 shard + metadata
当前运行: TP=2
当前 rank 生成自己需要的目标 shard
DCP/Megatron loader 根据 metadata 从 checkpoint storage 读取重叠 chunk
拼成当前 rank 需要的 tensor
```

简单例子：

```text
完整 W: [0:100]

保存时 TP=4:
  shard0 = W[0:25]
  shard1 = W[25:50]
  shard2 = W[50:75]
  shard3 = W[75:100]

加载时 TP=2:
  rank0 需要 W[0:50]   -> 读取旧 shard0 + shard1
  rank1 需要 W[50:100] -> 读取旧 shard2 + shard3
```

## 6. 这个能力是 Megatron 原生还是 PyTorch DCP 提供

分层理解最准确：

```text
Megatron-Core / Megatron-LM 做：
  - 给 Megatron 参数生成 ShardedTensor metadata
  - 声明哪些参数按 TP axis 0/axis 1 切
  - 声明 PP layer ownership
  - 处理 MoE expert 参数、optimizer/RNG、release checkpoint、strictness
  - 可选 fully_parallel_load/save 的 IO 分摊和 rank 间 exchange

PyTorch DCP 做：
  - 根据 checkpoint metadata 和目标 tensor shard metadata
  - 从 checkpoint storage 中读取对应 chunks
  - 把旧 sharding 布局的数据填到新 sharding 布局的 tensor
```

所以，“TP=4 保存、TP=2 加载”的核心重切片能力主要来自 PyTorch DCP；Megatron-Core 提供模型并行语义和 metadata。

DCP 自己并不知道哪个 Megatron 参数是 column-parallel、row-parallel、PP layer 或 MoE expert。这些信息是 Megatron-Core 的 `sharded_state_dict()` 生成的。

## 7. 普通 load 和 fully parallel load 的区别

普通 `torch_dist` load：

```text
每个当前 rank 根据自己的目标 ShardedTensor
直接从 checkpoint storage 读取需要的数据
通常不需要当前 TP ranks 之间通信来完成重切片
```

`--ckpt-fully-parallel-load`：

```text
当前 ranks 先交换 metadata
统一规划每个 rank 负责读取哪些 shard
每个 rank 只读分配到的 checkpoint 数据
再在当前 ranks 之间 exchange loaded tensors
```

fully parallel load 的目的主要是均衡 IO、减少重复读取或提升加载吞吐；这个路径会发生当前 ranks 之间的数据通信。

## 8. slime 的 convert_hf_to_torch_dist.py 是什么做法

slime 这个脚本的关键链路：

```text
初始化 Megatron distributed args
  -> get_model(...) 构造当前 Megatron 模型
  -> mbridge AutoBridge 加载 HF 权重到 Megatron 模型
  -> Megatron-LM save_checkpoint(...)
  -> tracker 改成 release checkpoint
```

它不是简单把 HF safetensors 改名，也不是自己写固定 rank 文件，而是调用 Megatron-LM 的 `save_checkpoint()` 保存成 Megatron/MCore 能理解的 checkpoint。

因此：

- 转换时确实会用某个 TP/PP 配置实例化模型并保存。
- 这个配置只是保存时的源 sharding，不是后续加载的永久绑定。
- 只要保存格式、模型 provider、参数命名和 Megatron loader 兼容，后续训练可以用不同 TP/PP 加载。

## 9. 实践结论

如果目标是“把 HF 模型转成 Megatron 初始化 checkpoint，然后多个实验用不同并行配置训练”，推荐理解为：

```text
AutoBridge/mbridge 负责一次性 HF -> Megatron 权重语义转换；
Megatron-Core distributed checkpointing 负责后续按当前并行配置 reshard 加载。
```

通常不需要每次改 TP/PP 都重新跑 HF 转换脚本。

需要重新转换或重点检查的情况：

- 换了模型结构、layer 数、hidden size、MoE expert 配置等架构参数
- vocab padding / tokenizer 配置不兼容
- 外部训练脚本的 provider 和转换脚本 provider 不一致
- checkpoint 不是 `torch_dist` 或兼容的 distributed checkpoint
- 要完整恢复 optimizer/RNG，并且保存时不是 fully reshardable optimizer 格式

一句话总结：

```text
AutoBridge 解决 HF <-> Megatron 的权重语义映射；
Megatron-Core + PyTorch DCP 解决 Megatron checkpoint 在不同并行配置之间的自动 reshard。
```

## 10. 本地源码参考

Megatron-Bridge:

- `src/megatron/bridge/models/conversion/auto_bridge.py`
- `src/megatron/bridge/training/model_load_save.py`
- `docs/training/checkpointing.md`

Megatron-LM:

- `/mnt/shared-storage-user/huanghaian/code/Megatron-LM/megatron/training/checkpointing.py`
- `/mnt/shared-storage-user/huanghaian/code/Megatron-LM/megatron/core/dist_checkpointing/mapping.py`
- `/mnt/shared-storage-user/huanghaian/code/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py`
- `/mnt/shared-storage-user/huanghaian/code/Megatron-LM/megatron/core/dist_checkpointing/strategies/fully_parallel.py`
- `/mnt/shared-storage-user/huanghaian/code/Megatron-LM/megatron/core/tensor_parallel/layers.py`

外部参考：

- slime `tools/convert_hf_to_torch_dist.py`

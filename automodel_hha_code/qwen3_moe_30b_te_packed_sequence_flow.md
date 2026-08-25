# Qwen3 MoE 30B TE Packed Sequence Flow

本文以 `examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence.yaml` 为例，记录一次从 recipe 启动到训练 step 的 infra 主流程。

本文刻意不展开 YAML 如何读取、`_target_` 如何解析等配置加载细节，只关注运行时流程、分布式拓扑、model infra、EP/FSDP、packed sequence 和训练 step。

## 配置核心

这个配置的核心特征：

- 模型：`Qwen/Qwen3-30B-A3B`
- recipe：`TrainFinetuneRecipeForNextTokenPrediction`
- 模型入口：`nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained`
- 自定义模型实现：`nemo_automodel.components.models.qwen3_moe.model.Qwen3MoeForCausalLM`
- backend：
  - `attn: te`
  - `linear: te`
  - `rms_norm: torch_fp32`
  - `experts: te`
  - `dispatcher: deepep`
  - `rope_fusion: false`
  - `enable_hf_state_dict_adapter: true`
- distributed：
  - `strategy: fsdp2`
  - `tp_size: 1`
  - `cp_size: 1`
  - `pp_size: 1`
  - `ep_size: 8`
  - `activation_checkpointing: true`
- sequence packing：
  - `packed_sequence_size: 1024`
  - `collate_fn: packed_sequence_thd_collater`
- checkpoint save disabled：
  - `checkpoint.enabled: false`

注意：配置中保留了 `distributed.pipeline` 块，但 `pp_size: 1`，所以运行时 pipeline block 会被判定为 inert 并忽略，不会创建 `AutoPipeline`。

## 总体调用链

高层流程：

```text
automodel qwen3_moe_30b_te_packed_sequence.yaml --nproc-per-node 8
  -> TrainFinetuneRecipeForNextTokenPrediction.setup()
    -> initialize_distributed()
    -> create_distributed_setup_from_config()
    -> build loss/checkpointer/model/optimizer/dataloader/scheduler
  -> TrainFinetuneRecipeForNextTokenPrediction.run_train_validation_loop()
    -> _run_train_optim_step()
      -> _forward_backward_step()
      -> optimizer step / logging / validation / checkpoint trigger
```

本文后面只展开 infra 相关部分。

## 1. 分布式拓扑

配置是：

```yaml
distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 8
  activation_checkpointing: true
```

假设按注释用 `--nproc-per-node 8` 跑，`world_size = 8`。

`create_distributed_setup_from_config()` 会得到：

- strategy config：`FSDP2Config`
- TP 关闭：`tp_size = 1`
- CP 关闭：`cp_size = 1`
- PP 关闭：`pp_size = 1`
- EP 开启：`ep_size = 8`
- DP 推导为 `world_size / (tp_size * cp_size * pp_size) = 8`

mesh 形态大致是：

```text
device_mesh:
  shape = (pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1)

moe_mesh:
  shape = (ep_shard=1, ep=8)
```

因此这个配置真正重要的是：

- 8 卡组成一个 expert parallel group。
- 每张卡负责一部分 experts。
- 非 expert 参数通过 FSDP2 在 DP 维度 shard。
- 因为 `ep_shard=1`，expert 参数本身不再额外 FSDP shard。

## 2. Infra 对象创建

进入 `NeMoAutoModelForCausalLM.from_pretrained()` 后，AutoModel 根据 `DistributedSetup` 调：

```python
instantiate_infrastructure(...)
```

这个配置会创建：

- `model_wrapper = FSDP2Manager(...)`
- `autopipeline = None`
- `parallelize_fn = partial(moe.parallelizer.parallelize_model, ...)`
- `qat_quantizer = None`

关键点：因为 `ep_size > 1`，这里不会走普通 dense FSDP2 parallelizer，而是走 MoE 专用 parallelizer。

普通 dense FSDP2 由 `FSDP2Manager.parallelize()` 调 dense path；本配置的 EP path 则是 `parallelize_fn(model, world_mesh, moe_mesh, ...)`，最终进入：

```text
nemo_automodel.components.moe.parallelizer.parallelize_model()
```

## 3. 模型构建

模型配置入口：

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  backend:
    attn: te
    linear: te
    rms_norm: torch_fp32
    experts: te
    dispatcher: deepep
    rope_fusion: false
    enable_hf_state_dict_adapter: true
```

AutoModel 会读取 HF config，看到 architecture 是 `Qwen3MoeForCausalLM`，然后通过 `_transformers/registry.py` 解析到仓库内自定义实现：

```text
nemo_automodel.components.models.qwen3_moe.model.Qwen3MoeForCausalLM
```

所以这里不是直接使用 Hugging Face 原生模型，而是使用 NeMo 自定义的 Qwen3 MoE 实现。

模型内部结构：

- `Qwen3MoeForCausalLM`
  - `self.model = Qwen3MoeModel(...)`
  - `self.lm_head`
  - `self.state_dict_adapter = Qwen3MoeStateDictAdapter(...)`
- `Qwen3MoeModel`
  - `embed_tokens`
  - `layers: ModuleDict[str, Block]`
  - `norm`
  - `rotary_emb`
- `Block`
  - `self_attn = Qwen3MoeAttention(...)`
  - `mlp = MoE(...)` for sparse layers
  - dense MLP for non-MoE layers
  - RMSNorms

backend 影响：

- `attn: te`：attention backend 用 Transformer Engine。
- `linear: te`：投影层用 TE Linear。
- `rms_norm: torch_fp32`：RMSNorm 走 torch fp32 路径。
- `experts: te`：MoE experts 使用 `GroupedExpertsTE`。
- `dispatcher: deepep`：token dispatch 使用 DeepEP。
- `rope_fusion: false`：禁用 fused TE RoPE。配置注释里说明 packed THD sequences 下 fused TE RoPE 会产生 NaN gradients。
- `enable_hf_state_dict_adapter: true`：启用 HF checkpoint 到 NeMo 自定义权重布局的适配。

因为这是多卡、自定义模型、无 quantization，模型通常先在 meta device 上构建，后续 shard 后再 materialize/load checkpoint。

## 4. Sharding / EP / FSDP 应用顺序

核心函数是：

```text
_transformers/infrastructure.py::apply_model_infrastructure()
```

本配置的主要顺序：

```text
apply_model_infrastructure()
  -> apply PEFT / FP8 / QAT      # 本配置没有，跳过
  -> decide checkpoint load timing
  -> _shard_ep_fsdp()
    -> parallelize_fn(...)
      -> moe.parallelizer.parallelize_model()
        -> apply_cp()            # cp_size=1，跳过
        -> apply_ep()            # ep_size=8，执行
        -> apply_ac()            # activation_checkpointing=true，执行
        -> apply_fsdp()          # dp mesh size > 1，执行
  -> materialize meta params
  -> load base model checkpoint
  -> model.to(device) / runtime compatibility fixes
```

### 4.1 EP

`apply_ep()` 会遍历所有 decoder block，找到 block 里的 `MoE`。

因为 `experts: te`，MoE experts 是 `GroupedExpertsTE`。这类 experts 不通过 DTensor 直接包 TE `GroupedLinear`，而是执行：

```python
moe_module.experts.init_token_dispatcher(ep_mesh=ep_mesh, moe_mesh=moe_mesh)
```

这一步会：

- 设置 `ep_mesh`
- 设置 `ep_rank`
- 设置 `ep_size`
- 计算本 rank 的 local experts 数量

```text
num_local_experts = n_routed_experts / ep_size
```

- 重新创建本地 `GroupedLinear`
- 初始化 DeepEP dispatcher

所以 EP 的效果是：每张卡只持有本地 experts，forward 时 token 根据 gate 结果通过 DeepEP 发到对应 expert 所在 rank。

### 4.2 Activation Checkpointing

配置：

```yaml
activation_checkpointing: true
```

MoE parallelizer 会执行：

```text
apply_ac(model, ignore_router=True)
```

默认 `ignore_router_for_ac=True`。这个名字容易误解，实际效果是：router 输出需要保存，避免 backward recompute 时重新 routing。

原因是 MoE routing 可能让每个 expert 收到的 token 数发生变化；如果 activation checkpointing 在 backward 中重新计算 router，可能导致 recompute 的 expert token shape 与 forward 不一致，进而触发 checkpoint metadata/shape 错误。

### 4.3 FSDP

`apply_fsdp()` 会做几类 wrap：

- 对 transformer block 做 FSDP2 `fully_shard`
- 对 embedding 做 FSDP wrap
- 对 lm_head 做 FSDP wrap
- 对 inner model / outer model 做 FSDP root wrap

MoE expert 参数的处理比较特殊：

- 如果 `ep_enabled=True`，expert 参数会被加入 block FSDP 的 ignored params。
- 也就是说 block FSDP 不会把 experts 当作普通 block 参数一起 shard。
- 本例 `ep_shard=1`，所以 expert 参数不会再额外按 `ep_shard` 做 FSDP。

最终参数分布可以概括为：

```text
non-expert params:
  FSDP2 shard over DP mesh

expert params:
  expert parallel over EP mesh
  each rank owns local experts
  token movement handled by DeepEP dispatcher
```

## 5. 权重加载

`from_pretrained` 需要加载 `Qwen/Qwen3-30B-A3B` 的 base checkpoint。

但本配置有：

- `ep_size = 8`
- `dp_shard_size = 8`

因此不能 load-before-shard。`_should_load_before_shard()` 会返回 false，走 post-shard load。

流程是：

```text
construct model on meta
  -> apply EP/FSDP sharding
  -> materialize parameters
  -> checkpointer.load_base_model(...)
```

由于模型启用了：

```yaml
enable_hf_state_dict_adapter: true
```

`Qwen3MoeStateDictAdapter` 会参与 HF checkpoint 到 NeMo 自定义结构的映射。

它很关键，因为当前 runtime 结构不是 HF 原始结构：

- attention/linear 可能是 TE module
- experts 是 TE grouped experts
- experts 已经按 EP 拆成本地 experts
- state dict 中 expert 权重需要在 router、gate/up/down、local expert 顺序等维度上正确映射

`checkpoint.enabled: false` 只表示训练中不保存 checkpoint；不影响 `from_pretrained` 加载 base model。

## 6. Dataset / Packed Sequence

dataset 配置：

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train
  pad_to_max_length: false

packed_sequence:
  packed_sequence_size: 1024

dataloader:
  collate_fn: nemo_automodel.components.datasets.utils.packed_sequence_thd_collater
```

recipe 构建 dataloader 时：

1. 自动根据模型名构建 tokenizer。
2. 实例化 `HellaSwag` dataset。
3. 因为 `packed_sequence_size > 0`，调用 `pack_dataset(...)`。
4. `pack_dataset()` 把多个变长样本拼成固定长度 1024 的 pack。
5. `packed_sequence_thd_collater()` 把 batch 整理成 THD 相关格式。

collater 输出大致包含：

```text
input_ids
labels
position_ids
seq_lens
seq_lens_padded
qkv_format = "thd"
```

其中：

- `seq_lens` 表示 pack 中每个原始 sequence 的真实长度。
- `seq_lens_padded` 表示包含 padding/separator 后的长度。
- `position_ids` 在 pack 内按原始 sequence 边界重置。
- `qkv_format = "thd"` 通知后续模型和 attention 走 total-token packed 格式。

### 6.1 `packed_sequence_size: 1024` 与长样本处理

这里的 `1024` 不是模型最大上下文长度，而是 recipe-side packing 的 pack 长度。它表示每个 pack 最终会被整理成 1024 token，用来把多个短样本拼在一起减少 padding 浪费。

对这个 HellaSwag 配置来说，单条样本来自：

```text
HellaSwag
  -> SFTSingleTurnPreprocessor
    -> context tokens + target tokens
    -> labels: context 部分为 -100，target 部分参与 loss
```

因为 `pad_to_max_length: false`，dataset 阶段不会把单条样本 pad 到固定长度；每条样本保留自己的变长 token 序列。随后 `pack_dataset()` 把多条短样本累积到一个 pack 中。

长样本处理逻辑在 `components/datasets/llm/packed_sequence.py::pack_dataset()`：

```python
seq_len = len(input_ids)
if drop_long_samples and seq_len > packed_sequence_size:
    logger.info(f"Dataset has sampels longer than {packed_sequence_size}, will be skipped")
    continue

if seq_len > packed_sequence_size:
    raise ValueError(
        f"Dataset sample is too long ({seq_len} > {packed_sequence_size}). "
        "Please increase `packed_sequence_size`.",
    )
```

因此：

- 单条样本 `<= 1024`：参与 packing。
- 单条样本 `> 1024` 且 `drop_long_samples=True`：整条样本被跳过。
- 单条样本 `> 1024` 且 `drop_long_samples=False`：直接报错，要求增大 `packed_sequence_size`。
- 当前 THD 分支没有做“截断长样本”或“把单条长样本切成多个 pack”的逻辑。

本 recipe 的 THD packing 分支调用是：

```python
ds = pack_dataset(
    ds,
    split=cfg_ds.split,
    packed_sequence_size=packed_sequence_size,
    max_packs=getattr(cfg_ps, "max_packs", None),
    padding_idx=getattr(tokenizer, "pad_token_id", 0),
    cp_size=cp_size,
)
```

注意这里没有透传 `drop_long_samples`。所以对当前配置来说，实际使用的是 `pack_dataset()` 默认值：

```python
drop_long_samples=True
```

也就是说，如果 HellaSwag 里存在 tokenizer 后长度超过 1024 的样本，它会被静默跳过一次日志提示，而不是被截断或切分。如果所有样本都超过 1024，最后会报错：

```text
No packs were produced: every sample was longer than packed_sequence_size=1024 ...
```

从任务性质上看，HellaSwag 的 `ctx + ending` 通常较短，1024 更像是为了性能 smoke/benchmark 选的短 pack 长度，而不是模型能力上限。要严格确认“是否所有样本都不超过 1024”，需要用同一个 tokenizer 实际扫一遍 tokenized dataset 的长度分布。

### 6.2 Dataloader + Packing 执行顺序

训练 dataloader 的构建在 `recipes/llm/train_ft.py::build_dataloader()` 中完成。对当前配置，顺序大致是：

```text
build_dataloader()
  -> _build_tokenizer()
  -> instantiate HellaSwag dataset
  -> if dataset has shuffle: ds = ds.shuffle(seed)
  -> pack_dataset(ds, packed_sequence_size=1024)
  -> StatefulDistributedSampler(packed_dataset, num_replicas=dp_world_size, rank=dp_rank, shuffle=true)
  -> StatefulDataLoader(dataset=packed_dataset, sampler=sampler, batch_size=local_batch_size, collate_fn=packed_sequence_thd_collater)
```

所以这里有两层顺序处理：

- packing 前：如果 dataset 支持 `shuffle`，会先按 seed 打乱原始 tokenized samples，再依次拼 pack。这个 shuffle 会影响哪些样本被拼进同一个 pack。
- packing 后：`pack_dataset()` 返回的是一个新的 HF `Dataset`，每条记录就是一个固定长度 pack。随后 `StatefulDistributedSampler` 再按 DP rank 切分这些 packs，并根据 `dataloader.shuffle: true` 在 pack 粒度 shuffle。

`pack_dataset()` 内部有一个 `current_pack` buffer：

```text
current_pack:
  input_ids
  labels
  position_ids
  seq_lens
```

它逐条读取 tokenized sample：

1. 取出 `input_ids` 和 `labels`。
2. 如果有 `loss_mask`，把不参与 loss 的 label 改成 `-100`。
3. 如果单条样本超过 `packed_sequence_size`，按 `drop_long_samples` 决定跳过或报错。
4. 把整条样本追加到 `current_pack`。
5. 记录这条样本长度到 `seq_lens`。
6. 如果追加后超过 1024，就在上一条样本边界处切开：
   - 前半部分 pad 到 1024，作为一个 pack 加入 `packs`。
   - 当前样本保留到新的 `current_pack`，作为下一个 pack 的开头。

因此 pack 不会在一条样本中间切分。它只会在样本边界切分。

最后一个不满 1024 的 `current_pack` 会被 pad 到 1024 后加入 `packs`。

以短样本长度为例：

```text
samples token lengths: [120, 200, 500, 300, 100]
packed_sequence_size: 1024

pack 0:
  120 + 200 + 500 = 820
  下一条 300 会让总长变成 1120 > 1024
  所以 pack 0 = [120, 200, 500] + pad 到 1024

pack 1:
  300 + 100 = 400
  dataset 结束后 pad 到 1024
```

对当前训练配置：

```yaml
step_scheduler:
  local_batch_size: 4

dataloader:
  collate_fn: packed_sequence_thd_collater
```

每个 DP rank 的一个 dataloader batch 包含 4 个 packs。collater 会把它们 stack 成：

```text
input_ids:     [local_batch_size, 1024]
labels:        [local_batch_size, 1024]
position_ids:  [local_batch_size, 1024]
seq_lens:      [local_batch_size, max_num_sequences_in_pack]
seq_lens_padded: [local_batch_size, max_num_sequences_in_pack]
qkv_format:    "thd"
```

这些 batch 字段随后进入 `_forward_backward_step()`，再被 `make_cp_batch_and_ctx()` 转成 TE THD attention 需要的 packed metadata。

validation dataloader 也会走类似 packing 逻辑。因为当前 `validation_dataloader.collate_fn` 也是 `packed_sequence_thd_collater`，`build_validation_dataloader()` 会判断需要把 validation dataset 按 training 的 packed sequence 设置进行 pack。

## 7. 训练 Step

非 PP 路径下，核心在：

```text
TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step()
```

流程：

```text
batch to cuda
  -> detect THD collater
  -> make_cp_batch_and_ctx(..., use_te=True)
  -> labels = batch.pop("labels")
  -> model(**batch)
  -> calculate_loss(MaskedCrossEntropy, ...)
  -> backward
```

虽然 `cp_size=1`，这里仍会因为 THD + TE attention 处理 packed metadata。`make_cp_batch_and_ctx()` 会基于 `seq_lens/seq_lens_padded` 准备 TE attention 需要的 packed sequence 信息，例如 `cu_seqlens`。

进入模型 forward：

```text
Qwen3MoeForCausalLM.forward()
  -> if qkv_format == "thd": squeeze_input_for_thd(...)
  -> Qwen3MoeModel.forward(...)
    -> each Block.forward(...)
      -> Qwen3MoeAttention.forward(...)
      -> MoE.forward(...)
  -> compute_lm_head_logits(...)
```

attention 侧：

- 输入从 `[batch, sequence, hidden]` 变为 packed total-token 形式。
- `Qwen3MoeAttention` 根据 `qkv_format == "thd"` 走 THD reshape。
- RoPE 使用 `position_ids` 和 `cu_seqlens` 保持 pack 内 sequence 边界。
- TE attention 使用 packed metadata，避免不同原始样本之间互相 attend。

MoE 侧：

1. gate 对 token 做 top-k expert 选择。
2. DeepEP dispatcher 按 expert assignment 把 token 发送到 expert 所在 rank。
3. 本地 `GroupedExpertsTE` 执行 grouped GEMM。
4. dispatcher combine 回原 token 顺序。
5. MoE 输出回到 block residual 路径。

loss 侧：

- loss 是 `MaskedCrossEntropy`。
- 只对 label 不等于 `-100` 的位置计 loss。
- packed padding、prompt 部分等通过 `-100` mask 掉。

backward 侧：

- FSDP2 负责非 expert 参数的 shard、all-gather、reduce-scatter、reshard。
- DeepEP/EP 路径负责 expert token dispatch/combine 对应的反向通信。
- activation checkpointing 会重算 block，但 router 输出按策略保存，避免重算 routing 造成 shape mismatch。

## 8. Batch / Step Math

配置：

```yaml
step_scheduler:
  global_batch_size: 32
  local_batch_size: 4
```

在 8 卡默认运行时：

```text
dp_size = 8
grad_acc_steps = global_batch_size / (local_batch_size * dp_size)
               = 32 / (4 * 8)
               = 1
```

所以每个 optimizer step 只需要 1 个 microbatch。

如果 world size 变化：

- `dp_size` 会重新根据 `world_size / (tp * cp * pp)` 推导。
- `ep_size=8` 仍然要求拓扑能构造出合法 MoE mesh。
- `n_routed_experts` 必须能被 `ep_size` 整除。

## 9. 一句话总结

这个配置可以理解成：

```text
Qwen3 MoE custom model
  + TE attention / TE linear / TE grouped experts
  + DeepEP token dispatcher
  + EP=8 for expert distribution
  + FSDP2 for non-expert parameter sharding
  + THD packed sequence training at sequence length 1024
  + no TP / no CP / no PP
```

真正的 infra 主轴是：

```text
DistributedSetup
  -> MeshContext(device_mesh + moe_mesh)
  -> instantiate_infrastructure()
  -> MoE parallelize_model()
  -> apply_ep()
  -> apply_ac()
  -> apply_fsdp()
  -> post-shard checkpoint load
  -> THD packed train step
```

## 10. 相关复杂对话数据入口

前面的 HellaSwag 路径是简单 single-turn prompt-completion。仓库里也有更复杂的多轮对话和 tool-calling 路径，后续可以单独展开。

### 10.1 普通多轮 LLM ChatDataset

代表配置：

```text
examples/llm_finetune/qwen/qwen2_5_7b_instruct_chat.yaml
examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml
examples/llm_finetune/glm/glm_5.2_tulu3_4k_tilelang_100k.yaml
examples/llm_finetune/deepseek_v4/deepseek_v4_flash_cp_tulu3.yaml
```

核心 dataset：

```text
nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
```

`ChatDataset` 支持：

- OpenAI `messages` 格式：`[{role, content}, ...]`
- ShareGPT `conversations` 格式：`[{from, value}, ...]`
- optional `tools`
- optional `reasoning_content`
- tokenizer chat template
- assistant-only loss mask
- `mask_history`：只训练最后一个 assistant turn

它和 HellaSwag 的区别是：token id 和 labels 不是简单 `ctx + target` 拼接，而是通过 tokenizer 的 `apply_chat_template()` 渲染整段多轮消息，再根据 assistant span 生成 loss mask。

### 10.2 ChatDataset 的 token id / labels 生成

这条路径的核心不是 `ctx + target`，而是：

```text
messages
  -> normalize 成 OpenAI messages
  -> tokenizer.apply_chat_template() 渲染整段多轮对话
  -> 得到完整 token stream
  -> 构造 assistant token mask
  -> shift 成 next-token-prediction 的 input_ids / labels
```

入口在 `nemo_automodel/components/datasets/llm/chat_dataset.py` 的 `ChatDataset.__getitem__()`：

1. 从样本里取 `messages`。
2. 如果是 ShareGPT 的 `conversations`，先转成 OpenAI 风格的 `{role, content}`。
3. `_normalize_messages()` 做 role/content 规整。
4. 如果有 `tools`，JSON string 会先 parse 成 list。
5. 调 `format_chat_template(...)`，把 tokenizer、messages、tools、`seq_length`、padding/truncation、`mask_reasoning_content` 等传进去。

真正的模板渲染和 mask 构造在 `nemo_automodel/components/datasets/llm/formatting_utils.py::format_chat_template()`：

```python
tokenized_chat = tokenizer.apply_chat_template(
    formatted_text,
    tools=tools,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=template_has_generation_kwd,
    padding=padding,
    truncation=truncation,
    max_length=seq_length,
)
```

这里 `formatted_text` 是完整多轮消息，不是单独 prompt 或 answer。也就是说 tokenizer 看到的是完整 conversation，由 chat template 决定 system/user/assistant/tool 这些 turn 怎么插特殊 token、role tag、分隔符、EOS 等。

assistant mask 有两条路径：

1. 如果 tokenizer 的 chat template 里有 HuggingFace 的 `{% generation %}` block，`apply_chat_template(..., return_assistant_tokens_mask=True)` 会直接返回 `assistant_masks`。这是最标准的路径，mask 由 tokenizer/template 自己给出。
2. 如果 template 没有 generation block，但 `answer_only_loss_mask=True`，AutoModel 用 fallback：`_build_multiturn_assistant_mask()`。它对每个 assistant turn 做 prefix tokenization：

```text
start = tokenized_length(messages[:idx])
end   = tokenized_length(messages[:idx + 1])
mask[start:end] = 1
```

所以 fallback 的本质是：用同一个 `apply_chat_template()` 分别渲染“assistant turn 前的前缀”和“包含这个 assistant turn 的前缀”，二者 token 长度差就是这个 assistant turn 在最终 token stream 里的 span。所有 assistant turn 都会标成 1，system/user/tool 等非 assistant token 是 0。

之后还会叠加几个修正：

- 用 tokenizer 返回的 `attention_mask` 把 padding 位置 mask 成 0。
- 如果 `mask_history=True`，`ChatDataset.__getitem__()` 最后只保留最后一段 supervised assistant run，历史 assistant turn 的 labels 会被改成 `-100`。
- 如果 `mask_reasoning_content=True`，会再构造 reasoning span mask，把 assistant 的 `reasoning_content` 部分从 loss 里扣掉。
- 如果是 left padding，fallback/reasoning mask 会根据 tokenizer attention mask 做位置平移。

最后 `_package_tokenized_example()` 做标准 next-token-prediction 对齐。假设完整 chat template token stream 是：

```text
x0, x1, x2, ..., xN
```

它先复制一份完整 tokens 当 labels 候选：

```python
labels = input_ids.copy()
```

然后按 assistant mask 把非 assistant token 改成 `-100`：

```python
labels[:] = [label if bool(m) else -100 for label, m in zip(labels, assistant_masks)]
```

再做 shift：

```python
input_ids = input_ids[:-1]
labels = labels[1:]
```

所以最终训练样本是：

```text
input_ids: x0, x1, ..., x(N-1)
labels:    x1, x2, ..., xN   # 但只有 assistant token 保留 id，其余是 -100
```

这个顺序很关键：assistant mask 是先作用在完整 token stream 上，然后 labels 再左移一位。这样如果 `xj` 是 assistant token，它会作为位置 `j-1` 的 label，让模型在看到 `x0..x(j-1)` 后预测 assistant token `xj`。这就是普通 causal LM SFT 里 answer-only loss 的正确对齐方式。

默认行为是训练所有 assistant turn；只有打开 `mask_history` 才会只训练最后一个 assistant turn。

### 10.3 assistant loss mask 的三条路径

这里容易混淆的一点是：判断条件不是“模型架构支不支持”，而是当前生效的 `tokenizer.chat_template` 字符串里有没有 `{% generation %}` block。AutoModel 最终都要得到一个和完整 `input_ids` 等长的 assistant mask，再把非 assistant label 置成 `-100`；差异只在 mask 怎么来。

#### 路径 1：HF tokenizer 自带 `{% generation %}`

如果 HF checkpoint 自带的 `tokenizer.chat_template` 已经把 assistant 输出包在：

```jinja
{% generation %}
{{ message["content"] }}
{% endgeneration %}
```

那么 AutoModel 会直接调用：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)
```

Transformers 渲染 Jinja 时记录 generation block 在 rendered text 里的字符区间，再通过 fast tokenizer 的 `char_to_token()` 映射成 token 区间，返回 `assistant_masks`。

优点：

- 最直接，训练使用模型官方模板，不需要额外维护模板文件。
- 一次 render + 一次 tokenize 即可得到 token id 和 assistant mask。
- mask 边界由模板显式标注，比启发式前缀长度更精确。

缺点：

- 依赖上游 tokenizer 模板已经写了 `{% generation %}`，现实里很多模型没有。
- 基本依赖 fast tokenizer 的 `char_to_token()`；slow/Python tokenizer 可能在 mask 映射阶段失败。
- 官方模板如果只面向推理格式，不一定把训练期想监督的 EOS、tool call、reasoning 内容放进合适的 generation span。

#### 路径 2：recipe 覆盖带 `{% generation %}` 的自定义 Jinja

如果官方模板没有 generation block，AutoModel 的一些 recipe 会显式覆盖 `dataset.chat_template`，例如：

```yaml
dataset:
  chat_template: examples/convergence/tulu3/models/qwen3-moe-30b/chat_template.jinja
```

或者直接在 YAML 里写一段模板：

```yaml
chat_template: |-
  {% for message in messages %}
  {% if message['role'] == 'assistant' %}
  {% generation %}{{ message['content'] }}{% endgeneration %}
  {% endif %}
  {% endfor %}
```

覆盖之后，当前 active template 就有 `{% generation %}`，所以后续和路径 1 一样，走 HF 的 `return_assistant_tokens_mask=True`。

优点：

- 可以在保留模型聊天格式的基础上，精确控制哪些 token 参与 loss。
- 对 tool-calling、reasoning、thinking 模型更可控：可以选择是否把 `<think>`、tool call、`<|im_end|>` 等纳入 loss。
- 避免 fallback 对多轮长对话反复 tokenize prefix 的开销。

缺点：

- 需要维护自定义模板；一旦上游模型更新 chat format，本地模板可能滞后。
- 模板写错会直接改变训练分布，例如漏掉 role header、EOS、tool schema，或者把不该训练的上下文包进 generation。
- 仍然依赖 fast tokenizer / `char_to_token()` 这条 HF mask 生成链路。AutoModel 对部分 TikToken tokenizer 会尝试转 fast backend，但不是所有 slow tokenizer 都能自动补齐。

#### 路径 3：AutoModel prefix-length fallback

如果当前 template 没有 `{% generation %}`，但 `answer_only_loss_mask=True`，AutoModel 会走 `_build_multiturn_assistant_mask()`。它不让 HF 返回 assistant mask，而是对每个 assistant turn 算前缀 token 长度：

```text
start = tokenized_length(messages[:idx])
end   = tokenized_length(messages[:idx + 1])
mask[start:end] = 1
```

也就是用同一个 `apply_chat_template()` 分别渲染“assistant turn 前”和“包含 assistant turn 后”的 conversation prefix，二者长度差就是该 assistant turn 的 token span。

优点：

- 不需要 `{% generation %}`，所以能兼容大量只提供普通 chat template 的 HF 模型。
- 不依赖 `char_to_token()`，slow tokenizer 也能工作。
- 对普通多轮 SFT 可以自动得到 answer-only loss，不要求用户额外写模板。

缺点：

- 长多轮样本会慢：每个 assistant turn 都可能触发额外的 prefix `apply_chat_template()` + tokenize。代码里有 `full_length` 和 `length_cache` 优化，但本质仍是 O(turns) 次模板/tokenizer 调用。
- 它假设 chat template 对 conversation prefix 的渲染和完整 conversation 是前缀一致的。大多数标准模板满足，但复杂模板如果依赖全局上下文、最后一轮、tool 状态等，边界可能不如 generation block 精确。
- 只能粗粒度地把整个 assistant turn 增量标成 supervised span；如果想精细地区分 reasoning、final answer、tool call、EOS，还是自定义 `{% generation %}` 更可靠。

实践上可以按这个优先级理解：

```text
fast tokenizer + 正确 generation block：首选
需要精细控制训练 span：recipe 覆盖自定义 chat_template.jinja
普通模板 / slow tokenizer / 兼容性优先：使用 prefix-length fallback
```

### 10.4 Qwen3 MoE Chat THD 示例

`examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml` 和本文主配置非常接近，但数据换成了 `ChatDataset`：

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: allenai/tulu-3-sft-mixture
  split: train
  shuffle_seed: 42
  truncation: true
  seq_length: 1024
  padding: max_length

packed_sequence:
  packed_sequence_size: 0

dataloader:
  collate_fn: nemo_automodel.components.datasets.utils.packed_sequence_thd_collater
```

这里没有 sequence packing。`packed_sequence_thd_collater` 会为非 packed 的 ChatDataset 输出合成 THD metadata，使 TE attention 仍然能走 THD 相关路径。

### 10.5 Agent / Function Calling 数据

代表配置：

```text
examples/llm_finetune/agent/qwen2_5_3b_function_calling.yaml
examples/llm_finetune/agent/qwen2_5_3b_function_calling_lora.yaml
```

核心 dataset：

```text
nemo_automodel.components.datasets.llm.agent_chat.make_agent_chat_dataset
```

它比普通 `ChatDataset` 更复杂，支持 function-calling traces：

- tool definitions
- user turns
- assistant tool calls
- tool responses
- final assistant answer
- ShareGPT function_call / observation 格式
- Swift/chatml `messages` 格式
- 可选 `truncate_history`，当对话超过 `seq_length` 时丢弃较早历史，而不是 token-level 截断 final answer

这个路径更接近真实 agent SFT。它最终也会调用 chat template 渲染，并构造 answer-only loss mask，但在进入模板前会先把 tool_call / tool_response 规整成 OpenAI chat-completions 风格。

# MoE LoRA Grouped Experts Analysis

本文分析 NeMo AutoModel 里 MoE 模型的 LoRA 微调是怎么实现的，重点解释 routed expert 层为什么需要特殊处理，以及 `grouped_gemm` / `torch._grouped_mm` 路径下 LoRA 增量是如何参与计算的。

对应代码主要在：

- `nemo_automodel/components/_peft/lora.py`
- `nemo_automodel/components/_peft/lora_experts.py`
- `nemo_automodel/components/moe/experts.py`
- `nemo_automodel/_transformers/infrastructure.py`
- `nemo_automodel/components/moe/state_dict_mixin.py`
- `nemo_automodel/components/models/qwen3_moe/state_dict_adapter.py`

## 结论概览

这个框架对 MoE expert 的 LoRA 不是简单地把普通 `nn.Linear` 替换成 `LinearLoRA`。

原因是 AutoModel 的 routed experts 通常不是多个独立的 `nn.Linear` module，而是合并成 grouped expert 参数：

```text
gate_and_up_projs: [num_experts, expert_dim, up_proj_dim]
down_projs:        [num_experts, moe_inter_dim, expert_dim]
```

其中 gated SwiGLU/GEGLU 类 expert 的 `up_proj_dim = 2 * moe_inter_dim`，非 gated ReLU2 类 expert 的 `up_proj_dim = moe_inter_dim`。

因此 routed expert LoRA 需要额外的 wrapper：

- `GroupedExpertsLoRA`
- `GroupedExpertsDeepEPLoRA`

这两个 wrapper 会给每个 expert 分别挂 3D LoRA 参数，并在 expert grouped GEMM 计算里显式加上低秩增量。

当前限制：

- `GroupedExperts` 支持。
- `GroupedExpertsDeepEP` 支持。
- `GroupedExpertsTE` 不支持 MoE expert LoRA，会在 patch 时抛 `NotImplementedError`。
- MoE expert 不支持 DoRA。
- 普通 dense/attention `nn.Linear` 或 TE `Linear` 仍然走 `LinearLoRA` 路径。

## 1. PEFT 注入时机

LoRA 注入入口在 `nemo_automodel/_transformers/infrastructure.py`：

```python
def _apply_peft_and_lower_precision(
    model, tp_size, autopipeline, peft_config, quantization_config, fp8_config, qat_quantizer
):
    if peft_config is not None:
        if tp_size > 1:
            peft_config.use_triton = False
        if autopipeline is not None:
            peft_config.use_triton = False
        apply_lora_to_linear_modules(model, peft_config, quantization_config=quantization_config, skip_freeze=True)
```

关键点：

- LoRA module surgery 在模型基础设施流程中完成。
- `skip_freeze=True`，所以这里先只 patch module，不立即全局冻结。
- 后面 checkpoint load、parallelization 完成后，会再次全局冻结非 LoRA 参数：

```python
if peft_config is not None:
    for name, param in mp.named_parameters():
        if "lora_" not in name and param.requires_grad:
            param.requires_grad_(False)
```

这个顺序很重要：FSDP/EP/PP 等并行改造可能创建或替换参数，最终 freeze 放在并行化和加载之后，避免漏冻结新产生的 base 参数。

## 2. target_modules 如何命中 MoE experts

LoRA patch 主函数是：

```python
apply_lora_to_linear_modules(model, peft_config, ...)
```

它遍历 `model.named_modules()`，分两类处理：

```python
if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE)):
    # MoE expert path
else:
    # ordinary Linear path
```

匹配逻辑来自 `ModuleMatcher`：

- `target_modules` 非空时，按完整 module name 或 wildcard 匹配。
- `target_modules` 为空，且 `match_all_linear=False` 时，默认变成 `*_proj`。
- `match_all_linear=True` 只按普通 Linear 类型匹配，不会因为它叫 match all linear 就覆盖 grouped expert wrapper。

这意味着：

```yaml
peft:
  target_modules: '*_proj'
```

通常只会命中 attention/dense/shared expert 里的普通 projection，不会命中 routed `GroupedExperts` 本体。

要 patch routed grouped experts，示例通常写：

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ["*"]
  dim: 32
  alpha: 32
  dropout: 0.0
  moe_rank_scaling: true
```

例如：

- `examples/llm_finetune/qwen/qwen3_moe_30b_lora.yaml`
- `examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence_lora.yaml`
- `examples/llm_benchmark/deepseek/dsv32_lora.yaml`

但是如果 backend 使用 `experts: te`，并且 `target_modules: ["*"]` 命中了 `GroupedExpertsTE`，当前代码会直接报错：

```python
if isinstance(orig_module, GroupedExpertsTE):
    raise NotImplementedError("LoRA is not supported for Transformer Engine (TE) expert modules.")
```

所以如果只想训 attention/dense Linear，应该用具体 projection pattern 排除 routed experts；如果想训 routed experts，应使用 `experts: torch_mm` 或 `experts: gmm` 这类会构造 `GroupedExperts` / `GroupedExpertsDeepEP` 的 backend。

## 3. 普通 Linear LoRA 路径

普通线性层走 `LinearLoRA` / `TritonLinearLoRA`。

核心计算是标准 LoRA：

```text
y = x W^T + bias + scale * ((x A^T) B^T)
scale = alpha / rank
```

实现上会给被 patch 的 Linear 挂：

```text
lora_A: [rank, in_features]
lora_B: [out_features, rank]
```

并把 base weight/bias 冻结。

普通 Linear 路径支持：

- `nn.Linear`
- `transformer_engine.pytorch.Linear`
- QLoRA 情况下通过 `quant_state` 保留原 forward
- memory-efficient LoRA autograd
- 可选 Triton LoRA kernel

但这条路径不能覆盖 grouped routed experts，因为 grouped routed experts 没有 `gate_proj/up_proj/down_proj` 这些普通子 Linear module。

## 4. MoE expert LoRA 参数形状

`GroupedExpertsLoRA` 初始化时会复制原 grouped expert base 参数：

```python
self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
self.down_projs.data.copy_(orig_module.down_projs.data)
```

然后冻结 base 参数：

```python
obj.gate_and_up_projs.requires_grad = False
obj.down_projs.requires_grad = False
```

并创建 per-expert LoRA 参数：

```text
lora_gate_and_up_A: [E, expert_dim, r]
lora_gate_and_up_B: [E, r, up_proj_dim]
lora_down_A:        [E, moe_inter_dim, r]
lora_down_B:        [E, r, expert_dim]
```

其中：

- `E = num_routed_experts`
- `r = lora_dim`
- `expert_dim = config.expert_dim`
- `moe_inter_dim = config.moe_inter_dim`
- `up_proj_dim = 2 * moe_inter_dim` for gated experts
- `up_proj_dim = moe_inter_dim` for non-gated experts

初始化策略：

- `lora_gate_and_up_A` 和 `lora_down_A` 用 Xavier 或 Kaiming。
- `lora_gate_and_up_B` 和 `lora_down_B` 初始化为 0。

因此刚注入 LoRA 后，模型输出与 base expert 输出保持一致，因为 `B = 0` 时 LoRA delta 为 0。

## 5. MoE expert 前向的基础计算

MoE router 给每个 token 选出 top-k experts：

```text
indices: [tokens, topk]
weights: [tokens, topk]
token_mask: [tokens]
```

对于每个被路由到 expert `e` 的 token，expert 计算大致是：

```text
h1_e = x_e @ W_gate_up_e
a_e  = activation(h1_e, route_weight)
y_e  = a_e @ W_down_e
```

多个 top-k expert 的输出会通过 `scatter_add_` 累加回原 token 位置。

注意 routing weight 的位置：当前 grouped expert 路径把 routing weight 放在 activation 和 down projection 之间，也就是：

```text
a_e = expert_activation_grouped(h1_e, route_weight)
```

这和代码里的注释一致：

```python
# Weighted activation (routing weight applied BETWEEN up and down projections)
```

如果 expert 有 bias：

- gate/up bias 在 activation 前加。
- down bias 会乘 routing weight 后加。

## 6. loop 路径下 LoRA 怎么算

非 grouped-mm loop 路径按 local expert 逐个处理。

base gate/up：

```python
gate_and_up_out = x_idx @ gate_and_up_projs[local_idx]
```

LoRA gate/up delta：

```python
gate_and_up_out = gate_and_up_out + (
    x_idx @ lora_gate_and_up_A[local_idx] @ lora_gate_and_up_B[local_idx]
) * scale
```

activation：

```python
activated = self.expert_activation_grouped(gate_and_up_out, w)
```

base down：

```python
expert_out = activated @ down_projs[local_idx]
```

LoRA down delta：

```python
expert_out = expert_out + (
    activated @ lora_down_A[local_idx] @ lora_down_B[local_idx]
) * scale
```

最终：

```python
y.scatter_add_(dim=0, index=idx_b, src=expert_out.float())
```

数学上就是每个 expert 独立有两组低秩增量：

```text
W_gate_up_e' = W_gate_up_e + scale * A_gate_up_e @ B_gate_up_e
W_down_e'    = W_down_e    + scale * A_down_e    @ B_down_e
```

但实现不会显式 materialize `W'`，而是在 forward 里用两次 matmul 计算 LoRA branch。

## 7. torch._grouped_mm 路径下 LoRA 怎么算

`GroupedExperts` 的 `experts: torch_mm` 路径使用 `torch._grouped_mm`。

这一节是最关键的路径。可以先把它理解成：

```text
同一批已经按 expert 排好序的 token-expert rows：

  base branch:  x_e @ W_e
  LoRA branch:  x_e @ A_e @ B_e

然后同一个 row 上逐元素相加：

  output_e = x_e @ W_e + scale * (x_e @ A_e @ B_e)
```

这里的相加不是 top-k experts 的输出聚合，只是同一个 expert projection 上的 base branch 和 LoRA branch 相加。真正把一个 token 的多个 top-k expert 输出聚合回 token 位置，是 down projection 之后的 `scatter_add_`。

### 7.1 token 先按 expert 排序

`_permute_tokens_for_grouped_mm()` 做四件事：

1. 把 masked token 的 expert index 置为 -1。
2. 展开 `[tokens, topk]` 到 `[tokens * topk]`。
3. 只保留当前 rank local experts 范围内的 token-expert pair。
4. 按 local expert id 稳定排序，得到：

```text
sorted_token_ids: [num_routed_pairs_local]
sorted_weights:   [num_routed_pairs_local]
tokens_per_expert:[num_local_experts]
offs:             [num_local_experts]
```

`offs = cumsum(tokens_per_expert).to(int32)`，用于告诉 `torch._grouped_mm` 每个 expert 的 token segment 边界。

### 7.2 grouped_mm 的输入形态

对 gate/up projection：

```text
permuted_x:        [sum(tokens_per_expert), expert_dim]
gate_and_up_projs: [num_local_experts, expert_dim, up_proj_dim]
offs:              [num_local_experts]
```

`torch._grouped_mm(permuted_x, gate_and_up_projs, offs=offs)` 的语义可以理解为：

```text
for expert e:
  out[start_e:end_e] = permuted_x[start_e:end_e] @ gate_and_up_projs[e]
```

其中 `start_e/end_e` 由 `offs` 推出来。

### 7.3 LoRA gate/up branch

代码：

```python
output1 = torch._grouped_mm(permuted_x, gate_and_up_projs, offs=offs)
lora_out1_A = torch._grouped_mm(permuted_x, lora_gate_and_up_A, offs=offs)
lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
output1 = output1 + lora_out1 * self.scale
```

shape 对应：

```text
E_local             = 当前 rank 上的 local experts 数
H                   = expert_dim，一般等于 hidden size
I                   = moe_inter_dim
r                   = LoRA rank，可能已经经过 moe_rank_scaling
N                   = sum(tokens_per_expert)，即当前 rank local experts 收到的 token-expert rows 总数

permuted_x:          [N, H]
gate_and_up_projs:   [E_local, H, 2I]
output1 base:        [N, 2I]

lora_gate_and_up_A:  [E_local, H, r]
lora_out1_A:         [N, r]
lora_gate_and_up_B:  [E_local, r, 2I]
lora_out1:           [N, 2I]

output1 final:       [N, 2I]
```

等价伪代码是：

```python
for e in range(E_local):
    start = 0 if e == 0 else offs[e - 1]
    end = offs[e]

    x_e = permuted_x[start:end]                 # [tokens_for_e, H]
    base_e = x_e @ gate_and_up_projs[e]         # [tokens_for_e, 2I]
    lora_e = (x_e @ lora_gate_and_up_A[e]) @ lora_gate_and_up_B[e]
                                                  # [tokens_for_e, 2I]

    output1[start:end] = base_e + lora_e * scale
```

实际代码不是 Python for loop，而是用三次 grouped GEMM 批量完成：

```text
1. base gate/up:  [N, H] @ grouped [E_local, H, 2I] -> [N, 2I]
2. LoRA A:        [N, H] @ grouped [E_local, H, r]  -> [N, r]
3. LoRA B:        [N, r] @ grouped [E_local, r, 2I] -> [N, 2I]
```

如果是非 gated expert，例如 ReLU2，`up_proj_dim = I` 而不是 `2I`，于是对应变成：

```text
gate_and_up_projs:  [E_local, H, I]
lora_gate_and_up_B: [E_local, r, I]
output1:            [N, I]
```

### 7.4 activation

```python
output1 = self.expert_activation_grouped(output1, permuted_probs)
```

shape：

```text
permuted_probs: [N, 1]
output1 after activation: [N, I]
```

### 7.5 LoRA down branch

代码：

```python
output2 = torch._grouped_mm(output1, down_projs, offs=offs)
lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
output2 = output2 + lora_out2 * self.scale
```

shape 对应：

```text
output1:       [N, I]
lora_down_A:   [E_local, I, r]
lora_out2_A:   [N, r]
lora_down_B:   [E_local, r, H]
lora_out2:     [N, H]
output2:       [N, H]
```

最后 scatter 回原 token：

```python
scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
y.scatter_add_(0, scatter_ids, output2.float())
```

因为 top-k routing 下同一个 token 可能对应多个 expert，`scatter_add_` 会把多个 expert output 累加。

## 8. LoRA rank padding

`torch._grouped_mm` 路径会对 LoRA rank 做 padding：

```python
_pad_lora_rank_for_grouped_mm(lora_A, lora_B)
```

逻辑：

```python
alignment = max(1, 16 // lora_A.element_size())
padding = (-rank) % alignment
```

比如 bf16 element size 是 2 bytes，`alignment = 8`，rank 需要 pad 到 8 的倍数。

padding 方式：

```python
F.pad(lora_A, (0, padding))
F.pad(lora_B, (0, 0, 0, padding))
```

也就是：

- `A` 在 rank 输出维补 0。
- `B` 在 rank 输入维补 0。

这样 `A @ B` 的新增 padded rank 对结果无贡献，只是满足 grouped-mm stride/alignment 要求。

## 9. DeepEP 路径下 LoRA 怎么算

`GroupedExpertsDeepEPLoRA` 处理 DeepEP dispatcher 的 expert 计算。

它首先调用 dispatcher 做 token permutation：

```python
permuted_local_hidden_states, tokens_per_expert, permuted_probs =
    self.token_dispatcher.token_permutation2(...)
```

然后分两种 GEMM backend：

### 9.1 DeepEP + torch_mm

如果 `self.use_torch_mm`：

```python
offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)
output1 = torch._grouped_mm(permuted_local_hidden_states, gate_and_up_projs, offs=offs)
lora_out1_A = torch._grouped_mm(permuted_local_hidden_states, lora_gate_and_up_A, offs=offs)
lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
```

后续 down projection 同理。

### 9.2 DeepEP + grouped_gemm.ops.gmm

如果不是 `torch_mm`，则使用外部 `grouped_gemm`：

```python
output1 = ops.gmm(
    permuted_local_hidden_states,
    gate_and_up_projs,
    tokens_per_expert,
    trans_b=False,
)
lora_out1_A = ops.gmm(
    permuted_local_hidden_states,
    lora_gate_and_up_A,
    tokens_per_expert,
    trans_b=False,
)
lora_out1 = ops.gmm(lora_out1_A, lora_gate_and_up_B, tokens_per_expert, trans_b=False)
```

这里 `tokens_per_expert` 扮演 `offs` 类似的分段描述角色，只是 API 不同。

最后：

```python
y = self.token_dispatcher.token_unpermutation(output2)
```

DeepEP 路径的核心差异不是 LoRA 数学不同，而是 token dispatch/un-dispatch 和 grouped GEMM backend 不同。

## 10. EP 下 local experts 与参数分片

MoE expert parallelism 下，每个 rank 只负责一部分 experts：

```text
n_local_experts = n_routed_experts // ep_size
experts_start_idx = ep_rank * n_local_experts
experts_end_idx = experts_start_idx + n_local_experts
```

`GroupedExpertsLoRA.forward()` 会根据 `gate_and_up_projs` 是否为 `DTensor` 判断 EP mesh：

```python
if isinstance(self.gate_and_up_projs, DTensor):
    ep_mesh = self.gate_and_up_projs.device_mesh
    ep_size = ep_mesh.size()
    ep_rank = ep_mesh.get_local_rank()
else:
    ep_size = 1
    ep_rank = 0
```

然后只计算当前 local expert 范围：

```text
[experts_start_idx, experts_end_idx)
```

普通 `GroupedExperts` 和 `GroupedExpertsLoRA` 的 EP gather/reduce 方式，与 DeepEP dispatcher 路径不同。DeepEP 依赖 token dispatcher；非 DeepEP wrapper 自己处理 EP 输入聚合和 partial 输出。

### 10.1 LoRA expert 参数什么时候变成 local shard

从代码顺序看，MoE LoRA 参数不是一开始就按 local rank shape 构造的。`apply_lora_to_linear_modules()` 在 EP parallelize 之前执行，`GroupedExpertsLoRA` / `GroupedExpertsDeepEPLoRA` 初始化时创建的是逻辑全量 expert 维：

```text
lora_gate_and_up_A: [num_routed_experts, expert_dim, r]
lora_gate_and_up_B: [num_routed_experts, r, up_proj_dim]
lora_down_A:        [num_routed_experts, moe_inter_dim, r]
lora_down_B:        [num_routed_experts, r, expert_dim]
```

随后 MoE parallelizer 的 `ExpertParallel` 对 expert module 的直接参数统一执行：

```python
dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
```

也就是沿 expert 维 `dim=0` 做 EP shard。因为 LoRA 参数是挂在 `GroupedExpertsLoRA` module 上的直接 `nn.Parameter`，所以它们会和 base expert 参数 `gate_and_up_projs` / `down_projs` 一样被 `Shard(0)`。

因此 EP 后每个 rank 上的 DTensor 仍然有逻辑 global shape，但本地 shard 是：

```text
E_local = num_routed_experts // ep_size

lora_gate_and_up_A.to_local(): [E_local, expert_dim, r]
lora_gate_and_up_B.to_local(): [E_local, r, up_proj_dim]
lora_down_A.to_local():        [E_local, moe_inter_dim, r]
lora_down_B.to_local():        [E_local, r, expert_dim]
```

forward 里传给 grouped GEMM 的正是 `to_local()` 后的 local shard：

```python
lora_gate_and_up_A = _to_grouped_mm_operand(self.lora_gate_and_up_A, compute_dtype)
```

这里 `_to_grouped_mm_operand()` 会先对 DTensor 调 `to_local()`，再 cast 到 compute dtype 并做 contiguous。

需要区分“逻辑初始化”和“真实内存分配”：

- 普通 eager 构造时，LoRA wrapper 会先创建全量形状参数，然后 EP parallelize 把它变成 `Shard(0)` 的 DTensor。
- meta/init-empty 构造路径下，全量形状可能只是 meta tensor，后续 sharding 和 checkpoint load 才真正物化本地 shard，所以不一定真的在每张卡上分配过完整 expert LoRA 参数。
- checkpoint load 路径也会根据 EP mesh 只加载或保留当前 rank 负责的 expert range；训练 forward 始终只对 local expert shard 做 grouped GEMM。

## 11. moe_rank_scaling 的意义

`PeftConfig` 有一个 MoE 专用字段：

```python
moe_rank_scaling: bool = False
```

如果开启：

```python
n_act = module.config.n_activated_experts
moe_dim = peft_config.dim // n_act
```

也就是 routed expert 的实际 LoRA rank 会变成：

```text
effective_expert_rank = dim // num_experts_per_tok
```

原因是每个 token 会激活 top-k 个 experts。如果每个 expert 都使用完整 rank，单 token 的 MoE LoRA 激活参数和计算量会随 top-k 放大。`moe_rank_scaling` 用一个简单策略把每个 expert 的 rank 降下来，使总激活 LoRA rank 粗略接近 dense LoRA 的预算。

注意：

- 如果 `dim < num_experts_per_tok` 会报错。
- 如果不能整除，会用 floor division 并打 warning。
- 这个缩放只作用于 MoE grouped expert module，不影响普通 Linear LoRA。

## 12. checkpoint 与 HF/PEFT 格式转换

AutoModel 内部 MoE LoRA 参数是 grouped 3D tensor：

```text
model.layers.N.mlp.experts.lora_gate_and_up_A
model.layers.N.mlp.experts.lora_gate_and_up_B
model.layers.N.mlp.experts.lora_down_A
model.layers.N.mlp.experts.lora_down_B
```

通用 `MoESplitExpertsStateDictMixin` 支持把 grouped expert LoRA 拆成 HF per-expert 格式。

Qwen3 MoE 还有专门的 `Qwen3MoeStateDictAdapter`，支持 PEFT v0.18+ ParamWrapper 兼容格式。

它把 grouped 3D LoRA tensor fold 成 2D：

```text
lora_down_B:        [E, r, H]   -> lora_A.weight             [r*E, H]
lora_down_A:        [E, I, r]   -> lora_B.weight             [I, r*E]
lora_gate_and_up_B: [E, r, 2I]  -> base_layer.lora_A.weight  [r*E, 2I]
lora_gate_and_up_A: [E, H, r]   -> base_layer.lora_B.weight  [H, r*E]
```

反向加载时再从 ParamWrapper 2D 格式恢复为 grouped 3D tensor。

这说明 checkpoint 层也知道 MoE expert LoRA 不是普通 `lora_A/lora_B` module，而是 grouped expert 参数。

## 13. 配置建议

### 13.1 只训练 attention/dense projections

如果目标是不训练 routed experts，只训练 attention 和 dense Linear：

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
    - "*self_attn.q_proj"
    - "*self_attn.k_proj"
    - "*self_attn.v_proj"
    - "*self_attn.o_proj"
    - "*.mlp.gate_proj"
    - "*.mlp.up_proj"
    - "*.mlp.down_proj"
  dim: 16
  alpha: 32
```

这种配置不会 patch `GroupedExperts` 本体。

### 13.2 训练 routed MoE experts

如果目标是训练 routed experts：

```yaml
model:
  backend:
    experts: torch_mm   # 或 gmm，避免 experts: te
    dispatcher: deepep  # 具体看模型和配置

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ["*"]
  dim: 32
  alpha: 32
  dropout: 0.0
  moe_rank_scaling: true
```

如果只想命中 expert，不想命中所有普通 Linear，可以根据实际 module name 写更窄 pattern，例如：

```yaml
peft:
  target_modules:
    - "*.mlp.experts"
```

具体 pattern 要以 `model.named_modules()` 里的名字为准。

### 13.3 避免的组合

不要把下面组合当成 routed expert LoRA：

```yaml
model:
  backend:
    experts: te

peft:
  target_modules: ["*"]
```

当前会命中 `GroupedExpertsTE` 并报不支持。

如果 backend 是 `experts: te`，但只想训普通 Linear，需要用更精确的 `target_modules`，不要用 `["*"]`。

## 14. 常见问题

### Q1: MoE LoRA 会不会真的训练每个 expert 的 adapter？

会。`GroupedExpertsLoRA` / `GroupedExpertsDeepEPLoRA` 的 LoRA 参数第一维是 `num_routed_experts`，每个 expert 有自己的 A/B 参数。

### Q2: top-k routing 下同一个 token 多个 expert 的 LoRA 怎么合并？

每个 token-expert pair 会先被展开并按 expert 排序。每个 expert 计算自己的 base output + LoRA delta。最后用 `scatter_add_` 或 DeepEP unpermutation 把多个 expert output 合回 token 位置。

### Q3: routing weight 乘在哪？

在 expert activation 里乘，位于 gate/up projection 之后、down projection 之前。down bias 如存在，会乘 routing weight。

### Q4: LoRA delta 是不是先 merge 到 base expert weight 再做 grouped GEMM？

不是。当前 forward 不 materialize merged weight，而是额外做两段低秩 grouped GEMM：

```text
x @ A @ B
```

这样训练时只需要 LoRA 参数参与梯度更新，base expert weight 冻结。

### Q5: 为什么 grouped-mm 要 padding LoRA rank？

为了满足 grouped-mm operand 的 stride/alignment 要求。padding 的 rank 维是 0，不改变数学结果。

### Q6: 默认 peft 配置会训练 expert 吗？

通常不会。默认空 `target_modules` 会退成 `*_proj`，只匹配普通 projection 名称。routed grouped expert 本体通常叫 `experts`，不是 `*_proj`。

### Q7: expert LoRA 和 QLoRA 是一回事吗？

不是。QLoRA 指 base weight 量化、LoRA adapter 用 bf16/fp16 等 dtype 训练。MoE expert LoRA 指 LoRA 注入对象是 grouped routed expert 参数。两者可以在概念上组合，但实际支持要看 quantization path、backend 和 checkpoint adapter。

## 15. Review checklist

后续如果 review 或修改 MoE LoRA 相关代码，建议重点检查：

- `target_modules` 是否真的命中预期 module。
- backend 是否是支持 MoE expert LoRA 的 `GroupedExperts` / `GroupedExpertsDeepEP`，而不是 `GroupedExpertsTE`。
- LoRA 参数第一维是否保持 expert index 顺序，不要排序、丢弃或 reshape 错专家维。
- gated expert 的 `gate_and_up` layout 是否仍是 `[gate | up]` 对应的 `2 * moe_inter_dim`。
- `tokens_per_expert` / `offs` 是否和 sorted token order 完全一致。
- `scatter_add_` 是否覆盖 top-k 多 expert 累加。
- zero-token routed path 是否仍有 dummy computation 保持梯度图连通。
- EP 下 `n_routed_experts % ep_size == 0` 是否满足。
- checkpoint adapter 是否能 round-trip grouped 3D LoRA tensor。
- `moe_rank_scaling` 是否符合期望的参数量和激活计算预算。

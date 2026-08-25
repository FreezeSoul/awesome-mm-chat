# Qwen3 MoE 30B TE Model Infrastructure Flow

本文只讲模型对象构建完成后的 infra 阶段，也就是：

```text
model 已经是 Qwen3MoeForCausalLM(...)
  -> instantiate_infrastructure(...) 已经产出 runtime infra 对象
  -> apply_model_infrastructure(...)
    -> PEFT / FP8 / QAT
    -> checkpoint load timing
    -> sharding / EP / CP / AC / FSDP / PP
    -> materialize meta params
    -> load base checkpoint
    -> device placement and runtime fixes
```

本文以 `examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence.yaml` 为主例，但会同时说明 `cp_size > 1`、`ep_shard > 1`、`pp_size > 1` 等组合。

相关代码入口：

- `nemo_automodel/_transformers/auto_model.py`
- `nemo_automodel/_transformers/infrastructure.py`
- `nemo_automodel/_transformers/utils.py`
- `nemo_automodel/components/distributed/mesh.py`
- `nemo_automodel/components/distributed/mesh_utils.py`
- `nemo_automodel/components/distributed/fsdp2.py`
- `nemo_automodel/components/moe/parallelizer.py`
- `nemo_automodel/components/moe/experts.py`

## 1. 本文讨论的起点

在 `NeMoAutoModelForCausalLM.from_pretrained(...)` 里，真正的顺序是：

```text
from_pretrained(...)
  -> resolve DistributedSetup / MeshContext
  -> instantiate_infrastructure(...)
  -> get HF config
  -> _build_model(...)
    -> _init_model(...)                      # 构建 Qwen3MoeForCausalLM
    -> apply_model_runtime_patches(...)
    -> attach_capabilities_and_validate(...)
    -> apply_model_infrastructure(...)       # 本文核心
```

也就是说，`instantiate_infrastructure()` 在模型构建前已经把 infra runtime 对象准备好；`apply_model_infrastructure()` 在模型构建后把这些对象真正作用到模型上。

当前配置的核心分布式参数是：

```yaml
distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 8
  sequence_parallel: false
  activation_checkpointing: true
```

假设启动命令是：

```text
automodel examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence.yaml --nproc-per-node 8
```

那么 `world_size = 8`。

## 2. MeshContext: 所有并行切分的源头

`MeshContext` 是 infra 阶段读并行拓扑的统一入口。它里面主要有两个 mesh：

```text
mesh.device_mesh    # 普通模型并行 / 数据并行 / PP / CP / TP / FSDP 用
mesh.moe_mesh       # MoE expert parallel 专用
```

FSDP2 的 root `device_mesh` 形状来自 `_create_fsdp2_device_mesh()`：

```text
shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
axes  = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
```

其中：

```text
dp_size = world_size / (tp_size * cp_size * pp_size)
dp_shard_size = dp_size / dp_replicate_size
```

同时会注册几个 flatten 轴：

```text
"dp"          = ("dp_replicate", "dp_shard")
"dp_shard_cp" = ("dp_shard", "cp")
"dp_cp"       = ("dp_replicate", "dp_shard", "cp")
```

这些 flatten 轴很重要，因为后面 FSDP 不是直接随便拿一个 mesh，而是通过 axis name 取出正确的 FSDP group。

### 2.1 当前 8 卡 EP 配置的 mesh

当前配置：

```text
world_size = 8
tp_size = 1
cp_size = 1
pp_size = 1
ep_size = 8
dp_replicate_size = 1
```

推导：

```text
dp_size = 8 / (1 * 1 * 1) = 8
dp_shard_size = 8 / 1 = 8

device_mesh.shape =
  (pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1)
```

MoE mesh 的创建逻辑是：

```text
non_pp_size = dp_size * cp_size * tp_size
ep_shard_size = non_pp_size / ep_size, if ep_size < non_pp_size
              = 1, otherwise
moe_mesh.shape = (ep_shard_size, ep_size)
moe_mesh.axes  = ("ep_shard", "ep")
```

当前：

```text
non_pp_size = 8 * 1 * 1 = 8
ep_size = 8
ep_shard_size = 1

moe_mesh.shape = (ep_shard=1, ep=8)
```

结论：

```text
8 张卡组成一个 EP group。
每张卡负责一部分 expert。
expert 参数不会再经过 ep_shard 维度的 FSDP 二次切分，因为 ep_shard=1。
非 expert 参数会通过 FSDP 在 dp_shard_cp mesh 上切分。
```

### 2.2 EP + CP 时 mesh 会变成什么

举例：

```text
world_size = 16
tp_size = 1
cp_size = 2
pp_size = 1
ep_size = 8
```

推导：

```text
dp_size = 16 / (1 * 2 * 1) = 8
non_pp_size = dp_size * cp_size * tp_size = 8 * 2 * 1 = 16
ep_shard_size = 16 / 8 = 2

device_mesh.shape = (pp=1, dp_replicate=1, dp_shard=8, cp=2, tp=1)
moe_mesh.shape    = (ep_shard=2, ep=8)
```

这时会出现两个层次：

```text
ep 维度:
  决定 expert ID 分到哪些 rank。
  apply_ep() 沿 ep mesh 建 token dispatcher。

ep_shard 维度:
  剩余的非 PP rank 维度。
  如果 ep_shard_size > 1，apply_fsdp() 会对每个 rank 持有的 local expert 参数再做 FSDP shard。
```

所以 `ep_size=8` 不等于“除了 EP 以外没有别的 expert 参数切分”。是否还有 expert 参数二次切分，要看：

```text
ep_shard_size = dp_size * cp_size * tp_size / ep_size
```

如果这个值大于 1，expert 参数会在 `ep_shard` 上再被 FSDP shard。

### 2.3 EP 的尺寸约束

代码要求：

```text
non_pp_size = dp_size * cp_size * tp_size
non_pp_size % ep_size == 0
```

MoE parallelizer 还要求：

```text
n_routed_experts % ep_size == 0
```

第一条保证 rank 拓扑能拆成 `(ep_shard, ep)`。
第二条保证 expert ID 能均匀分给 EP ranks。

如果 `n_routed_experts=128`、`ep_size=8`：

```text
num_local_experts = 128 / 8 = 16
```

每个 EP rank 负责 16 个 routed experts。

## 3. instantiate_infrastructure(): 把 config 变成 runtime 对象

`instantiate_infrastructure()` 做四件事：

```text
instantiate_infrastructure(...)
  -> _instantiate_distributed(...)
  -> _instantiate_pipeline(...)
  -> build parallelize_fn
  -> _instantiate_qat(...)
```

当前配置会得到：

```text
model_wrapper = FSDP2Manager(...)
autopipeline = None
parallelize_fn = partial(moe.parallelizer.parallelize_model, ...)
qat_quantizer = None
```

### 3.1 model_wrapper

因为 `strategy: fsdp2`，所以：

```text
_instantiate_distributed(...)
  -> FSDP2Manager(config, device_mesh=mesh.device_mesh, moe_mesh=mesh.moe_mesh)
```

`FSDP2Manager` 保存了这些策略字段：

```text
mp_policy
offload_policy
activation_checkpointing
sequence_parallel
tp_plan
defer_fsdp_grad_sync
reshard_after_forward
enable_compile
...
```

但对当前 MoE EP 配置，真正执行模型切分时不会直接走普通 dense path 的 `FSDP2Manager.parallelize(model)`；因为 `ep_size > 1`，系统会生成 MoE 专用的 `parallelize_fn`。

### 3.2 autopipeline

`_instantiate_pipeline()` 只有在下面条件同时满足时才返回 `AutoPipeline`：

```text
pipeline_config is not None
mesh.device_mesh is not None
mesh.pp_size > 1
```

当前 YAML 里虽然有 `distributed.pipeline` 块，但是：

```yaml
pp_size: 1
```

所以：

```text
autopipeline = None
```

也就是说 pipeline config 在当前配置里 inert，不会进入 PP build。

### 3.3 parallelize_fn

`parallelize_fn` 的选择逻辑：

```text
if mesh.ep_size > 1:
    parallelize_fn = partial(moe.parallelizer.parallelize_model, ...)
elif autopipeline is not None and model_wrapper is not None:
    parallelize_fn = partial(parallelize_for_pp, model_wrapper=model_wrapper)
else:
    parallelize_fn = None
```

当前 `ep_size=8`，所以：

```text
parallelize_fn = MoE parallelizer
```

这个函数后面会由 `_shard_ep_fsdp()` 调用：

```text
parallelize_fn(
    model,
    world_mesh=mesh.device_mesh,
    moe_mesh=mesh.moe_mesh,
    dp_axis_names=...,
    cp_axis_name=...,
    tp_axis_name=...,
    ep_axis_name=...,
    ep_shard_axis_names=...,
)
```

当前 axis kwargs 大致是：

```text
dp_axis_names = ("dp_shard_cp",)
cp_axis_name = "cp"
tp_axis_name = "tp"
ep_axis_name = "ep"
ep_shard_axis_names = ("ep_shard",)
```

注意：`ep_shard_axis_names` 存在不代表一定启用 expert FSDP shard；它还要看 `moe_mesh["ep_shard"].size() > 1`。当前 size 是 1，所以不会启用。

## 4. apply_model_infrastructure(): 总控流程

当前配置下，核心流程可以写成：

```text
apply_model_infrastructure(model)
  -> create Checkpointer for base checkpoint loading
  -> apply PEFT / FP8 / QAT                    # 当前没有，跳过
  -> maybe inject TE attention                 # custom Qwen3Moe 已按 backend 构建，通常不靠这里注入
  -> decide load_before_shard                  # 当前 false
  -> save pre-shard HF state_dict keys
  -> apply freeze / runtime parameter policies
  -> maybe replace loss_fn
  -> if autopipeline:
       _shard_pp(...)
     else:
       _shard_ep_fsdp(...)
  -> materialize meta params
  -> load base model checkpoint
  -> PEFT final freeze                         # 当前没有
  -> print trainable params
  -> model.to(device) decision
  -> dense CP hooks, if needed                 # 当前 cp=1，跳过
  -> cast frozen modules to compute dtype
  -> runtime compatibility fixes
```

下面按阶段展开。

## 5. PEFT / FP8 / QAT 阶段

入口：

```text
_apply_peft_and_lower_precision(
    model,
    mesh.tp_size,
    autopipeline,
    peft_config,
    quantization_config,
    fp8_config,
    qat_quantizer,
)
```

当前配置没有：

```text
peft_config = None
fp8_config = None
qat_quantizer = None
```

所以这个阶段对模型没有实质改动。

如果未来打开 LoRA：

- LoRA 会在 sharding 前插入。
- 如果模型在 meta device 上构建，LoRA module 也会在 `init_empty_weights()` 里建到 meta device。
- checkpoint load 会被强制放到 post-shard 路径，避免 base/adapter 的加载方式不一致。

如果打开 FP8：

- `apply_fp8_to_model()` 会在 sharding 前替换相应模块。

如果打开 QAT：

- QAT 要求模型参数是 bf16。
- `prepare_qat_model()` 会返回修改后的 model 和 `qat_mode`。

## 6. checkpoint load timing: 什么时候加载 base checkpoint

判断函数是 `_should_load_before_shard()`。

它只在所有条件都满足时返回 true：

```text
no_pp
no_tp
no_ep
no_dp_shard
no_peft
need_checkpoint_load
```

展开就是：

```text
autopipeline is None
tp_size <= 1
ep_size <= 1
dp_shard_size <= 1
peft_config is None
pretrained_model_name_or_path exists and load_base_model=True
```

当前配置：

```text
ep_size = 8
dp_shard_size = 8
```

所以：

```text
load_before_shard = False
```

当前会走：

```text
construct model on meta
  -> apply EP/FSDP sharding
  -> materialize meta parameters
  -> load base checkpoint into sharded model
```

为什么不能先 load？

- TP / PP / EP 下，每个 rank 的模型结构或参数视图不同，先 load 容易导致 key/device/collective 不一致。
- DP shard/FSDP 下，先在每张卡加载完整大模型会浪费显存，甚至 OOM。
- PEFT 下，先把 base load 到未 sharded PEFT model，再 post-shard load adapter，容易造成 base/adapter 设备和 key 处理不一致。

### 6.1 什么时候会 load-before-shard

典型单卡或无 sharding 场景：

```text
world_size = 1
tp_size = 1
cp_size = 1
pp_size = 1
ep_size = 1
dp_shard_size = 1
peft_config = None
from_pretrained(...)
```

这时可以：

```text
initialize_model_weights(...)
load_base_model(...)
then maybe model_wrapper.parallelize(...)   # 实际 world_size=1 会跳过
```

## 7. _shard_ep_fsdp(): 非 PP 路径的核心分支

因为当前：

```text
autopipeline = None
parallelize_fn = MoE parallelizer
world_size = 8
```

所以 `_shard_ep_fsdp()` 走第一条分支：

```text
if parallelize_fn is not None and world_size > 1:
    parallelize_fn(model, world_mesh=mesh.device_mesh, moe_mesh=mesh.moe_mesh, ...)
elif model_wrapper.parallelize exists:
    model = model_wrapper.parallelize(model)
```

也就是说当前不是：

```text
FSDP2Manager.parallelize(model)
```

而是：

```text
nemo_automodel.components.moe.parallelizer.parallelize_model(model, ...)
```

普通 dense 模型、或者 `ep_size=1` 的 MoE 配置，才会更可能走 `FSDP2Manager.parallelize(model)`。

## 8. MoE parallelize_model(): 顺序非常重要

MoE parallelizer 的顺序是：

```text
parallelize_model(...)
  -> assert TP unsupported for custom MoE
  -> apply_cp(...)      if cp_size > 1
  -> apply_ep(...)      if ep_size > 1
  -> apply_ac(...)      if activation_checkpointing
  -> apply_fsdp(...)    if FSDP mesh size > 1
```

当前配置实际执行：

```text
apply_cp()      # cp_size=1，跳过
apply_ep()      # ep_size=8，执行
apply_ac()      # activation_checkpointing=true，执行
apply_fsdp()    # dp_shard_cp mesh size=8，执行
```

### 8.1 TP 在 custom MoE path 里不支持

`parallelize_model()` 开头有断言：

```text
assert tp_axis_name is None or world_mesh[tp_axis_name].size() == 1
```

也就是说，当前这个 custom MoE parallelizer 不支持 `tp_size > 1`。

如果给 Qwen3 MoE custom model 配：

```yaml
tp_size: 2
ep_size: 8
```

会在 MoE parallelizer 里失败，而不是自动变成 TP+EP。

## 9. apply_cp(): context parallelism

当前 `cp_size=1`，所以跳过。

当 `cp_size > 1` 时：

```text
apply_cp(model, world_mesh["cp"])
```

它会做几件事：

```text
model._cp_enabled = True
inner_text_model._cp_enabled = True

for each transformer block:
  if attention is TE DotProductAttention:
      attn_module.set_context_parallel_group(...)
  elif attention exposes setup_cp_attention:
      self_attn.setup_cp_attention(cp_mesh)
  elif layer_type is mamba:
      install MambaContextParallel
  elif layer_type is linear_attention:
      store cp_mesh on linear_attn

  if block has MoE:
      moe_module.cp_mesh = cp_mesh
```

对当前 Qwen3 MoE TE backend 来说，如果打开 CP：

- attention backend 是 TE，所以会调用 TE `DotProductAttention.set_context_parallel_group(...)`。
- MoE module 会记录 `cp_mesh`，后续 forward / token dispatch 可以知道 CP 拓扑。
- model forward 侧会通过 `_cp_enabled` 处理 CP 下的输入和 attention mask。

### 9.1 CP 为什么在 EP 前执行

顺序是：

```text
apply_cp()
apply_ep()
```

CP 先设置 attention 和 MoE module 的 CP 语义；EP 再改 experts / token dispatcher。

这能保证后续 EP dispatcher 和 MoE module 看到的是已经带 CP 拓扑信息的模块状态。

### 9.2 dense CP hooks 和 MoE CP 的区别

`apply_model_infrastructure()` 末尾还有一个 dense CP hooks 分支：

```text
if mesh.cp_size > 1 and mesh.ep_size <= 1 and not _uses_te_attention(model):
    attach_context_parallel_hooks(...)
    attach_cp_sdpa_hooks(...)
```

这个只给 dense non-EP 模型使用。

MoE EP 模型的 CP 由 `moe.parallelizer.apply_cp()` 处理。代码明确避免对 EP 模型重复跑 dense CP hooks，因为重复安装 hooks 可能破坏模型自有的 CP attention / mask handling。

## 10. apply_ep(): expert parallelism

当前配置最重要的一步。

入口：

```text
apply_ep(model, ep_mesh=moe_mesh["ep"], moe_mesh=moe_mesh)
```

它会遍历 decoder blocks：

```text
for block in model.model.layers:
    moe_module = block.moe or block.mlp
    if moe_module is MoE:
        parallelize experts
```

对 Qwen3 MoE：

```text
Block(...)
  -> self.mlp = MoE(...)
```

因为 backend：

```yaml
experts: te
dispatcher: deepep
```

所以 experts 是：

```text
GroupedExpertsTE
```

`GroupedExpertsTE` 不能像普通 torch module 那样直接用 DTensor 包 TE `GroupedLinear`，所以代码走特殊分支：

```text
if isinstance(moe_module.experts, GroupedExpertsTE):
    moe_module.experts.init_token_dispatcher(ep_mesh=ep_mesh, moe_mesh=moe_mesh)
else:
    parallelize_module(moe_module.experts, ep_mesh, ExpertParallel())
```

### 10.1 GroupedExpertsTE.init_token_dispatcher()

这一步会重建 TE local experts 和 DeepEP dispatcher。

核心状态：

```text
self.ep_mesh = ep_mesh
self.ep_rank = ep_mesh.get_local_rank()
self.ep_size = ep_mesh.size()
self.moe_mesh = normalized moe_mesh
```

然后检查：

```text
n_routed_experts % ep_size == 0
```

并计算：

```text
num_local_experts = n_routed_experts / ep_size
```

例如 `n_routed_experts=128`、`ep_size=8`：

```text
rank 0 owns experts [0..15]
rank 1 owns experts [16..31]
...
rank 7 owns experts [112..127]
```

实际 local expert IDs 由：

```text
local_expert_indices_offset = ep_rank * num_local_experts
local_expert_indices = offset + range(num_local_experts)
```

然后重新创建两个 TE `GroupedLinear`：

```text
gate_up_linear:
  num_gemms = num_local_experts
  in_features = expert_dim
  out_features = moe_inter_dim * 2    # swiglu/gated path
  device = "meta"

down_linear:
  num_gemms = num_local_experts
  in_features = moe_inter_dim
  out_features = expert_dim
  device = "meta"
```

注意它们是在 EP 初始化时重新创建的，而且仍然在 meta device 上。真正 materialize / load 权重在后面。

然后创建 token dispatcher：

```text
MoEFlexTokenDispatcher(
    num_local_experts=num_local_experts,
    local_expert_indices=local_expert_indices,
    ep_group=ep_mesh.get_group(),
    config=TokenDispatcherConfig(... deepep ...)
)
```

所以 EP 的 runtime 语义是：

```text
router 给每个 token 选 top-k expert IDs
  -> DeepEP dispatcher 根据 expert ID 把 token 送到对应 EP rank
  -> 每个 rank 只计算自己的 local experts
  -> dispatcher 把结果按 token 顺序合并回来
```

### 10.2 非 TE experts 的 EP

如果 experts 不是 `GroupedExpertsTE`，则走：

```text
parallelize_module(module=moe_module.experts, device_mesh=ep_mesh, parallelize_plan=ExpertParallel())
```

`ExpertParallel` 的策略是：

```text
for each direct parameter:
    distribute_tensor(param, ep_mesh, [Shard(0)])
```

也就是沿 expert dimension shard 参数。对 `GroupedExpertsDeepEP`，还会调用：

```text
module.init_token_dispatcher(ep_mesh=device_mesh)
```

TE path 和非 TE path 的结果相似：每个 EP rank 只持有自己负责的 expert 分片；但 TE path 是重建 local `GroupedLinear`，非 TE path 是 DTensor shard 原参数。

## 11. apply_ac(): activation checkpointing

当前配置：

```yaml
activation_checkpointing: true
```

所以执行：

```text
apply_ac(model, ignore_router=True)
```

默认 `ignore_router_for_ac=True`，名字容易误解。实际含义是：

```text
不要在 backward recompute 时重新计算 router。
router projection / routing 结果必须保存下来。
```

原因：

```text
MoE routing 决定每个 expert 收到多少 token。
如果 backward recompute 时重新 routing，某个 expert 的 token 数可能和 forward 不一致。
这样 checkpoint recompute 的 tensor metadata/shape 会和 forward 保存的 metadata 不一致，触发 CheckpointError。
```

所以默认策略是 selective checkpoint policy：

```text
router projection: MUST_SAVE
其他算子: PREFER_RECOMPUTE
```

如果 `activation_checkpointing` 是 selective 字符串模式：

```text
"selective" / "selective_activation_checkpointing" 等
```

则走共享的 selective AC policy，会保存 attention、matmul、collectives、topk 等一组操作；这优先级高于 `ignore_router`。

## 12. apply_fsdp(): FSDP2 wrapping

当前配置：

```text
dp_axis_names = ("dp_shard_cp",)
fsdp_mesh = get_submesh(world_mesh, ("dp_shard_cp",))
fsdp_mesh.size() = 8
```

所以：

```text
fsdp_enabled = True
apply_fsdp(...)
```

传入的关键参数：

```text
ep_enabled = True
ep_shard_enabled = False      # 当前 ep_shard size = 1
ep_shard_mesh = moe_mesh["ep_shard"]
mp_policy = FSDP2Config.mp_policy
reshard_after_forward = MoEParallelizerConfig.reshard_after_forward
lm_head_precision = ...
wrap_outer_model = True
```

### 12.1 transformer block FSDP

`apply_fsdp()` 遍历每个 decoder block：

```text
for block in model.model.layers:
    moe_module = block.mlp
    ...
    fully_shard(block, mesh=fsdp_mesh, ignored_params=...)
```

因为 `ep_enabled=True`，它会把 expert 参数加入 ignored params：

```text
ignored_params = set(moe_module.experts.parameters())
```

这点非常关键：

```text
block FSDP 不会把 experts 当成普通 block 参数一起 shard。
```

否则 expert 参数会同时被 EP 和普通 block FSDP 处理，语义会混乱。

### 12.2 expert 参数是否会再 FSDP shard

取决于：

```text
ep_shard_enabled = ep_shard_mesh is not None and ep_shard_mesh.size() > 1
```

当前：

```text
moe_mesh.shape = (ep_shard=1, ep=8)
ep_shard_enabled = False
```

所以 expert 参数只按 EP 切分，不再做 FSDP shard。

如果是 EP+CP 或更大 world size，导致：

```text
moe_mesh.shape = (ep_shard=2, ep=8)
```

那么：

```text
ep_shard_enabled = True
fully_shard(moe_module.experts, mesh=ep_shard_mesh, shard_placement_fn=_moe_shard_placement, ...)
```

这表示：

```text
先按 ep 维度切 expert ID。
每个 EP rank 得到 local experts。
如果 ep_shard > 1，再把这些 local expert 参数沿 ep_shard 维度做 FSDP shard。
```

`_moe_shard_placement()` 的规则：

```text
1D param: shard dim 0
>=2D expert weight: shard dim 1
```

这里 shard dim 是存储细节。FSDP forward 前会 all-gather，所以不改变 expert compute 的逻辑维度。

### 12.3 embedding / lm_head / inner model / outer model

block 之后，`apply_fsdp()` 还会处理：

```text
embed_tokens
lm_head
inner model
outer model
```

大致逻辑：

```text
if embed_tokens exists and not tied with lm_head:
    fully_shard(embed_tokens)

if lm_head exists and not tied:
    fully_shard(lm_head)

fully_shard(inner text model)

if outer wrapper model exists and wrap_outer_model:
    fully_shard(outer model)
```

对 Qwen3 MoE：

```text
outer model = Qwen3MoeForCausalLM
inner model = Qwen3MoeModel
embed_tokens = model.model.embed_tokens
lm_head = model.lm_head
```

如果 `tie_word_embeddings=True`，代码会先尝试重新建立 input/output embedding tie，避免 FSDP 把 tied 参数当成两个独立参数 shard。当前 Qwen3 MoE 通常不走 tied lm_head，但这个逻辑属于通用保护。

### 12.4 最终参数分布

当前 8 卡配置可以概括为：

```text
non-expert params:
  FSDP2 shard over fsdp_mesh = dp_shard_cp, size 8

expert params:
  EP over ep mesh, size 8
  each rank owns local experts
  no ep_shard FSDP, because ep_shard size = 1

token movement:
  DeepEP dispatcher over ep group
```

如果 `cp_size=2`、`world_size=16`、`ep_size=8`：

```text
non-expert params:
  FSDP2 over dp_shard_cp mesh, size 16

expert params:
  EP over ep mesh, size 8
  plus FSDP over ep_shard mesh, size 2

attention:
  CP over cp mesh, size 2
```

## 13. PP 打开时会有什么不同

当前 `pp_size=1`，所以没有 PP。

如果：

```yaml
pp_size: 2
```

且 pipeline config 有效，则：

```text
autopipeline != None
```

`apply_model_infrastructure()` 会走：

```text
_shard_pp(autopipeline, model, loss_fn, parallelize_fn)
  -> autopipeline.build(model, loss_fn=loss_fn, parallelize_fn=parallelize_fn)
  -> model = autopipeline
```

注意：

```text
AutoPipeline takes care of applying PP + EP + FSDP.
```

也就是说 PP 是外层 orchestrator。非 PP 情况才由 `_shard_ep_fsdp()` 直接调用 MoE parallelizer。

如果 `pp_size > 1` 且 `ep_size > 1`：

```text
parallelize_fn 仍然是 MoE parallelizer
AutoPipeline 在切 stage 的过程中把 parallelize_fn 应用到每个 stage / part
```

如果 `pp_size > 1` 且 `ep_size == 1`：

```text
parallelize_fn = parallelize_for_pp(model_wrapper=FSDP2Manager)
AutoPipeline 用它对 stage 内模型做 FSDP/TP 等普通 parallelize
```

Qwen3 MoE 模型类里有：

```text
_pp_keep_self_forward = True
ModelCapabilities.supports_pp = True
```

这说明 Qwen3 MoE 支持 PP，但保留自己的 forward，不让 generic HF pipeline patch 覆盖它的 THD / CP / rotary 路径。

## 14. materialize meta params

Qwen3 MoE 30B 这种多卡 custom model 通常会在 meta device 上构建：

```text
is_meta_device = True
```

原因是 `_build_model()` 里：

```text
not MegatronFSDPManager/DDPManager
and (world_size > 1 or custom model)
and no quantization_config / native HF quant path
```

所以模型初始参数是 meta tensor。

在 sharding 后，`apply_model_infrastructure()` 判断：

```text
need_materialize =
  is_meta_device
  and not load_before_shard
  and (
      world_size == 1
      or parallelize_fn is not None and world_size > 1
      or model_wrapper.parallelize exists
  )
```

当前：

```text
is_meta_device = True
load_before_shard = False
parallelize_fn is not None
world_size = 8
```

所以：

```text
need_materialize = True
```

执行：

```text
for mp in model_parts:
    checkpointer.initialize_model_weights(mp, init_device)
```

`init_device` 的选择：

```text
if FSDP2 CPU offload enabled:
    init_device = cpu
else:
    init_device = cuda current device
```

当前通常是 CUDA。

这一步的作用：

```text
把 sharded 后仍在 meta 上的参数 materialize 成真实 tensor。
```

注意顺序是 sharding 后 materialize，不是先 materialize 完整模型再 shard。这样可以避免每张卡先持有完整 30B 模型。

## 15. load base model checkpoint

当前 `from_pretrained("Qwen/Qwen3-30B-A3B")` 会设置：

```text
load_base_model = True
pretrained_model_name_or_path = "Qwen/Qwen3-30B-A3B"
```

因为前面没有 load-before-shard，所以：

```text
should_load_checkpoint = True
```

执行：

```text
for mp in model_parts:
    checkpointer.load_base_model(
        mp,
        device,
        cache_dir,
        pretrained_model_name_or_path,
        load_base_model=True,
    )
```

当前不是 recipe checkpoint restore；这是 base HF checkpoint load。

`checkpoint.enabled: false` 只表示训练过程中不保存 checkpoint，不影响 `from_pretrained` 加载 base model。

### 15.1 state_dict_adapter 的作用

Qwen3 MoE 配置里：

```yaml
backend:
  enable_hf_state_dict_adapter: true
```

模型构建时会挂：

```text
model.state_dict_adapter = Qwen3MoeStateDictAdapter(...)
```

load base checkpoint 时，checkpointer 会通过 adapter 把 HF checkpoint key / layout 转成当前 runtime model 的 layout。

这一步很关键，因为当前 runtime 结构已经不是原始 HF 结构：

- attention / linear 可能是 TE module。
- experts 是 `GroupedExpertsTE`。
- EP 后每个 rank 只有 local experts。
- 如果 `ep_shard > 1`，local expert 参数还可能是 FSDP sharded。
- gate/up/down 的 fused 或 grouped layout 和 HF 权重 layout 不同。

所以 checkpoint load 必须理解：

```text
HF global expert weights
  -> 当前 rank local expert IDs
  -> TE GroupedLinear layout
  -> possible DTensor/FSDP placements
```

这就是 adapter 和 `moe_mesh` 参与 load 的原因。

## 16. model.to(device) 的条件

checkpoint load 后，非 PP 路径会检查：

```text
has_sharded_params = any(isinstance(p, DTensor) for p in model.parameters())
```

然后：

```text
if not cpu_offload and not (should_load_checkpoint and has_sharded_params):
    model.to(device, non_blocking=True)
```

当前通常：

```text
should_load_checkpoint = True
has_sharded_params = True
```

所以会跳过 `model.to(device)`。

原因是：

```text
post-shard checkpoint load 后，如果模型里已有 DTensor/FSDP sharded params，
再调用 model.to(device) 可能触发 FSDP reset_sharded_param，
在 tied parameters 或 TP 等场景下可能失败。
```

如果模型没有 sharded params，例如单卡、DDP、或者某些参数仍是普通 tensor，则仍需要 `model.to(device)`，把 checkpoint 里没有覆盖到的 buffers 移到 GPU。

如果 CPU offload 打开：

```text
跳过 model.to(device)
```

因为 FSDP2 期望参数常驻 CPU，并在 forward/backward 时自行搬运。

## 17. runtime compatibility fixes

最后会执行：

```text
_apply_runtime_compatibility_fixes(model)
```

当前主要包含：

```text
fix_rotary_embeddings(...)
install_kv_sharing_holder(...)
```

这是针对部分模型的运行时兼容修复。它不改变当前 EP/FSDP 拓扑，只是在 sharding/load 之后修补某些模型在 FSDP2 或 HF patch 下的 forward 语义。

另外如果 `model_wrapper.mp_policy.param_dtype` 存在，会对 frozen modules 做 compute dtype cast：

```text
cast_frozen_modules_to_compute_dtype(...)
```

这是为了避免冻结模块保留 fp32 参数/缓冲，而训练模块使用 bf16 compute，导致 matmul dtype mismatch。

## 18. 当前配置的完整核心调用链

把所有分支替换成当前配置，得到：

```text
Qwen3MoeForCausalLM 已构建在 meta device
  -> apply_model_infrastructure(...)
    -> create Checkpointer(model_repo_id="Qwen/Qwen3-30B-A3B", moe_mesh=...)
    -> PEFT / FP8 / QAT skipped
    -> load_before_shard = false
       because ep_size=8 and dp_shard_size=8
    -> pre_shard_hf_state_dict_keys captured
    -> freeze/runtime parameter policies
    -> autopipeline is None
    -> _shard_ep_fsdp(...)
      -> parallelize_fn(...)
        -> moe.parallelizer.parallelize_model(...)
          -> TP assert: tp_size must be 1
          -> apply_cp skipped
             because cp_size=1
          -> apply_ep executed
             -> find each block.mlp MoE
             -> GroupedExpertsTE.init_token_dispatcher(ep_mesh=moe_mesh["ep"], moe_mesh=moe_mesh)
             -> local experts recreated on meta
             -> DeepEP token dispatcher initialized
          -> apply_ac executed
             -> checkpoint wrapper per block
             -> router projection saved to avoid rerouting mismatch
          -> apply_fsdp executed
             -> block FSDP over dp_shard_cp mesh
             -> expert params ignored by block FSDP
             -> no expert ep_shard FSDP because ep_shard=1
             -> embed/lm_head/inner/outer model FSDP wrapping
    -> ensure tied lm heads if needed
    -> FSDP2 maybe_compile if configured
    -> materialize meta params on cuda
    -> load base HF checkpoint post-shard
       -> Qwen3MoeStateDictAdapter maps HF weights into TE + EP layout
    -> print trainable params
    -> skip model.to(device) if checkpoint loaded into DTensor-sharded params
    -> dense CP hooks skipped because cp_size=1
    -> cast frozen modules to compute dtype if needed
    -> runtime compatibility fixes
```

## 19. 三个常见变体

### 19.1 只开 FSDP，不开 EP

例如：

```yaml
ep_size: 1
cp_size: 1
pp_size: 1
```

此时：

```text
parallelize_fn = None
_shard_ep_fsdp(...)
  -> model_wrapper.parallelize(model)
    -> FSDP2Manager.parallelize(model)
      -> fsdp2_strategy_parallelize(...)
```

也就是说 dense path 由 `components/distributed/parallelizer.py` 的策略系统处理，而不是 `components/moe/parallelizer.py`。

### 19.2 EP + CP

例如：

```yaml
ep_size: 8
cp_size: 2
pp_size: 1
```

如果 world size 允许，核心顺序：

```text
parallelize_model(...)
  -> apply_cp()
     -> TE attention gets CP group
     -> MoE module records cp_mesh
  -> apply_ep()
     -> experts split across ep
     -> DeepEP dispatcher over ep group
  -> apply_ac()
  -> apply_fsdp()
     -> non-expert FSDP over dp_shard_cp
     -> expert FSDP over ep_shard if ep_shard > 1
```

注意 `dp_shard_cp` 会把 CP 维度纳入 FSDP mesh，所以非 expert 参数的 FSDP shard group 可能覆盖 DP shard 和 CP 的组合维度。

### 19.3 EP + PP

例如：

```yaml
ep_size: 8
pp_size: 2
```

此时：

```text
autopipeline != None
apply_model_infrastructure(...)
  -> _shard_pp(...)
    -> autopipeline.build(..., parallelize_fn=moe.parallelizer.parallelize_model)
```

PP 是外层。模型先按层切 stage，然后每个 stage 内再按传入的 `parallelize_fn` 做 EP / AC / FSDP。

checkpoint load 仍然是 post-shard，因为：

```text
autopipeline is not None
```

`_should_load_before_shard()` 会返回 false。

## 20. 最容易混淆的点

### 20.1 `ep_size=8` 和 `dp_size=8` 不是同一个东西

当前 8 卡配置里它们数值相同，但语义不同：

```text
dp_size=8:
  来自 world_size / (tp * cp * pp)
  用于普通数据并行/FSDP shard 拓扑

ep_size=8:
  来自 distributed.ep_size
  用于 expert ID 的分布
```

当前它们刚好都等于 8，所以：

```text
moe_mesh = (ep_shard=1, ep=8)
```

但 world size 或 cp 改变后，二者关系会变化。

### 20.2 EP 不等于 FSDP

EP 做的是：

```text
expert ID 按 rank 分布
token 按 routing 结果跨 rank dispatch
```

FSDP 做的是：

```text
参数存储 shard
forward 前 all-gather
backward 后 reduce-scatter / reshard
```

当前 expert 参数主要靠 EP 降低每卡持有的 expert 数；非 expert 参数靠 FSDP 降低每卡持有的普通参数。

### 20.3 `ep_shard` 是 EP 之后的剩余 rank 维度

公式：

```text
ep_shard_size = (dp_size * cp_size * tp_size) / ep_size
```

如果等于 1：

```text
expert 参数只 EP，不再 FSDP shard
```

如果大于 1：

```text
expert 参数先 EP，再在 ep_shard 上 FSDP shard
```

### 20.4 activation checkpointing 保存 router 不是性能细节，而是正确性保护

MoE routing 影响 tensor shape。
如果 recompute 走出不同 routing，expert input shape 可能变化，checkpoint backward 会失败。

所以默认：

```text
ignore_router_for_ac=True
```

实际含义是保存 router / topk 相关输出，避免 backward 重新 routing。

### 20.5 当前 `pipeline:` YAML 块没有生效

因为：

```yaml
pp_size: 1
```

所以：

```text
autopipeline = None
```

只有 `pp_size > 1` 时 pipeline config 才会实例化 `AutoPipeline` 并接管 PP + stage 内 sharding。


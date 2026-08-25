# vLLM PR 46701: Trace Replay Notes

PR: <https://github.com/vllm-project/vllm/pull/46701>

## PR 要做什么

这个 PR 给 vLLM V1 sampling 增加一个新的参数：

```python
SamplingParams(trace_decode_token_ids=[...])
```

它的含义是：用户提前给定一串 decode token ids，vLLM 在 decode
过程中不让 sampler 自己决定输出 token，而是每一步强制输出 trace
里对应位置的 token。

普通确定性生成是：

```text
y_t = argmax/model_sample(p(. | prompt + y_<t))
```

trace replay 是：

```text
y_t = trace[t]
同时计算 logprob(trace[t] | prompt + trace[:t])
```

所以它不是为了让同一个 prompt 的输出更稳定。它是为了让模型沿着一条
已经存在的 token 轨迹走，并返回模型在每个 decode step 对这些 token
的 logprob/rank。

## 为什么 deterministic inference 不完全覆盖这个需求

如果模型权重、dtype、backend、TP 配置、采样参数都不变，并且使用确定性
推理，那么同一个 prompt 理论上应该输出一样的 token。这种情况下确实没有
必要为了拿同一批 token 的 logprob 再 trace replay 一遍。

但 trace replay 关注的是另一个问题：

```text
给定一条固定 token trace，另一个配置/engine/权重在每一步给这条 trace
里的 token 多少概率？
```

例如配置 A 生成：

```text
prompt -> A B C D
```

如果配置 B 自己 deterministic decode，可能得到：

```text
prompt -> A B X Y
```

一旦第 3 个 token 从 `C` 分叉成 `X`，后续上下文就变了。此时配置 B 生成
得到的是：

```text
logprob(Y | prompt + A + B + X)
```

但我们真正想比较的是：

```text
logprob(D | prompt + A + B + C)
```

trace replay 的作用就是强制配置 B 也沿着 `A B C D` 这条轨迹走，这样才能
逐 token 做 apples-to-apples 的 logprob 对比。

## PR message 里提到的两个用途

### 1. 对比不同 settings/backends

PR 里说可以比较不同配置下的数值差异，例如：

- dtype
- quantization
- attention backend
- MoE backend
- TP size

这个用途比较容易理解，本质上是 debug/evaluation/logprob divergence
分析。固定同一条 token trace 后，不同配置都在完全相同的上下文和目标 token
上计算 logprob，比较结果才有意义。

### 2. RL training

PR 里还说这个功能可以用于 RL training：对 rollout 的 exact tokens 在
training engine 下重新计算 logprobs，从而测量并最小化 rollout-train
logprob gap，避免重新采样导致轨迹漂移。

这里需要区分几种 logprob：

```text
old_logprob:
  rollout engine 生成 token 时返回的 logprob

new/current_policy_logprob:
  training engine / current policy 对同一批 rollout tokens 的 logprob

ref_logprob:
  reference model 对同一批 rollout tokens 的 logprob
```

在常规 RL 训练流程里，rollout engine 如果已经 request logprobs，那么
`old_logprob` 本来就会随生成结果返回。训练侧也通常会用 teacher forcing /
forward pass 对 `prompt + response_tokens` 一次性计算 `new_logprob` 或
`ref_logprob`。

因此，如果只是同一个 rollout engine、同一套配置，并且生成时已经返回
logprobs，那么这个 trace replay 接口对常规训练热路径没有额外必要。

它更可能用于 RL infra 的对齐、校验或补算场景：

- rollout 当时没有打开 logprobs，事后想对已生成 tokens 补算 decode logprobs。
- 想用另一个 vLLM engine/config 对同一条 rollout trace 重新打分。
- 想比较 vLLM decode-time logprobs 和 training forward/teacher-forcing
  logprobs 的 gap。
- 想确认 rollout engine 和 training engine 之间的 logprob mismatch 来自哪里。
- 想在不同 dtype/backend/quant/TP 下对同一条 rollout tokens 做 decode-path
  logprob 对比。

所以我目前的理解是：

```text
这个接口不是常规 RL loss 计算必需的新 API。
它更像是 RL 对齐、数值校验、补算 decode logprob 的工具。
```

## 当前疑问

我的主要疑问是 PR message 中的 RL training 用途是否表述得过宽。

如果 rollout engine 本来就会返回 generated tokens 的 logprob，training
engine 也会通过 teacher forcing / forward pass 对同一批 rollout tokens
计算 logprob，那么 rollout-train logprob gap 已经可以直接测量。

因此需要进一步确认：

1. 这个接口在 RL 场景下是否只是为了比较 vLLM decode-path logprob 和
   training forward logprob？
2. 是否存在某些训练设置，training engine 必须通过 decode path replay tokens，
   而不能用普通 forward pass？
3. 如果只是为了补算/校验 logprob，那么 PR description 里说的 RL training
   是否应该更准确地描述为 RL infra validation / logprob alignment？

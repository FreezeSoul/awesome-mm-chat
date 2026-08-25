# Polar 论文阅读笔记：RL 训练系统关注点

论文：Polar: Agentic RL on Any Harness at Scale  
arXiv: https://arxiv.org/abs/2605.24220  
定位：Polar 不是 trainer，而是面向复杂 agent harness 的 rollout service。它负责运行真实 harness、代理模型请求、捕获 token 级交互、重建训练轨迹、执行 evaluator，并把 traces / rewards 回传给外部训练框架。

## 1. 整体训练视角

Polar 的训练系统接口可以理解为：

```text
trainer -> submit rollout task -> Polar 跑真实 harness -> callback traces/rewards -> trainer update
```

论文实验里，trainer 侧使用的是 Slime asynchronous GRPO。Polar 本身不替代 PPO / GRPO / SFT trainer，它解决的是 agentic RL 中 rollout 难以接入真实 harness 的问题。

它的核心设计是：不把 agent harness 改写成 RL environment，而是把集成边界放到 LLM API endpoint。harness 正常运行，只是模型请求被路由到 Polar gateway proxy。proxy 记录 prompt tokens、sampled response tokens、logprobs、消息结构和工具信息，然后重建 trainer 可消费的 trajectory。

## 2. 3.1 Architecture

Polar 分成两个核心组件：

- Rollout server
- Gateway nodes

Rollout server 是全局协调层。它接收一个 `TaskRequest`，根据 `num_samples` 展开成多个独立 sessions。session 是调度单位，包含 `session_id`、`task_id`、timeout、runtime spec、agent spec、trajectory builder、evaluator 和 callback URL。

Rollout server 负责：

- 接收任务提交；
- 展开 sessions；
- 分发 sessions 到 gateway nodes；
- 持久化终态结果；
- 提供任务状态查询；
- 接收 gateway 完成后的 callback。

Gateway node 是实际执行层。它负责一个 session 的完整生命周期：

- 启动 runtime，比如 Docker 或 Apptainer；
- 准备 harness 环境；
- 运行 harness 命令；
- 托管模型 API proxy；
- 捕获 harness 发出的模型请求；
- 根据捕获 completions 构建 trajectories；
- 运行 evaluator；
- 清理资源；
- 回传结果。

关键点是 gateway 和 model proxy co-locate。这样 proxy 捕获到的 completion 可以直接绑定到当前 session registry，不需要额外 trace collection service。

## 3. Task / Session / GRPO Group 语义

一个 `TaskRequest` 里有 `num_samples`。Polar 会把一个 task 展开成多个独立 sessions。对于 GRPO 来说，这通常对应同一个 prompt 下的多个 sampled rollouts，用于组内 relative advantage。

训练系统需要重点确认：

- `num_samples` 是否等于 GRPO group size；
- 每个 sample 是否使用独立 runtime；
- 一个 task 下的 samples 是否必须全部完成才训练；
- timeout / failed sample 是丢弃、补采样，还是给 0 reward；
- reward normalization 是按 task group 做，还是按全局 batch 做；
- callback 是 session 级返回，还是 task-level 全部完成后返回；
- group 内样本是否必须来自同一个 policy version。

论文给出了 payload 里的 `policy_version` 和 `rollout_step`，说明它们至少在 metadata 里被显式记录，但 staleness 处理细节主要在外部 async trainer 侧。

## 4. 3.2 Harness and Proxy Capture

Proxy capture 是训练信号正确性的源头。Polar 不是从 harness log 里事后解析文本，而是在模型 API proxy 处捕获 token 级数据。

每个模型请求大致经过四步：

1. 检测 provider API 类型：Anthropic Messages、OpenAI Chat Completions、OpenAI Responses、Google `generateContent` 等。
2. 规范化请求：转换 roles、content parts、tools、tool choice、stop controls、generation params，并补充训练所需字段，比如 `logprobs=true`。
3. 捕获 token 级数据：转发到 inference backend，并保存 request messages、response messages、prompt token IDs、sampled response token IDs、finish reason 和 logprobs。
4. 返回 provider 兼容格式：把本地 inference response 转回 harness 期望的 provider schema。

训练系统需要关注：

- `old_logprobs` 必须来自 rollout 时真实 behavior policy；
- 不要把 response text 重新 tokenize 后当成 rollout tokens；
- tool call JSON、reasoning 字段、stop tokens、role/content 转换都可能改变 token 序列；
- provider schema 转换如果不严格，会导致训练信号和真实 harness 行为不一致。

### Streaming 处理

论文中一个重要实现细节是：对于 streaming request，Polar 上游实际拿 non-streaming response，再向 harness 发 synthetic provider-shaped stream。

这个设计利于完整捕获 token IDs 和 logprobs，但有潜在风险：

- 如果 harness 依赖真实 streaming timing，行为可能被改变；
- 如果 harness 会根据 partial output 早停或触发工具调用，synthetic stream 可能改变执行路径；
- 如果只是普通 SSE 消费，风险较低。

## 5. 3.3 Asynchronous Rollout Staging

这一节是 rollout 吞吐的关键。Agentic RL 的 rollout 不是简单的 prompt -> response，而是混合了容器启动、repo 准备、依赖安装、harness 执行、多轮模型调用、工具调用、patch 生成、评测、轨迹构建和清理。

这些阶段的资源瓶颈不同：

- runtime setup 多是 CPU / IO / filesystem heavy；
- harness execution 往往受模型 inference 和工具执行影响；
- evaluator 可能长尾严重，比如跑 SWE-Bench 测试；
- trajectory reconstruction 和 callback 属于 post-run 工作。

Polar 在 gateway 内部把 session lifecycle 拆成多个阶段：

- `INIT`：启动 runtime，执行 prepare actions；
- `READY`：有界 buffer，存放已经初始化好、等待执行的 runtime；
- `RUNNING`：运行真实 harness，模型调用通过 proxy 捕获；
- `POSTRUN`：构建 trajectory，运行 evaluator，执行 post-run hooks，发送 callback，清理资源。

这里的 pool 不一定是线程池。它更准确地说是阶段隔离的 worker pool，可以用线程、进程、asyncio task pool 或多进程 worker 实现。重点不是实现形式，而是保留以下语义：

- 各阶段独立并发上限；
- READY buffer 有界，避免预热 runtime 过多；
- INIT / RUNNING / POSTRUN 互不阻塞；
- 队列之间有 backpressure；
- timeout / cancellation 后仍能进入 cleanup；
- evaluator prewarm 可以和 agent run 并行。

如果用 `async def` 重写，不能简单把所有函数改成 async。`async def` 内部如果调用阻塞操作，比如 `subprocess.run()`、同步 Docker SDK、同步 requests、大量文件 IO 或 CPU-heavy evaluator，会直接卡住 event loop。

更合理的 async 实现方式是：

- HTTP proxy、callback、status API 用 async；
- 子进程使用 `asyncio.create_subprocess_exec`；
- 同步 Docker / shell wrapper 用 `asyncio.to_thread()` 或 executor；
- CPU-heavy trajectory / evaluator 逻辑考虑 process pool；
- 每个 stage 用 bounded queue / semaphore；
- session cleanup 放在 `try/finally`。

## 6. 3.4 Trajectory Reconstruction

Trajectory reconstruction 是训练正确性的核心。Polar 捕获到的是一个 session 里的多次 LLM API calls，但 trainer 需要的是结构化训练样本：`prompt_ids`、`response_ids`、`loss_mask`、`response_logprobs`、reward 和 metadata。

输入是 `CompletionSession`，即一个 harness session 中按顺序捕获到的 completions。每个 completion 包含：

- prompt token sequence `p_i`；
- 真实 sampled response tokens `a_i`；
- response logprobs `l_i`；
- prompt / response messages；
- tools、finish reason、metadata 等。

输出是 `Trajectory`。一个 trajectory 里可以有一个或多个 `Trace`。每个 trace 是 trainer 真正消费的样本。

### Per Request

`per_request` 是保守 baseline：每次 completion 都变成一个 trace。

优点：

- 简单；
- 单次调用内 token / logprob 对齐直接；
- 不会错误合并不同上下文。

问题：

- 一个 agent session 可能产生几十到上百个 traces；
- trainer 样本数量暴涨；
- 短 trace 难表达完整多轮行为；
- 如果把 session-level outcome reward 广播给每个 request-level trace，会带来严重 credit assignment 噪声。

论文明确提到，他们尝试过 `per_request` 加 outcome-reward broadcasting，观察到了明显 reward hacking。

### Prefix Merging

`prefix_merging` 的目标是：当 harness 的对话历史是 append-only 时，把多次 completion 合成长 trace；当发生 context compaction、prompt rewrite、subagent 分支、并行对话等情况时，不强行合并。

合并条件很严格。对于相邻 completions `C_i` 和 `C_{i+1}`，后一次 prompt 必须以前一次 prompt 为 token prefix：

```text
p_{i+1}[1 : |p_i|] = p_i
```

此外还需要 normalized message-level grouping key 匹配，用来避免把不同子 agent 或不同 conversation branch 错合并。

这意味着：

- append-only 多轮对话可以合并；
- context compaction 会断开；
- subagent 会形成独立 chain；
- prompt rewriting 或工具上下文替换会断开；
- 并行分支不会被串到同一条 trace。

### 为什么不能直接使用后一次 prompt 的 tail

在多轮对话中，`p_{m+1}` 通常包含：

```text
p_m + 上一轮 assistant 回复的 canonical rendering + harness 插入内容
```

但上一轮 assistant 回复在 `p_{m+1}` 中是 provider / server 重新渲染后的 canonical tokens，不一定等于模型当时真实采样出的 `a_m`。

训练时必须使用真实 sampled tokens `a_m`，因为它们才是 behavior policy 的动作，并且有对应 old logprobs。直接用 canonical rendering 会引入 retokenization drift 和行为策略不一致。

Polar 的构造是：

```text
z = p_1 || a_1 || u_1 || a_2 || u_2 || ... || a_K
```

其中：

- `a_m` 是真实 sampled assistant tokens，`loss_mask = 1`；
- `u_m` 是 canonical interstitial tokens，比如 end-of-turn、工具结果、下一轮用户上下文等，`loss_mask = 0`；
- `u_m` 的 logprob 用 synthetic entries 填齐，只为了保证 `response_ids` 和 `response_logprobs` 对齐；
- 真正训练只看 `loss_mask = 1` 的 tokens。

论文里的正确性 invariant 是：

```text
Every trainable token matches the behavior policy during rollout,
and any non-generated tokens are masked out.
```

训练系统必须支持 response 内部的局部 loss mask。不能默认整个 response 都参与 policy loss。

### Prefix Merging 对 trainer contract 的要求

trainer 侧必须保证：

- policy loss 只在 `loss_mask=1` tokens 上计算；
- KL 只在 `loss_mask=1` tokens 上计算；
- ratio / old_logprobs 只在 `loss_mask=1` tokens 上计算；
- synthetic logprob entries 不参与训练；
- sequence packing 不能破坏 `response_ids`、`response_logprobs`、`loss_mask` 的对齐；
- 如果 trainer 支持 prompt/response 二段 mask，但不支持 response 内部 mask，需要改造。

论文实验中，`prefix_merging` 把 3 个训练 step 的 trainer stream 从 1185 个 request-level updates 降到 218 个 merged-trace updates，wall-clock time 从 189.5 分钟降到 35.2 分钟，约 5.39x。rollout GPU utilization 从 20.4% 提升到 87.7%。

结论：trajectory builder 不是普通数据预处理模块，而是训练吞吐和训练正确性的核心组件。

## 7. 3.5 Evaluation and Reward Propagation

Polar 的 evaluator 是 registry-backed，可以扩展不同 reward 逻辑。内置包括：

- session-completion reward；
- configurable test-on-output evaluator；
- SWE-Bench / SWE-Gym evaluator。

Evaluator 在 trajectory construction 之后运行，接收 trajectory、session artifacts，以及可选的 fresh runtime context。

训练系统需要关注 reward granularity：

- session-level reward；
- trace-level reward；
- token-level / process reward。

对于 SWE 类任务，reward 通常是 session-level outcome reward：最终 patch 是否通过 verifier。这个 reward 可以广播到每个 trace，但论文指出这在 `per_request` 下容易 reward hacking，因为 request-level traces 收到 session-level credit 会带来噪声 credit assignment。

因此需要明确：

- 多 trace 来自同一 session 时 reward 怎么分配；
- GRPO advantage 是按 session group 还是 trace group；
- 是否需要 session normalization；
- 是否需要 process reward model；
- failed / timeout / empty generation 是否进入训练；
- evaluator crash 或没有结果时如何标记 reward。

## 8. Async RL：Policy Version 和 Staleness

Polar 是 rollout-as-a-service，trainer 和 rollout 解耦，所以样本天然可能 stale。附录 payload 中包含：

```json
{
  "metadata": {
    "group_id": "{rollout_group_id}",
    "policy_version": "{policy_version}",
    "rollout_step": "{rollout_step}"
  }
}
```

训练系统需要重点处理：

- rollout 使用的是哪个 policy version；
- trainer 收到样本时当前 policy 已更新多少步；
- 是否丢弃 stale samples；
- 是否设置 staleness threshold；
- 是否启用 TIS / importance correction；
- GRPO group 内样本是否必须同版本；
- old_logprobs 是否和 behavior policy version 精确对应。

论文附录超参中写了 `TIS Enabled`，但没有展开算法细节。这个点如果要落实现，需要结合 Slime / trainer 代码继续确认。

## 9. Timeout 和 Partial Trace

论文提到，如果 harness timeout，但之前已经捕获到模型调用，gateway 仍会进入 POSTRUN，尽量恢复 partial traces，并带上 terminal timeout status。

训练系统需要决定：

- partial trace 是否进入训练；
- reward 给 0、负数，还是丢弃；
- incomplete GRPO group 是否补采样；
- evaluator 没结果时 callback schema 如何表达；
- cancellation 时是否仍然保证 cleanup；
- 是否可能重复 callback；
- timeout session 的 old_logprobs 是否完整。

这类边界情况在 agentic RL 里会很常见，因为 SWE / OS / browser 任务都有明显长尾。

## 10. Gateway Staging 参数对训练吞吐的影响

即使 trainer 和 inference engine 很快，如果 gateway staging 配置不合理，trainer 也会等 rollout。

需要关注的配置包括：

- `INIT` worker 数：控制 runtime/container 准备并发；
- `READY` buffer 大小：控制预热 runtime 数量；
- `RUNNING` worker 数：控制真实 harness 执行并发；
- `POSTRUN` worker 数：控制 trajectory/eval/callback 并发；
- evaluator prewarm 是否开启；
- callback 和 result persistence 是否会阻塞 postrun；
- backpressure 是否能防止任务堆积压垮机器；
- gateway health / heartbeat 是否能反映 stage pressure。

这些配置决定 rollout supply 是否稳定，也决定 trainer 是否能持续拿到完整 batch。

## 11. 实验中和训练系统最相关的数字

SWE-Gym GRPO：

- Base checkpoint：`Qwen/Qwen3.5-4B`
- Training data：`NovaSky-AI/SkyRL-v0-293-data`，293 tasks
- Trainer：Slime asynchronous GRPO
- Epochs：1
- Rollout batch size：4
- Samples per prompt：16
- Trace construction：`prefix_merging`
- Optimizer：Adam
- Learning rate：`1e-6`
- Weight decay：0.1
- TIS：Enabled

SWE-Bench Verified 结果：

| Harness | Base | Polar RL | Gain |
| --- | ---: | ---: | ---: |
| Codex | 3.8% | 26.4% | +22.6 |
| Claude Code | 29.8% | 34.6% | +4.8 |
| Qwen Code | 34.6% | 35.2% | +0.6 |
| Pi | 34.2% | 40.4% | +6.2 |

这些分数说明 harness-native RL 对不熟悉的 action protocol / context policy / tool schema 帮助最大。Codex harness 下 Qwen3.5-4B base 很弱，RL 后提升最大。

但对训练系统来说，更关键的是 builder ablation：

| Builder | Updates / traces over 3 steps | Time | Rollout GPU util |
| --- | ---: | ---: | ---: |
| `per_request` | 1185 | 189.5 min | 20.4% |
| `prefix_merging` | 218 | 35.2 min | 87.7% |

这说明 trace 粒度会直接影响训练吞吐、GPU 利用率和 credit assignment。

## 12. 最容易漏的点

如果只关注 RL 训练系统，最容易漏下面这些：

1. `old_logprobs` 必须来自 proxy 捕获的真实 sampled tokens，不能从文本重 tokenize。
2. Provider schema 转换可能改变 prompt / tool call tokenization，进而影响训练信号。
3. Synthetic streaming 可能改变依赖真实 streaming 的 harness 行为。
4. `prefix_merging` 后 response 内部有 `loss_mask=0` tokens，trainer 必须支持局部 mask。
5. KL、ratio、entropy、policy loss 都应该只在 trainable tokens 上算。
6. Session-level reward 直接广播到 request-level traces 可能 reward hacking。
7. `num_samples`、GRPO group、callback 粒度、timeout 样本补齐策略需要明确。
8. Async rollout 一定要处理 `policy_version`、staleness 和 TIS / importance correction。
9. Partial traces 和 timeout sessions 是常态，不是异常小概率。
10. Gateway 的 INIT / READY / RUNNING / POSTRUN 配置会直接决定 trainer 是否饥饿。

## 13. 三句话总结

1. 训练正确性靠 proxy capture + token-faithful reconstruction，不能靠文本重放。
2. 训练吞吐靠 prefix_merging + asynchronous rollout staging，不只是 trainer / inference engine 的问题。
3. RL 稳定性靠 reward granularity、loss mask、policy version / staleness、GRPO group 语义处理正确。

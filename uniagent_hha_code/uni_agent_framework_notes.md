# Uni-Agent 框架源码阅读总结

> 日期：2026-07-12  
> 仓库：`/mnt/shared-storage-user/huanghaian/code/uni-agent`  
> 目标：理解这个基于 `verl` 的 agent 验证、推理和训练框架，以及当前 open PR 透露的发展方向。

## 1. 总体判断

Uni-Agent 不是一个单纯的 SWE-Bench agent，也不是只在 `verl` 外面包一层脚本。它更像一个面向 agent 的 rollout/runtime layer：

- 下接 `verl` 的 rollout server、fully async trainer、TransferQueue、RewardLoopWorker。
- 上接各种 agent 形态：内建白盒 agent loop、黑盒 coding agent、OpenAI/Anthropic-compatible 外部 agent、在线 OpenClaw client、GUI/VLM agent。
- 中间统一处理 sandbox 环境、工具调用、trajectory/token mask、reward 验证、并发调度和日志。

一句话概括：

```text
Uni-Agent = verl 训练/推理底座 + agent 环境执行层 + Gateway 协议/轨迹层 + benchmark reward 验证层
```

目前主干里已经有两条核心路径：

```text
白盒路径:
  verl rollout model
    -> UniAgentLoop
    -> AgentInteraction
    -> ToolsManager
    -> AgentEnv sandbox
    -> RewardSpec
    -> AgentLoopOutput

黑盒路径:
  external agent runner
    -> Uni-Agent Gateway
    -> verl rollout model
    -> GatewaySession materializes token-level trajectories
    -> RewardLoopWorker / TransferQueue
```

open PR 说明他们正在把黑盒、在线训练、多分支 Gateway session、GUI/VLM 都做成一等能力。

## 2. 仓库模块分层

核心代码在 `uni_agent/`：

```text
uni_agent/
  agent_loop.py            # 白盒 agent loop: verl AgentLoopBase 实现
  interaction/             # model/tool/env 三者闭环
  deployment/              # host/local/local_attach/modal/vefaas sandbox 后端
  tools/                   # 内建工具及 OpenAI function schema
  reward/                  # SWE-Bench/R2E/Terminal-Bench/Search 等 reward spec
  framework/               # 黑盒 OpenAI-compatible AgentFramework + TQ 写入
  gateway/                 # 黑盒路径核心: session, codec, FastAPI/Ray actor
  skills/                  # skill manifest + sandbox 内按需读取
```

周边模块：

```text
examples/
  agent_interaction/       # 并发推理/验证入口
  agent_train/             # fully async RL 训练脚本
  data_preprocess/         # benchmark parquet 生成
  search_agent/            # search agent 训练例子
  swe_agent_235b/          # 大模型 SWE agent recipe

dashboard/                 # 读取 log_dir 的轻量运行监控面板
docs/                      # readthedocs 文档
verl/                      # git submodule，提供训练、rollout、TQ 等底座
```

## 3. 白盒路径：UniAgentLoop

入口是 `uni_agent/agent_loop.py` 的 `UniAgentLoop`，继承 `verl.experimental.agent_loop.agent_loop.AgentLoopBase`。

每个 sample 的执行流程：

```text
1. _init_config()
   YAML defaults + sample.extra_info.tools_kwargs deep merge

2. 初始化组件
   AgentChatModel
   ToolsManager
   SkillsManager
   AgentEnv
   RewardSpec

3. 启动环境
   env.start()
   env.install_tools()
   env.install_skills()

4. 多轮交互
   AgentInteraction.run()

5. reward
   reward_spec.compute_reward()

6. 转成 verl 输出
   convert_to_agent_output() -> AgentLoopOutput
```

关键设计是 `_init_config()`：

- YAML 放全局默认值，比如 `concurrency`、`log_dir`、默认 `tools/env/reward/interaction`。
- 每个 parquet sample 的 `extra_info.tools_kwargs` 可以覆盖这些字段。
- `model` 字段禁止从 sample 覆盖，永远由 verl rollout config 和 server manager 合成。

这使得两种使用方式都成立：

```text
模式 1：YAML 配大部分内容，dataset 只补 image/reward metadata。
模式 2：dataset 每条样本带完整 env/reward/tools/interaction，YAML 只当 thin shell。
```

Terminal-Bench v2 就是模式 2：每个 task 的 Docker image、资源、测试 archive、reward metadata 都在 parquet 行里。

## 4. AgentInteraction：多轮状态机

`uni_agent/interaction/interaction.py` 是白盒 agent 的核心状态机。

一轮 step 的逻辑：

```text
messages -> model.query()
       -> parse tool calls
       -> tool call -> bash command
       -> env.run_action()
       -> observation -> role=tool message
       -> append to messages and rollout_cache
```

它维护两个并行状态：

- `messages`：chat history，给模型下次生成使用。
- `rollout_cache`：token 级训练轨迹，包括 `prompt_ids`、`response_mask`、`response_logprobs`、`routed_experts`、metrics。

mask 的含义很重要：

```text
模型生成 token: response_mask = 1
工具 observation / user/tool continuation token: response_mask = 0
```

因此 RL 训练只优化模型自己生成的内容，不优化工具输出。

退出条件包括：

- `finished`：模型调用 `finish` 或 `submit`。
- `token_limit`：上下文超长。
- `terminal_dead`：sandbox shell 不可用。
- `timeout_budget_exhausted`：命令超时超过预算。
- `max_step_limit`：达到最大轮数。
- `format_error`：模型没按工具调用格式输出，会作为 tool error 拼回上下文继续。

这套设计很适合训练 coding agent：模型可以犯格式错、命令错、超时，但这些错误会变成轨迹的一部分，而不是直接 kill 整个训练进程。

## 5. Model 抽象

`uni_agent/interaction/model.py` 里有两个 model wrapper。

### AgentChatModel

训练/verl 路径使用：

```text
AgentChatModel
  -> self.client.generate(request_id, prompt_ids, sampling_params)
  -> token_output.token_ids
  -> tokenizer.decode()
  -> 更新 rollout_cache
```

这里的 `client` 是 verl 的 async LLM server manager client。它直接拿 token ids，不走 OpenAI HTTP API，因此可以拿到 logprobs、routed experts、extra fields。

### OpenAICompatibleChatModel

推理/外部 API 场景使用：

```text
OpenAICompatibleChatModel
  -> AsyncOpenAI.chat.completions.create()
  -> OpenAI structured tool_calls
```

它更像一个 inference-only adapter，不是主训练路径。

## 6. ToolsManager 与工具系统

工具定义在 `uni_agent/tools/`，抽象基类是 `AbstractTool`。

一个 tool 需要提供：

- `name`
- `local_path`
- `get_tool_schema()`：OpenAI function schema
- `get_install_command()`
- 是否 `copy_to_remote`

`ToolsManager` 做三件事：

```text
1. 根据 YAML 里的 tools list 实例化工具。
2. 生成 tools_schemas，传给模型 chat template。
3. 把模型输出解析成 OpenAI-style function call，再转成 bash 命令。
```

内建工具包括：

- `execute_bash`
- `str_replace_editor`
- `submit`
- `finish`
- `search`
- `search_arxiv`
- `lark-cli`

设计上，工具调用最后都落到 sandbox shell 命令。比如：

```text
execute_bash(command="pytest ...") -> 直接执行 command
submit() -> echo '<<<Finished>>>'
str_replace_editor(...) -> str_replace_editor CLI 参数
```

## 7. AgentEnv 与 Deployment

`uni_agent/interaction/env.py` 负责统一环境操作：

```text
start()
close()
install_tools()
install_skills()
run_action()
communicate()
copy_to_container()
read_file()
write_file()
```

它不直接知道底层是 Docker、Modal 还是 veFaaS。底层由 `deployment` 决定。

`uni_agent/deployment/config.py` 定义了 deployment 类型：

```text
host:
  直接在宿主机执行，非容器。

local_native:
  通过 pexpect/PT​​Y 在宿主机执行，非容器。

local:
  Uni-Agent 自己启动本机容器，并在容器里启动 swerex.server。

local_attach:
  用户提前启动好 swerex.server，Uni-Agent 只连接 host/port/token。

modal:
  Modal sandbox 后端。

vefaas:
  火山 veFaaS sandbox 后端。
```

注意 `local` 和 `local_attach` 的区别：

```text
local:
  Uni-Agent -> 启动容器 -> 启动/连接 swerex.server -> 执行动作

local_attach:
  用户已启动容器 + swerex.server
  Uni-Agent -> 连接已有服务
```

open PR #44 修的是 `local` 模式下 Docker/Podman endpoint 选择问题：宿主机访问应该连 `published_port`，容器互访才连 sandbox container IP + `runtime_port`。

## 8. RewardSpec

reward registry 在 `uni_agent/reward/registry.py`。

当前支持：

- `search`
- `swe_bench`
- `swe_bench_multilingual`
- `swe_rebench`
- `swe_rebench_v2`
- `r2e_gym`
- `terminal_bench`
- `terminal_bench_v2`

RewardSpec 接口很小：

```python
async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple:
    ...
```

但实现里会做比较重的事情。

### SWE-Bench

`uni_agent/reward/swe_bench.py`：

- 根据 metadata 构造 eval script。
- 在 sandbox `/testbed` 里运行测试。
- 解析 SWE-Bench harness output。
- 返回是否 resolved。

open PR #62 把 `make_eval_script(metadata, workdir)` 和 `parse_eval_output(metadata, output)` 抽成 public helper。这说明 reward 正从“只给 UniAgentLoop 内部用”变成外部 sandbox 框架也能复用的库函数。

### Terminal-Bench

`uni_agent/reward/terminal_bench.py`：

- 上传 task 的 tests archive 到 `/tests`。
- 运行 `test.sh`。
- 读取 `/logs/verifier/reward.json` 或 `/logs/verifier/reward.txt`。

这类 reward 非常适合 per-sample tools_kwargs：每个 task 的测试文件、工作目录、timeout 都不同。

## 9. 并发推理与训练入口

### 并发验证

入口样例：`examples/agent_interaction/parallel_infer.py`

它做的事情：

```text
1. ray.init()
2. 从 verl config 初始化 LLMServerManager
3. AgentLoopManager.create()
4. 读取 parquet dataset
5. DataProto non_tensor_batch:
   - raw_prompt
   - agent_name
   - tools_kwargs
6. agent_loop_manager.generate_sequences()
7. 从 rm_scores 计算 mean score
```

这条路已经能跑 SWE-Bench Verified、Terminal-Bench v2 等并发验证。

### 训练

`examples/agent_train/*.sh` 基本都是把 UniAgentLoop 接到 verl fully async trainer：

```text
python -m verl.experimental.fully_async_policy.fully_async_main
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.agent.num_workers=8
  actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH}
```

训练脚本关键参数：

- `NNODES_ROLLOUT`
- `NNODES_TRAIN`
- `n_resp_per_prompt`
- `staleness_threshold`
- `trigger_parameter_sync_step`
- `partial_rollout`
- `max_prompt_length`
- `max_response_length`

设计目标是长轨迹 agent RL。因为每个 task 的 rollout 延迟差异很大，fully async 比 synchronous batch 更合理。

## 10. 黑盒路径：AgentFramework + Gateway

黑盒路径代码在：

```text
uni_agent/framework/
uni_agent/gateway/
```

它的核心目标是：外部 agent 不需要使用 Uni-Agent 的 `AgentInteraction`，只要它能调用 OpenAI-compatible API，就能被 Uni-Agent 记录成 verl 可训练的 trajectory。

基本链路：

```text
AgentFrameworkRolloutAdapter
  -> GatewayManager
  -> GatewayActor pool
  -> per-session GatewaySession
  -> external agent runner 调 session.base_url
  -> GatewaySession 调 verl backend.generate()
  -> finalize_session()
  -> Trajectory
  -> RewardLoopWorker / TransferQueue
```

`OpenAICompatibleAgentFramework` 做 batch/session 管理：

- 每个 prompt 可以跑 `rollout.n` 个 session。
- 每个 session 创建 `SessionHandle`。
- runner 可以 inline async，也可以 Ray task。
- session finalize 后得到 trajectories。
- 可选调用 RewardLoopWorker 打分。
- 最终写入 TransferQueue schema。

关键意义：

```text
白盒 UniAgentLoop:
  Uni-Agent 控制工具解析和环境执行。

黑盒 AgentFramework:
  外部 agent 自己控制工具和环境。
  Uni-Agent/Gateway 只负责模型调用、token truth、trajectory、reward。
```

## 11. Gateway 当前设计

主干 Gateway 在 `uni_agent/gateway/`。

组件：

```text
GatewayManager:
  driver 侧 actor pool 管理，按 session_id 路由。

GatewayActor:
  Ray actor + FastAPI server，暴露 /sessions/{id}/v1/chat/completions 和 reward_info。

GatewaySession:
  单个 session 的状态机，负责 message history、token buffer、trajectory materialization。

MessageCodec:
  provider request -> chat template/token ids
  token ids -> assistant message/tool_calls
```

当前主干的 `GatewaySession` 是比较线性的：

```text
message_history
active_trajectory
trajectories
```

如果新的 request 是当前 history 的前缀扩展，就增量编码；否则 materialize 当前 active trajectory，重新 full encode。

这个设计对普通 multi-turn OK，但对复杂黑盒 agent 不够：

- sub-agent 分支
- best-of-N / retry
- context compaction
- Claude Code 重复 prompt 或变体 prompt
- 同一个 session 内多条逻辑链

这就是 #65/#66 都在重构 GatewaySession 的原因。

## 12. 当前 open PR 透露的发展方向

截至 2026-07-12，GitHub open PR 有 9 个。它们高度集中在几个方向。

### 12.1 黑盒 coding agent 成为一等能力

PR:

- #73 `[examples] feat: blackbox mini-swe-agent training recipe`
- #78 `[examples]feat: add Claude Code blackbox training recipe`

共同模式：

```text
runner host
  -> 创建远端 sandbox
  -> 挂载 sidecar tool image
  -> 在 sandbox 内运行外部 agent
  -> agent 通过 Gateway 访问 rollout model
  -> sandbox 内执行任务
  -> 同 sandbox 内 reward eval
  -> reward_info 回传
```

#73 是 mini-swe-agent，#78 是 Claude Code。#78 还新增 Anthropic Messages adapter，因为 Claude Code 原生走 Anthropic 协议。

这说明 Uni-Agent 的目标不是只训练“自己写的 agent”，而是训练/验证任何能通过协议接入的 agent。

### 12.2 Gateway 是未来核心

PR:

- #65 prefix trie storage
- #66 multiple linear chains
- #76 OpenClaw online training via Gateway TurnCapture
- #78 OpenAI/Anthropic adapters

Gateway 正在从 proxy 变成核心状态层：

```text
provider protocol adapter
token truth source
trajectory materializer
multi-branch session state
online sample capture
debug launcher
```

这块后续很可能是最大重构区域。

### 12.3 GatewaySession 状态模型会重构

#65 和 #66 都在解决 single-active trajectory 的限制。

#### #65 prefix trie

思路：

```text
session 内维护 prefix trie
incoming request longest-prefix match 任意历史路径
从最近 checkpoint 继续
finalize 导出 terminal checkpoints
```

适合：

- sub-agent
- best-of-N
- context condensation
- warm-start
- retry

#65 是 gated flag：`gateway_trie_enabled=False` 默认关。

#### #66 multiple linear chains

思路：

```text
session 内维护多个 append-only linear chains
每个 chain 有自己的 message history、prefix hash、tool/schema state、trajectory buffer
选择 longest compatible active chain
```

#66 当前描述里说 multiple chains 已是默认且唯一状态模型，不再保留 single-active flag。

它还加了：

- `enable_parallel_session_generation=False`
- `ignore_cch_for_prefix_hash=False`
- Claude Code `cch=` prefix hash 兼容
- same-session backend 并发实验支持

我判断 #66 更贴近 #78 Claude Code 需求，也更可能成为最终方向；#65 像较早的 trie RFC 实现。两者都 open，后面需要看 maintainer 如何取舍或合并思想。

### 12.4 在线 client-driven training

PR #76：OpenClaw online RL/OPD/Combine training path。

它改变了数据来源：

```text
过去:
  static parquet -> rollout -> train

OpenClaw:
  external client -> embedded FastAPI proxy -> GatewaySession
  -> per-turn TurnCapture
  -> sample builder/scorer
  -> fully async training queue
```

关键点：

- training samples 可以按 turn 动态产生。
- token truth 来自 Gateway/backend token ids 和 logprobs。
- 避免 decode -> re-encode。
- 支持 RL、OPD、Combine 三种训练模式。

这把 Uni-Agent 从 benchmark runner 推向在线训练系统。

### 12.5 GUI/VLM agent

PR #49：GUI agent loop。

新增：

```text
uni_agent/gui_agent_loop.py
uni_agent/gui_utils.py
uni_agent/tools/os_sandbox_tool.py
```

模式：

```text
VLM 输入截图 + history
模型输出 CoT + pyautogui-style action
sandbox 执行动作
返回下一张截图
直到 DONE/FAIL/max_steps
```

这说明 Uni-Agent 想覆盖 terminal/code 之外的 agent 环境，尤其是 GUI RL。

### 12.6 稳定性和大规模训练容错

PR #50：训练失败路径安全网。

背景是大规模 Qwen3-235B SWE-Bench campaign 暴露：

- 单条 trajectory failure 不能 kill rollouter。
- MoE `routed_experts` shape 要和正常样本一致。
- `_build_empty_agent_output()` 自己也可能失败，需要 Layer-2 minimal output。

这类 PR 说明项目已经经历过大规模生产训练，不只是 demo。

## 13. 我对后续架构演进的判断

### 13.1 Gateway 会成为 repo 的中心，而不是 interaction

当前主干最成熟的是白盒 `UniAgentLoop` + `AgentInteraction`。但 open PR 的重心明显在 Gateway：

- Anthropic/OpenAI adapters
- multi-chain session state
- TurnCapture
- blackbox recipes
- debug launcher

未来可能会形成：

```text
AgentInteraction:
  内建白盒 agent 的一个 runner。

Gateway:
  所有黑盒/在线/复杂 agent 的统一模型访问和轨迹记录层。
```

### 13.2 黑盒 recipe 会被抽象出公共 runner/sandbox 层

#73 和 #78 都有类似文件：

```text
Dockerfile.*-tool
build_tool.sh
runner.py
dataset.py
reward.py
parallel_infer.py
run_train.sh
sandbox_client.py
```

现在放在 `examples/blackbox_recipes/`，但重复结构明显。后续可能抽：

```text
uni_agent.blackbox
uni_agent.sandbox_sidecar
uni_agent.gateway.adapters
```

### 13.3 Reward 会从 class plugin 变成可复用 helper 库

#62 是信号。外部框架只想复用 SWE-Bench eval script 和 parser，不一定想实例化 `SWEBenchRewardSpec`。

后续 reward 可能拆成两层：

```text
pure helpers:
  make_eval_script()
  parse_eval_output()
  build_reward_context()

runtime spec:
  compute_reward(env, metadata)
```

### 13.4 Session trajectory 不再是“一次会话一条线”

黑盒 agent 很容易出现：

- 多个工具 schema 变体
- 子代理分支
- retry 重放
- compaction
- 相同 prefix 多次 continuation
- 同 session 并发请求

所以 trajectory materialization 必须支持多分支。#65/#66 都是围绕这个。

### 13.5 fully async 会继续是训练主线

agent rollout 时间高度不均匀：

- sandbox cold start
- package install
- long tests
- command timeout
- multi-turn exploration

因此 synchronous batch 会被长尾拖慢。open PR #76 也明确在 fully async 上加在线样本路径。后续训练 recipe 大概率继续围绕 fully async，而不是 sync PPO。

## 14. 推荐阅读顺序

如果后续继续读源码，我建议按这个顺序。

### 第一轮：理解白盒 agent

```text
uni_agent/agent_loop.py
uni_agent/interaction/interaction.py
uni_agent/interaction/model.py
uni_agent/interaction/tools_manager.py
uni_agent/interaction/env.py
examples/agent_interaction/parallel_infer.py
examples/agent_interaction/agent_config_modal.yaml
```

目标：掌握 `raw_prompt + tools_kwargs -> AgentLoopOutput`。

### 第二轮：理解环境和 reward

```text
uni_agent/deployment/config.py
uni_agent/deployment/local/deployment.py
uni_agent/deployment/modal/deployment.py
uni_agent/reward/swe_bench.py
uni_agent/reward/terminal_bench.py
examples/data_preprocess/swe_bench_verified.py
examples/data_preprocess/terminal_bench_v2.py
```

目标：掌握 per-sample sandbox/reward metadata 为什么能支撑多个 benchmark。

### 第三轮：理解黑盒 Gateway

```text
uni_agent/framework/entry.py
uni_agent/framework/framework.py
uni_agent/gateway/manager.py
uni_agent/gateway/gateway.py
uni_agent/gateway/session/session.py
uni_agent/gateway/session/codec.py
uni_agent/gateway/session/types.py
```

目标：掌握外部 agent 如何通过 OpenAI-compatible API 被转成 trajectory。

### 第四轮：看 open PR 的未来方向

按优先级：

```text
#66 Gateway multiple linear chains
#78 Claude Code blackbox recipe
#76 OpenClaw online RL
#73 mini-swe-agent blackbox recipe
#49 GUI agent loop
#62 reward helpers
#50 failure safety
#44 local deployment stability
```

其中 #66/#78/#76 最能代表未来架构。

## 15. 可能的设计风险

### 15.1 Gateway 状态模型尚未收敛

#65 和 #66 都在改 GatewaySession 多轨迹存储，方案不同。如果都长期 open，说明 maintainers 还在权衡：

- trie 更通用，但实现复杂。
- multiple linear chains 更直观，更贴近 Claude Code 场景。

这块 merge 后可能造成主干 Gateway 大改。

### 15.2 Reward 广播策略对多分支 session 可能不够精细

当前黑盒 framework 的 `_score_trajectories()` 是 score 最后一条 trajectory，然后 broadcast 给 session 所有 trajectories。

多分支之后，这个策略可能过粗：

```text
best-of-N sibling chains
parallel branch
sub-agent branch
```

这些分支是否应该共享同一个 reward，需要按任务定义重新考虑。

### 15.3 黑盒 agent 的 protocol fidelity 很难

Claude Code 走 Anthropic Messages，会涉及：

- `tool_use`
- `tool_result`
- `system` top-level vs mid-list system reminders
- streaming response
- cache/billing header
- thinking/redacted_thinking
- image block

#78 已经做了大量 adapter 测试，但 provider 兼容会是长期维护成本。

### 15.4 sidecar image 和 sandbox 后端绑定较强

#73/#78 目前都依赖远程 sandbox、sidecar tool image、upstream tunnel。不同云后端的 mount、network、auth、cleanup 能力不同，后续如果要通用化，需要更清晰的 sandbox sidecar abstraction。

### 15.5 大规模训练 failure path 是核心质量指标

#50 说明一条失败样本如果 shape 不一致，就能拖垮整个 batch/rollouter。agent RL 的稳定性不只在模型和 reward，也在：

- token shape
- routed experts
- env close
- tokenizer partial init
- reward eval timeout
- logprob completeness

这些细节决定能否长时间跑 fully async。

## 16. 最后总结

当前 Uni-Agent 的定位可以分成三层：

```text
1. Agent runtime
   tools/env/skills/model/reward，负责让 agent 在真实 sandbox 里行动。

2. Trajectory infrastructure
   Gateway/AgentLoop/rollout_cache/TransferQueue，负责把多轮交互变成可训练 token trajectory。

3. Training integration
   verl fully async / GRPO / OPD / Combine / reward managers，负责规模化 RL。
```

主干已经具备白盒 agent 验证和训练能力；open PR 则显示未来重点是：

- 黑盒 agent 接入。
- Gateway 多协议、多分支、多轨迹。
- 在线 client-driven training。
- GUI/VLM sandbox-as-environment。
- reward 和 deployment 的公共化、稳定化。

如果要跟进这个项目，最值得盯的不是单个 benchmark 分数，而是 GatewaySession 的最终状态模型，以及黑盒 recipe 是否会从 examples 沉淀成正式 API。

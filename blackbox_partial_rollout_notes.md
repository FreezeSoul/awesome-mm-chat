# 黑盒 Agent Partial Rollout 机制与风险分析

本文记录对 Dressage 黑盒 agent partial rollout 实现的源码分析，重点解释它到底能暂停/恢复什么，以及在闭源或强超时 agent 上的风险边界。

## 结论摘要

Dressage 的黑盒 partial rollout 更准确地说是 **model-generation partial preemption**，不是完整的 **agent-execution preemption**。

它能做的是：

- 在黑盒 agent 的一次 LLM HTTP 调用还未返回时，中断下游 SGLang `/generate`。
- 收集已经生成的 partial tokens。
- 权重更新后，用 `原 prompt tokens + partial output tokens` 重新请求剩余生成。
- 把多个 generation chunk 拼成一次完整 OpenAI-compatible response 返回给 agent。
- 记录 token 级 weight version，并可 mask 掉非最后版本生成的训练 token。

它不能保证的是：

- 黑盒 agent 进程本身可以被暂停和恢复。
- 黑盒 agent 内部 HTTP client 不会因为长时间无响应而超时。
- streaming 请求在 pause 期间有 heartbeat。
- agent 正在执行工具调用、shell 命令、测试、文件操作时可以被 abort。
- 任意闭源 agent 都能安全支持这种透明挂起。

## 黑盒 Agent 的调用链路

黑盒 agent 并不直接访问 SGLang。实际链路是：

```text
blackbox agent process
  -> BlackboxServer 内部 RolloutLLMProxy
    -> Dressage proxy /v1/chat/completions
      -> SGLang /generate
```

`RolloutLLMProxy` 会把 agent 的 LLM provider base URL 指到本地代理，并给每次 chat completion 注入：

- `X-Session-Id`
- `X-Instance-Id`
- `X-Turn-Id`
- `X-Dressage-Partial-Rollout`
- `X-Dressage-Expected-Version`，如果有当前版本

因此 agent 自己并不知道调用被 Dressage 记录和控制。

## Pause/Resume 的实际流程

权重更新前，训练入口会调用 Dressage proxy 的 pause 接口。

```text
trainer 准备 update weights
  -> POST /v1/rollout/pause
  -> GenerationController 标记 paused
  -> abort active SGLang /generate request
  -> 等待 active generation quiesced
  -> trainer 执行 weight update
  -> POST /v1/rollout/resume
  -> GenerationController 允许继续生成
```

如果此时 agent 正在等待一次 LLM 响应，agent 到本地 proxy 的 HTTP 请求并不会返回。中断发生在更下游的 SGLang request。

恢复时，Dressage proxy 用已经收集到的 partial tokens 作为前缀重新发起 generation：

```text
new_input = original_input_ids + generated_ids_so_far
remaining_max_new_tokens = original_max_new_tokens - len(generated_ids_so_far)
```

最后返回给黑盒 agent 的是一次拼接后的完整响应。agent 看到的现象只是这次 LLM 调用变慢了。

## Token Version 与 Loss Mask

每个 SGLang response chunk 会带一个 weight version。Dressage 对 chunk 内所有输出 token 填同一个 version：

```text
chunk v0: t1 t2
chunk v1: t3 t4

full_versions: v0 v0 v1 v1
```

finalize trajectory 时，proxy 会把这些 version 写入 segment 的 `full_versions`。上下文 token 使用 `-1` 作为 sentinel。

如果启用 `--mask-nonlast-version-tokens`，样本构造时会只保留最后一个真实版本生成的可训练 token。例如：

```text
full_loss_mask: [0, 1, 1, 0, 1]
full_versions:  [-1, v0, v1, -1, v1]

response_start 之后的训练 loss_mask:
                [0, 1, 0, 1]
```

原始 `full_versions`、`version_spans`、`dressage_start_token_version`、`dressage_end_token_version` 会保留在 metadata 里，便于审计。

## Stream Idle 风险

这是黑盒 partial rollout 最大的不确定性之一。

源码里 Dressage proxy 对 `stream=True` 并不是边从 SGLang 收 token 边转发给 agent。它先等待：

```python
router_response = await generation_controller.generate_preemptible(...)
```

等整个 generation 完成后，再用 `_pseudo_stream_chunks()` 把完整结果切成 SSE chunk 返回。

因此，pause 期间 agent 侧可能出现：

- HTTP 请求已发出，但长时间没有 response body。
- streaming 请求长时间没有 SSE event。
- 没有 heartbeat 或 keepalive chunk。

如果黑盒 agent 内部有不可配置的 read timeout、stream idle timeout、watchdog 或 retry 策略，就可能：

- 报错退出当前 turn。
- 重试 LLM 请求，导致重复请求或状态混乱。
- 标记 session desynced。
- 直接退出 agent 进程。

Dressage 自己做了一些缓解：

- `RolloutLLMProxy` 的 httpx client 使用无限 timeout。
- BlackboxServer 等待 backend turn 时会排除 pause 时间。
- opencode adapter 会把 provider timeout 配成 `router_timeout`。
- openclaw adapter 侧请求 gateway 时使用 `stream=False`，并对 adapter 到 gateway 的请求设置 `timeout=None`。

但这些只能保证 Dressage 自己不先超时，不能保证外部或闭源 agent 内部没有独立 timeout。

## 工具调用期间不能 Abort

如果 agent 正在执行工具调用，比如 shell、测试、构建、文件操作、浏览器操作等，此时通常没有 active SGLang generation。

这种情况下 pause 的效果是：

```text
agent 正在跑工具
  -> GenerationController 没有 active generation 可 abort
  -> proxy 标记 paused
  -> trainer 可继续 weight update
  -> agent 工具自然结束
  -> agent 下一次请求 LLM 时，如果仍 paused，则被挂起
```

也就是说，工具执行本身不会被中断。

风险包括：

- 长工具调用期间，系统并没有真正停在 agent 可恢复点。
- 工具可能占用资源、修改 sandbox 文件、卡住或超时。
- 工具结束后下一次生成会使用新权重，轨迹语义跨版本。
- 如果工具调用挂死，partial rollout 无法挽救，只能依赖工具 timeout、sandbox timeout 或 abort/retry。

因此，`active_sglang_generations == 0` 只能说明模型侧静默，不能说明 agent idle。

## 哪些黑盒后端相对可控

仓库中已实现的黑盒后端是：

- `opencode`
- `openclaw`

`claude_code` 只是保留名，未实现。

`opencode` 和 `openclaw` 更像“外部 agent 框架接入”，不是绝对黑盒。因为可以控制配置、网关、启动参数，甚至 patch 源码，所以可以一定程度上处理 provider timeout 或 stream 行为。

闭源 agent，例如 Claude Code 这类源码不可见的 CLI，风险更高：

- 内部 timeout 可能不可配置。
- stream idle 行为可能不可控。
- 失败重试可能破坏 turn attribution。
- agent 内部状态无法 checkpoint 或恢复。

对这类 agent，Dressage 的 token-level partial rollout 不能无条件保证可靠。

## 生产使用建议

如果要把黑盒 partial rollout 用在真实训练中，建议至少做以下验证：

- 启动真实 opencode/openclaw 进程，而不是 mock。
- 在一次 LLM 生成中强制 pause 60s、300s、900s，确认 agent turn 能继续完成。
- 检查 pause 期间是否出现 stream idle、read timeout、session desync。
- 在 agent 执行长工具调用时触发 weight update，确认工具 timeout 和 sandbox 状态可控。
- 记录并监控：
  - pause duration
  - active SGLang generations
  - suspended generations
  - blackbox session desync rate
  - aborted/no-grad sample rate
  - retry rate

更保守的策略是：

- 对闭源 agent 禁用 token-level in-flight partial resume，只使用普通 async rollout。
- 对长工具调用设置硬 timeout。
- 对 pause 后失败的 trajectory 标记 aborted/no-grad，重新开新实例。
- 把权重更新窗口尽量压短，避免长时间 HTTP 挂起。

## 总体判断

Dressage 的实现并不是简单 demo，核心 proxy、version tracking、masking、partial chunk stitch 都有比较完整的工程路径。但它的可靠性依赖一个重要前提：

**agent 到 LLM provider 的请求可以被透明挂起足够久，且 agent 不会因为无响应而失败。**

这个前提对可控的 opencode/openclaw 可以通过配置和 patch 尽量满足；对真正闭源黑盒 agent 则不能默认成立。

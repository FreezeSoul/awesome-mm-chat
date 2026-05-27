# MiniMax-M2 Paper Notes

## 总述

MiniMax-M2 系列这篇报告最有价值的部分，不是提出了一个特别新的 Transformer 架构，而是把 agent 训练所需的任务、环境、数据、验证器和 RL 系统做成了相对完整的工程闭环。

它的核心判断是：下一阶段模型能力不只是继续扩大参数量，而是要让模型在真实工作流中完成长程、多工具、多 artifact 的任务。为了做到这一点，M2 走的是一条很明确的路线：

```text
低激活 MoE 主干
    + 可验证 agent 数据环境
    + 面向长轨迹的 Forge RL 系统
    + 推理/训练效率优化
    + 模型参与模型工程迭代
```

这篇报告最硬的资产是数据和环境。它不是只合成 prompt-response，而是为多类真实任务构造：

```text
task spec + executable workspace + tools + verifier + reward
```

这件事很难，尤其是在规模化之后。SWE 任务要有真实 repo、依赖、Docker 环境和测试；AppDev 要能部署、交互、检查视觉质量；Terminal 任务要有可执行环境和脚本验证；Office、Excel、Finance、Slides 要能围绕最终 artifact 做判分。相比单纯依赖 LLM-as-a-judge，这种 reward 更贴近真实交付物。

因此我对这篇报告的整体判断是：它的架构创新不是最强，但它非常清楚地展示了 agent scaling 的核心资产在哪里，即 **谁能把真实任务变成可验证训练环境，谁就掌握了 agent 能力提升的关键杠杆**。

## 1. 模型主干：小激活量服务长程 agent

M2 是一个 MoE 模型，旗舰版本总参数为 229.9B，但每 token 只激活约 9.8B 参数。这个设计在 agent 场景里尤其重要，因为一次 agent 任务往往会触发大量模型调用。如果每一步都用高激活量 dense 模型，训练 rollout 和线上推理成本都会迅速放大。

模型架构上，它采用 62 层 decoder-only Transformer，MoE feed-forward 层包含 256 个细粒度 experts，每 token 激活 8 个 experts。注意力部分则比较保守，采用 full attention + GQA，并支持 192K 原生上下文。

比较值得注意的是，报告里明确说他们试过 hybrid sliding-window attention 等方案，但长上下文、检索、多跳推理和 agent 任务表现会受影响。因此 M2 的取舍是：

```text
FFN 用 MoE 降低每 token 计算量
Attention 保留 full attention 保证长上下文质量
```

这个选择很务实。对于 agent 来说，长上下文里保留任务状态、工具观察、代码片段、日志和中间结论非常关键。相比进一步压缩 attention 计算，MiniMax 更愿意先保证质量。

## 2. 数据与可验证环境：最值得关注的贡献

M2 的 post-training 数据覆盖非常广，主要包括 agentic coding、AppDev、Terminal、Deep Search、Office、Finance、Spreadsheet、Slides、Reasoning、General Conversation 和 Role-play。

这里最重要的不是任务类别多，而是每一类都尽量构造可执行、可验证的训练环境。

### 2.1 Agentic Coding

Coding 数据覆盖 SWE、AppDev 和 terminal interaction。

SWE 部分从 GitHub PR、issue、diff、测试中构造任务。一个合格样本不只是自然语言描述，还包含：

- 可运行的 Docker 环境；
- 问题描述；
- golden patch 或参考行为；
- F2P / P2P 测试；
- task-specific reward。

报告里强调了多语言环境构造的难点。Python 项目相对容易，但 Java、Go、Rust、C++、JavaScript 等生态有不同的构建系统、依赖管理、测试框架和错误格式。因此他们使用 agent-driven execution loop 来生成和修复 build scripts。

这部分的价值在于，它把真实软件工程任务转成了可训练 RL 环境，而不是只做代码补全或单文件修复。

### 2.2 AppDev

AppDev 关注从零构建完整应用。它的难点是传统单元测试不足以覆盖质量，因为应用还涉及交互、部署、视觉和 UX。

报告提出 Agent-as-a-Verifier，用多层检查做 rejection sampling：

- Execution Layer：检查文件、依赖、构建、服务启动、页面错误；
- Interaction Layer：用 Playwright 检查按钮、表单、核心流程；
- Visual Aesthetics Layer：检查布局、视觉层级、配色和现代 UI 质量。

这个思路很好，因为它把“应用是否真的可用”作为 reward，而不是只看代码文本是否像样。

### 2.3 Terminal-Gym

Terminal-Gym 从 Stack Overflow 等真实问题出发，筛选 terminal-compatible、scriptable、Docker-relevant、verifiable 的任务，再生成环境和测试脚本。

它的核心流程是：

```text
真实问题筛选
 -> 结构化任务描述
 -> Docker 环境生成
 -> 测试脚本生成
 -> query evolution
 -> difficulty calibration
```

这类数据对 agent 很重要，因为真实工作中大量任务不是“写一段代码”就结束，而是要在 shell 里调试环境、安装依赖、运行命令、定位系统问题。

### 2.4 Cowork / Office / Finance

这部分覆盖深度搜索、知识工作、金融分析、Excel 操作、slide 生成等任务。

它的共同点是每个任务都围绕最终 artifact 设计 reward：

- 搜索任务要求答案基于实际检索证据；
- Office 任务检查最终报告、memo、deck 是否符合要求；
- Spreadsheet 任务可以重算公式并比较 cell values；
- Finance 任务结合工具结果、工作簿和专家 rubric；
- Slides 任务渲染成图像后检查视觉和内容质量。

这部分说明 MiniMax 对 agent 的理解很接近真实生产：agent 最后交付的不是一段回答，而是一个可检查的文件、表格、网页、报告或代码修改。

## 3. MTP Expansion via Weight Copying

M2 在预训练中使用 Multi-Token Prediction，既提供额外训练信号，也为推理阶段 speculative decoding 服务。

预训练阶段先使用一个 MTP module。到 continued pre-training 的 decay phase，为了支持 multi-step speculative decoding，将 MTP 从一个 module 扩展到三个 modules。新增 MTP modules 不是随机初始化，而是从已有的 MTP module 复制权重。

```text
K=1 MTP 预训练
 -> 将已有 MTP module 权重复制给新增 MTP modules
 -> freeze main model，warm up MTP modules
 -> MTP loss 稳定后 joint training
 -> 推理时三个 MTP modules 生成 draft tokens，由 main model 验证
```

这部分的工程意义在于，MTP 不是一个孤立的辅助 loss，而是训练和推理一体化设计。对 agent 来说，推理调用次数很多，speculative decoding 带来的吞吐收益会被多轮调用持续放大。

## 4. Forge：面向 agent 长轨迹的 RL 系统

Forge 是报告里最有工程含量的系统。它不是单纯的 RL 算法，而是一套支持大规模 agent RL 的基础设施。

普通 RLHF 通常是：

```text
prompt -> response -> reward -> update
```

Agent RL 则是：

```text
task
 -> model action
 -> tool call
 -> observation
 -> model action
 -> file edit / shell / browser / sub-agent
 -> ...
 -> final artifact
 -> verifier reward
```

一个 episode 可能包含几十轮到上千轮交互，轨迹长度、耗时、工具类型和 reward 形式都高度异构。Forge 主要解决三个目标之间的冲突：

- 吞吐；
- 训练稳定性；
- agent scaffold 灵活性。

### 4.1 建模方式

Forge 把 LLM 看成 policy，把模型外部的所有东西都看成 environment，包括工具执行、上下文管理、memory、sub-agent 和 agent harness。

每一步可以写成：

```text
s_t -> a_t -> o_t -> s_{t+1}
```

其中 `s_t` 是当前给模型的上下文，`a_t` 是模型输出，`o_t` 是工具或环境观察。这样，一条长 agent 轨迹可以拆成多个 `(s_t, a_t)` 训练样本，同时 reward 可以从完整 episode 回传。

这个边界很关键，因为训练系统不必绑定某一种 agent 实现。

### 4.2 三层系统架构

Forge 分为三层：

```text
Agent Side
    |
Gateway Server / Data Pool
    |
Training & Inference Side
```

Agent Side 负责执行真实任务，包括调用工具、运行环境、管理上下文、写文件、跑测试等。

Gateway Server 把不同 agent 的 completion 请求统一转发给 rollout engine。Data Pool 异步收集轨迹，供训练侧消费。

Training & Inference Side 包括 Rollout Engine 和 Train Engine。Rollout Engine 负责高吞吐生成，Train Engine 消费轨迹、计算 policy gradient、更新权重，并同步给 rollout engine。

这个设计把 agent rollout 和 training 解耦，有利于大规模异步训练。

### 4.3 白盒与黑盒 agent

Forge 同时支持 white-box 和 black-box agent。

White-box agent 会暴露上下文管理逻辑，训练系统可以复现 truncate、summarize、memory injection 等操作，构造更精确的训练状态。

Black-box agent 不暴露内部机制。Forge 只记录每次模型调用时实际看到的上下文和模型输出。这种方式对生产系统很重要，因为真实 agent scaffold 可能非常复杂，不适合为了 RL 重新实现。

这个抽象让 Forge 能支持大量不同 agent，而不是绑定某一种工具协议。

### 4.4 Windowed FIFO

Agent rollout 的耗时差异非常大。严格 FIFO 会被长任务卡住，完全 greedy 又会让训练 batch 偏向短任务和简单任务，造成分布漂移。

Windowed FIFO 的思路是：

```text
只允许在滑动窗口内贪心取已完成任务
窗口外即使完成也不能提前进入训练
```

这样在窗口内减少 head-of-line blocking，在全局上保留接近 FIFO 的分布稳定性。它是一个很实用的系统折中。

### 4.5 Prefix Tree Merging

多轮 agent 轨迹有大量共享前缀。传统训练会重复计算这些前缀：

```text
sample 1: A B C D
sample 2: A B C E
sample 3: A B C F
```

Prefix tree merging 会把它们变成：

```text
      A B C
     / |  \
    D  E   F
```

共享前缀只 forward 一次，分支部分分别计算。由于 causal attention 下前缀 hidden states 不依赖后续 token，这个方法在数学上等价于独立样本训练，不是近似优化。

报告称该方法最高可带来 40x training speedup。对于长上下文、多 rollout、多 turn agent 训练，这个优化非常关键。

### 4.6 推理侧优化

Forge 还对 rollout 推理做了优化：

- MTP-based speculative decoding：用 MTP draft tokens，主模型验证；
- Prefill / Decode disaggregation：把长上下文 prefill 和 decode 拆开调度；
- Global L3 KV Cache Pool：利用多轮 agent 的共享前缀，提高 KV cache 命中率。

这些优化说明 Forge 不是只关注训练算法，而是把 rollout 成本、缓存、调度和 MoE 推理负载一起考虑。

### 4.7 Reward 与 mixed-domain RL

Forge 使用 CISPO 做 policy optimization，并设计了复合 reward：

```text
r_t = alpha * process_reward
    + beta * speed_reward
    + performance_reward
```

Performance reward 来自最终任务结果，如测试通过、artifact 合格、答案正确。Process reward 约束中间行为，如工具调用格式、语言混乱、步骤结构。Speed reward 鼓励更快完成任务。

此外，训练不是单一 agent 域，而是混合 reasoning、coding、agent 和 general 数据，避免只优化 agent 行为导致基础能力退化。

## 5. Self-Evolution：模型参与模型研发流程

M2.7 的 Self-Evolution 不是指模型完全自主修改权重并训练下一代模型，而是指模型开始参与模型研发工程本身。

报告里的 Self-Evolution 主要包括两层：

```text
训练/实验迭代自动化
    + agent scaffold 自我改进
```

在训练和实验迭代中，M2.7 可以读取日志、分析 metric anomaly、诊断训练失败、修改配置、调试训练代码，并在 human review 之间继续做 bounded analysis。报告称这吸收了 RL 团队 30% 到 50% 的日常迭代工作量。

在 scaffold 自我改进中，M2.7 会分析内部 agent scaffold 的失败案例，修改代码或参数，重新评测，再进入下一轮。报告提到它做过 100 轮 autonomous iteration，引入 loop detection，并找到更好的参数组合，使内部评测提升约 30%。

这部分的意义不是“模型已经能完全自我进化”，而是模型开始承担模型研发中的大量工程迭代工作：

```text
发现异常
 -> 查日志
 -> 猜原因
 -> 改配置/代码
 -> 跑实验
 -> 汇总结果
```

如果这类流程能稳定自动化，会直接缩短下一代模型的研发周期。

但这部分也要谨慎看：

- 它仍然是 human-in-the-loop；
- 主要改的是 scaffold、配置和工程代码，不是完全自动改权重训练目标；
- 内部评测和内部 scaffold 难以外部复现；
- 报告没有充分披露失败案例、审计机制和长期 memory 污染问题。

## 6. 总结

MiniMax-M2 的核心价值在于，它把 agent 训练从“模型回答问题”推进到“模型在可执行环境中完成真实工作”。

它最值得学习的不是某个单点技术，而是完整系统思路：

```text
低激活 MoE 降低 agent 调用成本
可验证环境提供可靠 reward
Forge 支撑长轨迹 RL
MTP 和缓存优化 rollout 吞吐
Self-Evolution 自动化模型工程迭代
```

如果要用一句话概括：**M2 系列展示的是 agent system scaling，而不是单纯 model scaling。**
# OPD 原理和 Logprob 对齐问题

官方中文文档：[`docs/zh/advanced/on-policy-distillation.md`](../docs/zh/advanced/on-policy-distillation.md)

## OPD 在 slime 里的核心逻辑

OPD，也就是 on-policy distillation，是在 RL 的 on-policy 训练样本上额外加入 teacher 约束。学生模型先按当前 policy rollout 出样本，然后 teacher 对同一段 token 序列打分，得到 response token 级别的 `teacher_log_probs`。训练时再用 student 当前 logprob 和 teacher logprob 的差值修正 advantage。

当前 slime 里的实际形式可以理解为：

```text
reverse_kl = student_logp - teacher_logp
advantage = advantage - opd_kl_coef * reverse_kl
```

这里的 OPD 项是叠加在已有 advantage estimator 上的。GRPO、PPO、REINFORCE++ 等先算出原始 advantage，然后 OPD 再按 token 级 reverse KL 做惩罚。它不是替代 reward，也不是替代 advantage estimator。

直观上：

- 如果 student 给某个已生成 token 的概率比 teacher 更高，`student_logp - teacher_logp` 为正，advantage 被减小。
- 如果 student 给某个已生成 token 的概率比 teacher 更低，惩罚项相对更小，甚至会抬高该 token 的学习信号。
- 目标是让 student 在自己 on-policy 采出来的轨迹上，逐步贴近 teacher 的 token 分布。

## 当前两种 OPD 模式

### SGLang teacher 模式

配置形态：

```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
--rm-url http://<teacher-host>:<teacher-port>/generate
```

流程是：

1. rollout engine 生成 student 的样本。
2. reward 阶段把 `sample.tokens` 发给外部 SGLang teacher server。
3. teacher server 返回 token logprob。
4. `post_process_rewards` 从 SGLang 返回格式里取出 logprob，裁剪到 response span。
5. 结果写到 `sample.teacher_log_probs`。
6. rollout data 打包训练时把 `teacher_log_probs` 传给 Megatron 训练侧。
7. loss 里使用这些 teacher logprob 修正 advantage。

这个模式的 teacher logprob 是在 rollout/reward 阶段提前算好的。优点是 teacher 和训练解耦，可以单独部署、单独扩容，也可以是和 student 不同架构或更大的模型。

### Megatron teacher 模式

配置形态：

```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_megatron_checkpoint
```

流程是：

1. 初始化 Megatron train actor 时额外加载 teacher checkpoint。
2. 训练 step 里，在计算 advantage 前切到 teacher 权重。
3. 对当前 rollout data 做一次 teacher forward，得到 `teacher_log_probs`。
4. 再切回 actor/student 权重，算 student logprob 和训练 loss。
5. loss 里使用现场算出的 teacher logprob 修正 advantage。

这个模式的 teacher logprob 是在训练阶段用 Megatron 现场算的。优点是 teacher logprob 和训练侧 student logprob 使用同一套 Megatron forward、mask、token shift、并行切分和 logprob 计算语义。

## 我关心的问题：哪个 logprob 更准

这个问题不能简单说 SGLang 或 Megatron 哪个“天然更准”。更准确的判断标准是：

> OPD 里的 KL 项应该尽量比较同一数值语义下的 student 分布和 teacher 分布。

slime 当前默认情况下，OPD 里用来训练的 `student_logp` 是 Megatron training actor 重新 forward 算出来的，而不是 rollout engine 的 logprob。除非显式打开 `--use-rollout-logprobs`，否则 KL 项里的 student 侧就是 Megatron 数值体系。

因此，对 OPD loss 本身来说，teacher logprob 更应该和 Megatron training 侧对齐，而不是优先和 rollout engine 对齐。

原因是 OPD 的惩罚项是：

```text
student_logp_training_engine - teacher_logp
```

如果 `student_logp` 来自 Megatron，而 `teacher_logp` 来自 SGLang，那么这个差值里可能混入两类东西：

1. student 和 teacher 真实分布差异。
2. Megatron 和 SGLang 在 logprob 计算上的实现差异。

第二类差异不是算法想优化的目标。它可能来自：

- tokenizer 或 chat template 不一致；
- logits shift 位置不一致；
- attention mask 或 packed sequence 处理不一致；
- padded vocab / vocab size 处理不一致；
- bf16/fp16/fp32 logsoftmax 细节差异；
- fused kernel、量化、prefix cache、serving 优化；
- HF checkpoint 和 Megatron checkpoint 转换时的权重映射差异。

所以如果 teacher 和 student 架构兼容、teacher checkpoint 能在 Megatron 里跑，并且资源允许，Megatron teacher 模式在训练 loss 上通常更自洽。

## rollout 一致性和 OPD 的关系

SGLang 引擎和 rollout 一致，这对 behavior policy、采样轨迹、rollout logprob、importance ratio 这类东西很重要。

但 OPD teacher logprob 的角色不一样。teacher 不是产生这些 rollout tokens 的行为策略，它只是对已经生成好的 tokens 做打分。OPD 项最终作用在训练侧 advantage 上，所以它更关心和训练侧 `student_logp` 的可比性。

因此：

- 讨论 rollout behavior policy 时，和 rollout engine 一致很重要。
- 讨论 OPD teacher KL 时，和 training engine 的 student logprob 一致更重要。

## 什么时候选哪种

优先选 Megatron teacher 模式的情况：

- teacher 和 student 架构一致或高度兼容；
- teacher checkpoint 是 Megatron 格式；
- 训练资源能承受额外 teacher forward；
- 目标是让 OPD loss 数值最自洽；
- 希望减少 SGLang/Megatron 实现差异带来的 KL 噪声。

优先选 SGLang teacher 模式的情况：

- teacher 太大，不适合加载到训练 actor；
- teacher 只方便以 serving 形式部署；
- teacher 架构和 student 不同；
- 希望 teacher 独立扩容或被多个任务复用；
- 目标是蒸馏某个外部 serving teacher 的实际行为。

## 新增 Megatron server 的位置

`slime/backends/megatron_utils/server/megatron_server.py` 看起来是在补一种中间形态：

> 外部 teacher server，但底层用 Megatron 而不是 SGLang。

它的价值是同时获得：

- 外部服务化，teacher 不必塞进训练 actor；
- Megatron checkpoint 和 Megatron 并行能力；
- 更接近 Megatron training 侧的 logprob 计算语义。

不过当前 `slime.rollout.on_policy_distillation.post_process_rewards` 主要解析的是 SGLang 返回格式。Megatron server 返回的是自己的 `log_probs` 格式，所以如果要把它接进现有外部 OPD reward 流程，还需要一个适配 Megatron server 返回格式的 reward/post-process 函数。

## 实用判断

如果只是从算法和数值自洽角度判断：

```text
Megatron student logprob + Megatron teacher logprob
```

通常比

```text
Megatron student logprob + SGLang teacher logprob
```

更干净。

但如果业务目标是“蒸馏某个线上 SGLang teacher 的行为”，那 SGLang teacher logprob 反而是目标本身，不应该强行换成 Megatron。

最稳的做法是抽样同一批 token 序列，同时用 SGLang teacher 和 Megatron teacher 算 logprob，比较：

- response token 对齐是否一致；
- mean absolute diff；
- max diff；
- 按长度分桶后的 diff；
- 特殊 token、padding token、EOS 附近是否异常。

如果差异很小，两条路线都可用；如果差异明显，OPD loss 里优先相信和训练侧 student logprob 同实现的 Megatron teacher。

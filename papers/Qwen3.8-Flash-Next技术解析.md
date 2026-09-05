# Qwen3.8-Flash-Next 技术解析：Qwen4 前瞻架构、QSA、门控残差与低成本容量扩展

> 整理日期：2026-08-27  
> 主要资料：[Qwen 官方博客](https://qwen.ai/blog?id=qwen3.8-flash-next)、[技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)、[Hugging Face 公开配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) 与 Transformers 参考实现  
> 本地资料：[技术报告 PDF](./qwen3.8_flash_next_tech_report.pdf)  
> 说明：文中的 Benchmark、`7.6×`、`4.9×`、`8.6×` 等结果均为 Qwen 官方报告。参数量、KV Cache 和训练 FLOPs 部分包含依据公开配置所做的近似复算，已与官方口径分开标注。

## 1. 全局概览

### 1.1 一句话定位

Qwen3.8-Flash-Next 是一个原生图文/视频多模态、超稀疏 MoE 模型，也是 Qwen 团队提前公开的 **Qwen4 架构预览版**。它用 `125B` 主模型参数、额外 `51B` N-gram Embedding 和约 `6B` 每 token 激活参数，尝试在接近上一代 `397B-A17B` 旗舰 Qwen3.7-Plus 能力的同时，把训练和长上下文推理成本压到更低。

如果把它只理解为“更小的 Qwen3.8 MoE”，会遗漏真正的重点。它同时改变了四条主线：

1. **Attention**：`3 × Gated DeltaNet + 1 × Qwen Sparse Attention` 的混合主干；
2. **Residual**：把单路 residual stream 扩成 4 路，并用逐元素动态门控读、逐分支门控写；
3. **Embedding**：在第 2 层注入 51B 参数的二元组/三元组查表记忆；
4. **Optimization**：用 Muon、重新拟合的 scaling law 和稳定性压力测试共同确定训练 recipe。

它更适合高调用量的 Agent、编码、办公自动化和超长上下文服务；Qwen3.8-27B 则更适合单机部署、参数全部参与计算且部署形态更简单的场景。

### 1.2 核心规格

| 项目 | Qwen3.8-Flash-Next |
|---|---:|
| 模型类型 | 原生图像/视频多模态稀疏 MoE |
| 主模型总参数 | 125B |
| N-gram Embedding | 额外约 51B |
| 每 token 激活参数 | 约 6B |
| 文本主干层数 | 48 |
| Hidden size | 2560 |
| Token mixer | 36 层 GDN + 12 层 QSA |
| 层排列 | 12 组 `GDN × 3 → QSA × 1` |
| QSA Query / KV heads | 24 / 2 |
| QSA Head dim | 256 |
| QSA Indexer | 4 个 Query heads、1 个共享 Key head、128 dim |
| QSA 压缩率 | 4 token → 1 micro-block |
| QSA Token budget | 2048，即最多 512 个完整 block，再加未满尾块 |
| GDN QK / V heads | 16 / 48 |
| GDN QK / V head dim | 128 / 128 |
| GDN 短卷积核 | 4 |
| Routed experts | 512 |
| 每 token Routed experts | 10 |
| Shared expert | 1 |
| 单专家 Intermediate size | 640 |
| Residual branches | 4 |
| GR 低秩瓶颈 | 320，即 `hidden_size / 8` |
| N-gram | 2-gram + 3-gram，各 8 个哈希头 |
| N-gram 基础词表 | 每个哈希头约 20M slots |
| N-gram 注入位置 | 第 2 层 |
| MTP | 多步训练；公开配置含 1 层 MTP 模块 |
| 原生上下文 | 262,144 token |
| 扩展上下文 | 通过 YaRN 扩到 1,000,000 token |
| Vision Tower | 27 层、1152 hidden、16 heads、4304 FFN |
| Vision patch / merge | `2 × 16 × 16` 时空 patch，`2 × 2` 空间合并 |

这里最容易混淆的是参数口径：

```text
125B  主模型参数
+51B  N-gram 查表参数
≈176B 权重容量
```

但“总共约 176B 权重”不等于每个 token 做 176B 参数的矩阵乘法。MoE 每次只路由 10 个专家，N-gram 参数又通过确定性哈希查表访问，因此官方把主要逐 token 计算口径写成约 `6B activated parameters`。

### 1.3 官方博客中的主要亮点

#### 编码与 Agent

| Benchmark | Qwen3.8-Flash-Next | Qwen3.8-27B | Qwen3.7-Plus | DeepSeek-V4-Flash-0731 |
|---|---:|---:|---:|---:|
| DeepSWE 1.1 | 58.7 | 42.2 | 16.5 | 54.4 |
| SWE-bench Pro | 62.5 | 61.7 | 55.8 | 56.0 |
| SWE-bench Multilingual | 81.0 | 73.8 | 75.8 | — |
| CoWorkBench | 73.9 | 70.7 | 65.1 | 45.1 |
| JobBench | 55.7 | 33.4 | 27.6 | 41.3 |
| Toolathlon Verified | 73.5 | 67.1 | 50.6 | 70.3 |
| LiveCodeBench v6 | 91.9 | 90.3 | 89.6 | 90.6 |

这些结果说明它的提升重点并不只是传统知识问答，而是代码库修改、工具调用和长程办公任务。需要注意，DeepSWE、SWE-bench、Vision2Web 等结果依赖具体 Agent harness、上下文窗口、采样参数和工具权限；它们不能被简单解释为模型裸能力的绝对排名。

#### 多模态 Agent

| Benchmark | Qwen3.8-Flash-Next | Qwen3.8-27B | Qwen3.7-Plus |
|---|---:|---:|---:|
| ClawEval-MM Pass@3 / Avg. | 64.4 / 60.4 | 57.4 / 56.9 | 57.4 / 60.1 |
| RecreationBench | 49.9 | 47.1 | 30.2 |
| AndroidWorld | 84.5 | 81.9 | 81.0 |
| OSWorld 2.0 Binary / Partial | 19.4 / 52.3 | 19.4 / 48.0 | 2.8 / 21.5 |
| Vision2Web | 64.0 | 62.9 | 42.1 |
| LVBench | 76.6 | 72.4 | 76.2 |
| RealWorldQA | 88.5 | 85.9 | 86.9 |

Flash-Next 的多模态亮点更偏“看完之后继续行动”：识别屏幕、操作移动端/桌面、复刻应用和生成网页，而不只是静态 VQA。

#### 长上下文效率

官方报告的三个数字应分开理解：

- 1M token 时，QSA Attention kernel 相对 dense GQA 的 **Prefill 加速为 `7.6×`**；
- 同一长度下，QSA Attention kernel 的 **Decode 加速为 `4.9×`**；
- 在 `90% Prefix Cache hit rate` 的在线服务实验中，整模型相对 Qwen3.7-Plus 的 **Prefill throughput 为 `8.6×`**。

前两个是 Attention 模块的 kernel 对比；第三个是指定缓存命中率、模型规模和服务负载下的端到端吞吐对比。三者不能互换，也不能推出任意 batch、任意命中率下都会稳定得到相同加速。

## 2. 整体模型结构

![Qwen3.8-Flash-Next 官方架构图](./qwen3.8_flash_next_architecture.png)

文本主干可以压缩成下面这条路径：

```text
Text token ── Vocabulary Embedding ───────────────────────────────┐
                                                                 │
2-gram / 3-gram token IDs ── Hash Lookup ── PLE（仅第 2 层）─────┤
                                                                 ▼
                 4 路 Expanded Residual Stream
                              │
          ┌───────────────────┴───────────────────┐
          │  GR Read → GDN → GR Write →           │ × 3
          │  GR Read → MoE → GR Write             │
          ├───────────────────────────────────────┤ × 12 组
          │  GR Read → QSA → GR Write →           │ × 1
          │  GR Read → MoE → GR Write             │
          └───────────────────┬───────────────────┘
                              │
                    Final GR Read → LM Head
                              └────────→ MTP draft path
```

图像和视频走独立 Vision Tower，经过空间合并与投影后变成 2560 维视觉 token，再与文本 token 一起进入同一个 decoder-only 主干：

```text
Image / Video → Vision Encoder → Patch Merger → 2560-d visual tokens
Text          → Token Embedding ────────────────────────────────┐
                                                                ▼
                                                    Unified LLM Backbone
```

因此“原生多模态”不代表像素直接进入语言 Transformer；结构仍是成熟的 `Vision Encoder + Merger/Projector + Decoder-only LLM`。原生性的重点在于统一预训练、后训练和 Agent 数据，而不是取消视觉编码器。

## 3. 参数量与训练成本复算

### 3.1 125B 主模型为什么能做到每 token 只激活约 6B

公开配置中，每个 routed expert 是一个 SwiGLU FFN：

$$
P_{\text{expert}}
\approx
3d_{\text{model}}d_{\text{ff}}
$$

代入 `d_model = 2560`、`d_ff = 640`：

$$
P_{\text{expert}}
=3\times2560\times640
=4{,}915{,}200
\approx4.92\text{M}
$$

48 层、每层 512 个 routed experts，因此仅 routed expert 权重就约为：

$$
4.9152\text{M}\times512\times48
\approx120.80\text{B}
$$

这已经解释了 125B 主模型的大部分参数从何而来。剩余参数主要来自 GDN/QSA、GR、共享专家、词嵌入、LM Head、Router、Vision Tower 和 MTP。

但一个 token 每层只走 10 个 routed experts，并始终走 1 个 shared expert。仅专家路径的激活参数粗算为：

$$
4.9152\text{M}\times(10+1)\times48
\approx2.60\text{B}
$$

再加所有 token 都会经过的 GDN/QSA、GR、Router、输出投影和 LM Head，最终落到官方约 `6B activated parameters` 的数量级。

这个复算的意义是解释数量级，而不是重建官方逐算子参数统计。激活参数量也不等于实际 FLOPs 或延迟：专家并行通信、Router、输出词表投影、Attention、显存带宽和 batch 形态都仍会影响端到端速度。

### 3.2 额外 51B N-gram 参数怎样得到

配置给出：

```text
ngram_size           = 3
heads_per_ngram      = 8
ngram_vocab_size_base= 20,000,000
ple_embed_dim        = 2560
```

模型同时建立 2-gram 和 3-gram 表，每种 n-gram 各有 8 个哈希头，因此共有：

$$
(3-1)\times8=16\ \text{个哈希头}
$$

每个头约有 20M 个 slot；2560 维 PLE embedding 被均分成 16 份，每份 160 维。忽略为了减少哈希冲突而选用相邻质数及对齐带来的小差异：

$$
P_{\text{ngram}}
\approx16\times20\text{M}\times160
=51.2\text{B}
$$

这与官方“额外 51B”完全吻合。若用 BF16 保存，原始权重体积约为：

$$
51.2\text{B}\times2\ \text{bytes}
=102.4\text{GB}
\approx95.4\text{GiB}
$$

Transformers 参考实现也专门把这张约 95 GiB 的表标记为可跳过常规设备放置，由 Host Memory 承担存储。

### 3.3 “训练成本约 1/9”怎样理解

技术报告称，相比 Qwen3.7-Plus，Flash-Next：

- 激活参数由 17B 降到 6B，约为 `6/17`；
- 训练 token 数约为上一代的 `1/3`。

若用大模型训练最常见的粗略口径：

$$
C_{\text{train}}\propto P_{\text{active}}\times T
$$

则：

$$
\frac{C_{\text{Flash-Next}}}{C_{\text{Qwen3.7-Plus}}}
\approx
\frac{6}{17}\times\frac{1}{3}
=0.1176
\approx\frac{1}{8.5}
$$

这与官方“约 `1/9` 训练 FLOPs”相符。该估算没有显式展开 Attention、N-gram 查表、Router、通信和 Vision 训练成本，所以适合解释官方比例，不应当用于预测具体 GPU 小时。

## 4. Gated DeltaNet：用固定状态高效“记忆”

### 4.1 为什么不是全部使用稀疏 Attention

全局 Attention 可以直接检索任意历史 token，但 prefill 计算量随序列长度近似二次增长，decode 时还要不断读取随历史线性增长的 KV Cache。纯滑窗 Attention 虽然便宜，却只能让窗口外信息通过层数间接传播。

GDN 把历史压缩进固定大小的矩阵状态；QSA 层则周期性恢复精确的 token 级检索。二者形成分工：

> GDN 负责持续、低成本地“记住”；QSA 负责从长历史中精确“找回”。

技术报告在 25B-A3B、28 层的架构消融中给出：

| Token mixer | 9 项平均分 |
|---|---:|
| Full Attention | 49.87 |
| SWA Hybrid | 51.15 |
| GDN Hybrid | 53.81 |

两个 Hybrid 都是每四层保留一层全局 Attention；GDN Hybrid 在 9 项中有 8 项超过 Full Attention，并在 7 项上超过 SWA Hybrid。这说明 GDN 不只是在换取效率，至少在该消融规模和训练 recipe 下也改善了能力。

### 4.2 核心递推

对每个 head，令 Query、Key、Value 分别为 $q_t,k_t,v_t$，GDN 维护固定大小状态 $S_t$：

$$
\widetilde S_{t-1}=\alpha_t S_{t-1}
$$

$$
e_t=v_t-\widetilde S_{t-1}^{\top}k_t
$$

$$
S_t=\widetilde S_{t-1}+\beta_t k_t e_t^{\top}
$$

$$
y_t=S_t^{\top}q_t
$$

$\alpha_t$ 是内容相关的衰减门，控制旧状态保留多久；$\beta_t$ 控制写入强度。关键在误差项 $e_t$：模型先读取当前 Key 已经关联的 Value，只写入预测误差，而不是无条件累加新的外积。这种“定向擦除再写入”比简单累加型 linear attention 更不容易让状态无界膨胀。

### 4.3 Flash-Next 的公开维度

GDN 层使用：

- 16 个 Q/K heads，head dim 为 128；
- 48 个 V heads，head dim 为 128；
- 每个 Q/K head 在实现中复制给 3 个 V heads；
- Q/K/V 投影后先经过 kernel size 为 4 的 depthwise causal convolution；
- Q/K 做 L2 normalization；
- 输出经过 Zero-centered RMSNorm 和 Sigmoid output gate。

主要 recurrent state 可按下面的数量级理解：

$$
48\times128\times128
=786{,}432\ \text{elements/layer/sequence}
$$

它不会随着上下文从 128K 增长到 1M 而线性膨胀；另有短卷积状态和门控状态。公开配置把相关 SSM 计算 dtype 设为 FP32，因此不能只拿 BF16 元素大小直接预测真实显存。

### 4.4 Kernel

Qwen 团队用基于 TileLang 的 FlashQLA 优化 GDN。技术报告称，相对 FLATriton baseline：

- Forward 约 `2–3×`；
- Backward 约 `2×`。

这属于同一 GDN 算法不同 kernel 实现的对比，不是整模型相对 Transformer 的端到端倍数。

## 5. Qwen Sparse Attention：压缩“检索器”，不只是稀疏主 Attention

### 5.1 QSA 解决了 DSA 的哪一层瓶颈

典型 learned sparse attention 分两步：

1. 轻量 Indexer 扫描完整历史并决定重要 token；
2. 主 Attention 只读取 Top-K token 的真实 K/V。

主 Attention 从全量历史降到固定 Top-K 后，Indexer 自身的全序列打分可能成为新瓶颈。若每个 Query 仍逐 token 扫描全部历史，Indexer 的 prefill 复杂度仍是 $O(n^2)$。

QSA 先把 Key 序列每 4 个 token 平均池化成一个 micro-block，再做打分：

$$
n\rightarrow\frac{n}{r},\qquad r=4
$$

于是 Indexer 复杂度从：

$$
O(n^2)
$$

下降为：

$$
O\left(\frac{n^2}{r}\right)
$$

核心稀疏 Attention 则约为：

$$
O(nK),\qquad K=2048
$$

### 5.2 Indexer 的实际流程

公开配置与报告给出的 QSA Indexer 是：

```text
Hidden state
   ├─ 4 × Query heads，128 dim
   └─ 1 × shared Key head，128 dim

Token Keys
   └─ 每 4 token AvgPool
      └─ 对压缩后的 block key 做 RMSNorm + Partial RoPE

Query × Block Keys
   └─ ReLU
      └─ 跨 4 个 Query heads 求和
         └─ Top-512 blocks
            └─ 展开为最多 2048 个原始 token
               └─ 加入当前未满 4 token 的尾块
                  └─ 供主 GQA Attention 使用
```

打分可写为：

$$
I_{ib}
=
\sum_{h=1}^{4}
\operatorname{ReLU}
\left(\langle q_i^h,\bar k_b\rangle\right)
$$

其中 $i$ 是 Query token，$b$ 是压缩 block。只有已经完整出现的因果 block 才能参与打分；当前尚未组成完整 block 的尾部 token 始终加入候选集合，避免最近上下文因池化边界被遗漏。

一个重要实现细节是：**先对未加位置编码的 Key 做平均池化，再为 block 起始位置添加 Partial RoPE**。如果先做 RoPE 再平均，不同 token 的旋转相位会相互抵消，内容摘要会被位置相位污染。

### 5.3 主 Attention 仍是带门控的 GQA

QSA 的 Indexer 只负责选位置，真正的内容交互仍由主 Attention 完成：

- 24 个 Query heads；
- 2 个 KV heads；
- head dim 为 256；
- 其中 64 维使用 RoPE；
- Query 投影同时产生逐 head 输出门，最终用 Sigmoid gate 调制 Attention 输出。

因此 QSA 并不是“用 block 平均值代替真实 token 做 Attention”。block 平均值只用于检索；选中 block 后，主 Attention 仍读取其中原始 token 的 K/V。

### 5.4 两阶段训练

QSA 不是从随机 Indexer 直接切到稀疏训练，而是在 256K 序列上分两阶段引入：

#### 阶段 1：Dense Distillation

- Full Attention 给出 token 级 teacher distribution；
- 对每个 block 使用 MaxPool 保留其中最显著 token；
- 用 KL divergence 把分布蒸馏给 block-level Indexer；
- 仅训练 Indexer 1000 steps；
- 每 step 为 8 条 256K 序列，总计约 2B token。

这里 Teacher 使用 MaxPool 而不是 Average Pool，是为了避免一个 block 内少数关键 token 的注意力质量被平均稀释。

#### 阶段 2：Sparse Training

- Indexer 选择 Top-512 blocks；
- 展开为原始 token 后执行稀疏主 Attention；
- Backbone 与 Indexer 联合适应稀疏模式；
- 训练 8000 steps，每 step 96 条 256K 序列，总计约 200B token。

最终 QSA 与 Full Attention 的训练 loss 差约在 $10^{-4}$ 数量级。短任务平均分从 75.9 提高到 76.8；在长上下文实验中，QSA 也没有出现系统性退化。

### 5.5 长上下文结果

| Benchmark | Full Attention | QSA |
|---|---:|---:|
| RULER 512K–1M | 90.08 | 93.00 |
| MRCR 512K | 30.66 | 40.53 |
| MRCR 1M | 20.71 | 26.44 |
| 长上下文两项 Macro Avg. | 78.76 | 80.93 |

QSA 在部分超长上下文任务上反而更高，可能来自稀疏训练适配、检索归纳偏置或评估波动；这不能外推成“稀疏 Attention 普遍比 Full Attention 更准确”。更稳妥的结论是：在官方训练流程和这些评测上，QSA 没有以明显能力损失换效率。

### 5.6 QSA、DSA 与 IndexShare 的区别

| 方法 | 降低什么开销 | 依赖 | 潜在问题 |
|---|---|---|---|
| Token-level DSA | 主 Attention 的 token 数 | 每层独立 Indexer | Indexer 自身仍逐 token 扫描 |
| IndexShare | 跨层复用检索结果 | 相邻 Attention 层的 Top-K 相似性 | Hybrid 中 Attention 层相隔 3 个 GDN 层，相似性可能较低 |
| QSA | 层内把 Key 压成 micro-block | block 内语义可被摘要 | block 太大可能稀释细粒度信号 |

QSA 的核心取舍是把“跨层复用”改成“层内序列压缩”。报告中的消融显示，block size 为 4 时，相对 Indexer latency 约 0.25 且 RULER 回到 Full Attention 水平；IndexShare 在相对 latency 0.5 时仍低于 baseline。最终选择 `r=4`，而不是更激进的 8 或 16。

### 5.7 `7.6×` 和 `4.9×` 为什么不等于理论上的固定倍数

在 1M token：

- 仅 Indexer，压缩率 4 对应实测 Prefill `3.8×`、Decode `4.4×`；
- 把 Indexer 与 sparse core attention 合在一起，对 dense GQA 的 Attention 模块为 Prefill `7.6×`、Decode `4.9×`。

Indexer 的收益大致受 $r=4$ 限制；主 Attention 从扫描 1M token 变为读取约 2048 token，理论稀疏度更高。最终速度还受 Top-K、索引展开、内存访问、kernel 启动和 batch 影响，所以不会等于一个只由 `1M/2048` 决定的理想倍数。

### 5.8 Transformers 参考实现的边界

公开的 Transformers eager/SDPA 参考实现会显式循环 Query、计算 block、构造稀疏 mask，再调用通用 Attention。它适合验证语义，不代表官方生产 kernel 的性能路径。

技术报告中的高性能实现使用融合 QSA kernel，把稀疏 Attention 输出和 KL loss 合并计算，避免物化大型中间结果。若直接用参考 Python 路径跑 256K 或 1M，不应期待复现官方的 `7.6×/4.9×`。

## 6. KV Cache 与固定状态：长上下文显存大致怎样看

### 6.1 只有 12 层保留逐 token K/V

Flash-Next 的 36 个 GDN 层保存固定 recurrent state；只有 12 个 QSA 层需要保存逐 token 的主 Attention K/V。每个 QSA 层有 2 个 KV heads、每头 256 维，因此每 token、每 QSA 层为：

$$
D_{\text{KV,QSA}}
=2\times2\times256
=1024\ \text{elements}
$$

主干 12 个 QSA 层合计：

$$
D_{\text{KV,total}}
=12\times1024
=12{,}288\ \text{elements/token}
$$

在 1M token、单序列、BF16 且不量化的纯主 K/V 粗算为：

$$
12{,}288\times10^6\times2
=24.576\text{GB}
\approx22.9\text{GiB}
$$

这还没包括 QSA Indexer Key、GDN state、卷积 state、MTP、分页对齐和运行时 workspace，因此不能当作部署显存承诺。

### 6.2 与 Qwen3.8-27B 的配置级比较

Qwen3.8-27B 有 64 层，其中 16 层全局 Attention，每层 4 个 KV heads：

$$
D_{\text{KV,27B}}
=16\times2\times4\times256
=32{,}768\ \text{elements/token}
$$

因此只按公开配置中的主 K/V 元素计算：

$$
\frac{32{,}768}{12{,}288}
\approx2.67
$$

Flash-Next 的逐 token 主 KV Cache 约低 `2.67×`。QSA 的意义主要是进一步减少每次从这份 Cache 中读取和计算的 token 数，而不是让所有原始 K/V 都消失；要精确检索选中位置，主 K/V 仍需保留。

## 7. Gated Residual：四路残差怎样读写

### 7.1 为什么要拓宽 Residual Stream

传统 Pre-Norm Transformer 中，每层都从同一条 residual stream 读写。早期层写入的特征会持续和后续所有信息叠加，深层再想取回某个早期特征时，它已经和大量中间结果混在一起。

GR 把 residual state 从：

$$
R\in\mathbb{R}^{d}
$$

拓宽为：

$$
R\in\mathbb{R}^{4\times d}
$$

四路不是固定分工，而是在训练中自然形成不同路径。报告的路径分析发现，通常有一路保留 Layer 0 的早期输出并跨越十多层送到后续全局 Attention，另外三路更多承担 1–4 层范围的局部连接。

### 7.2 GR Read

公开实现先对四个分支分别做 Zero-centered RMSNorm，再把 `4 × 2560 = 10240` 维状态经过 `10240 → 320 → 10240` 的低秩门控网络：

$$
G=
\sigma\left(
W_{up}\operatorname{SiLU}
\left(\frac{1}{4}W_{down}\operatorname{GroupRMSNorm}(R)\right)
\right)
$$

然后逐通道门控并平均四个分支：

$$
x=
\frac{1}{4}
\sum_{i=1}^{4}G_i\odot\widehat R_i
$$

和 mHC 每个分支一个标量的读法相比，GR 的读取粒度细到“每分支、每通道”。

### 7.3 GR Write

子层输出 $y$ 不经完整的分支混合矩阵，而是预测四个写入标量：

$$
s=2\sigma\left(\frac{1}{4}W_w\operatorname{GroupRMSNorm}(R)\right)
$$

$$
R_i'=R_i+s_i y
$$

$s_i\in(0,2)$，使模型可以动态决定当前 Attention/MoE 输出写入每个分支的强度。

### 7.4 与 mHC 的关键差异

mHC 通常还维护一个 $4\times4$ 的分支混合矩阵 $H_{res}$，并用双随机约束控制它。Qwen 的消融发现，当 read/write 已足够表达时，$H_{res}$ 没有显著收益，却需要额外读取整份 residual state，也引入了新的数值约束。

GR 因此选择：

- 把表达力集中在逐元素 Read gate；
- Write 只保留每分支标量；
- 完全删除跨分支 $H_{res}$。

在 25B-A3B、560B token 的消融中：

| Residual | Loss | 9 项平均分 |
|---|---:|---:|
| Pre-Norm | 1.617 | 50.91 |
| mHC Static | 1.596 | 52.49 |
| mHC Dynamic | 1.594 | 54.47 |
| GR | 1.590 | 54.66 |

### 7.5 GR 不是免费午餐

四路 residual 会显著增加每层的激活读写量，推理成本主要受显存带宽而不是门控 FLOPs 影响。官方采用三项补救：

1. 删除 $H_{res}$，少完整扫描一次四路状态；
2. 把 GR Read/Write 各自融合成单个 kernel；
3. residual state 支持 FP8 保存，相对 BF16 把这部分传输字节减半。

即便采用 FP8，四路 residual 的原始字节量仍约是单路 BF16 的两倍。因此 GR 的价值来自能力和稳定性收益是否覆盖额外内存流量，而不是算术上“几乎没有矩阵乘法”就等于没有成本。

## 8. N-gram Embedding：把容量放到加速卡之外

### 8.1 不是普通的三元组词表

对当前位置 token，模型分别构造：

- 当前 token 与前一个 token 的 2-gram；
- 当前 token 与前两个 token 的 3-gram。

每种 n-gram 使用 8 组不同哈希，共 16 个 hash heads。实现用不同的 64-bit 奇数乘子和 XOR 混合 token ID，再对各自接近 20M 的质数词表取模。多哈希头可以降低单次碰撞对整体表示的破坏。

在 EOS 边界处，历史 token 不跨段拼接，避免把两个独立样本或对话段落错误组合成 n-gram。

### 8.2 查表之后还发生了什么

每个 hash head 取回一个 160 维向量，16 个头拼成 2560 维：

```text
16 × 160-d lookup
       ↓ concat
2560-d N-gram embedding
       ├─ Key projection   → 4 × 2560
       └─ Value projection → 2560

4 路 residual state → Query normalization
Query · Key → 每路注入 gate
Value × gate → depthwise dilated conv → 写入四路 residual
```

PLE 还使用 kernel size 4、dilation 3 的 depthwise causal convolution，给查表值增加局部组合能力。也就是说，N-gram 表不是简单与 token embedding 相加，而是在第 2 层根据当前四路 residual 内容做门控注入。

### 8.3 为什么只放在第 2 层

报告对第 1、2、3、4、10、15、25 层以及双层注入做了消融：

- 第 1、2 层整体表现最强；
- 中层、深层仍有效，但没有稳定优势；
- 固定参数预算下，多层分摊没有带来一致收益；
- 第 2 层方便在计算第 1 层时异步预取 Host Memory 中的表项。

因此最终只保留 Layer 2，而不是在每一层重复放置巨型 embedding。

### 8.4 为什么 51B 不能简单算成“免费参数”

N-gram 查表几乎不增加大矩阵乘法，但仍包含：

- 16 路哈希与索引计算；
- Host Memory 容量和带宽；
- CPU/加速卡间的数据传输；
- 批内访问离散时的缓存命中问题；
- 碰撞造成的容量利用率损失；
- 后续 Key/Value projection、门控和空洞卷积。

官方方案成立的关键是索引可由 token 序列提前确定，因此可以和第一层计算异步重叠。若部署框架没有实现预取或主机带宽不足，“低 FLOPs”未必会转化成“低延迟”。

### 8.5 Loss 与下游准确率并不单调一致

在保持 MoE 不变、额外扩大 N-gram 词表时，训练 loss 会随着词表从 `20×` 增至 `200×` 基础词表而持续下降；但多数下游 Benchmark 较早饱和或波动，只有中文 C-Eval、CMMLU 等更稳定地继续提升。

这也是整篇技术报告反复强调的设计原则：不能只用 loss 选架构。更多查表参数能增强对局部模式的拟合，却不保证同比例提升推理、代码或 Agent 能力。

## 9. Ultra-Sparse MoE、MTP 与其他稳定性设计

### 9.1 512 专家、Top-10 + Shared Expert

每个 decoder layer 都使用 MoE：

```text
Router(2560 → 512)
    ├─ Top-10 routed experts，概率重新归一化
    └─ 1 shared expert，经独立 Sigmoid gate
```

单专家只有 640 intermediate，比 dense 模型的 FFN 窄很多；靠 512 个专家扩大总容量，靠 Top-10 保持逐 token 计算量低。Global load balancing 用于避免少数专家过热或长期得不到训练信号。

### 9.2 MTP

模型保留 Multi-Token Prediction 路径，并在训练时使用多步预测，使训练与实际 speculative decoding 对齐。QSA 也替换了 MTP 中的全 Attention，并在连续 draft step 间复用 Top-K 索引。

四步 speculative decoding 的实验中，平均接受长度从 Full Attention 的 4.06 变为 QSA 的 4.07，基本无损。这里“四步”对应主模型当前预测加后续 MTP draft steps；公开 Transformers 配置中的 MTP 模块层数为 1，不应误读成主干额外复制四个完整 decoder layer。

### 9.3 其余稳定性组件

- Zero-centered RMSNorm，并对 Norm 权重应用 weight decay；
- GDN 与 QSA 输出都使用有界 Sigmoid gate；
- MoE Router 使用归一化初始化；
- 不依赖 qk-clip 或 SwiGLU-clip 等显式激活裁剪完成整次训练。

## 10. Vision Tower 与多模态输入

### 10.1 Vision 配置

| 项目 | Qwen3.8-Flash-Next Vision |
|---|---:|
| 层数 | 27 |
| Hidden size | 1152 |
| Attention heads | 16 |
| FFN intermediate | 4304 |
| Patch size | 16 |
| Temporal patch size | 2 |
| Spatial merge | 2 |
| 输出维度 | 2560 |
| DeepStack visual injection | 未启用，`deepstack_visual_indexes=[]` |

视觉处理器沿用 Qwen3-VL 协议：动态图像分辨率、时空 patchify、`2×2` spatial merge、mRoPE，以及独立的 image/video placeholder。

一个合并后的视觉 token 在空间上大约覆盖：

$$
(16\times2)^2=1024\ \text{pixels}
$$

图像处理配置给出 `65,536–16,777,216` 的像素面积预算，对应约：

$$
64\sim16{,}384
$$

个合并后视觉 token。实际数量还受长宽比、patch 对齐和 temporal 维度影响。

### 10.2 与 Qwen3.8-27B 的关系

Flash-Next 和 Qwen3.8-27B 使用相同的 27 层、1152 hidden Vision Tower 与相同的图像/视频 Processor。主要区别在语言主干：

- Qwen3.8-27B 把视觉特征投影到 5120 维 dense backbone；
- Flash-Next 投影到 2560 维四路 GR + MoE backbone；
- 两者图像归一化、patch size、temporal patch、spatial merge 基本相同。

因此 Flash-Next 的多模态提升不能简单归因于换了更大的 ViT，更可能来自统一后训练数据、Agent 轨迹、语言主干、长上下文检索和推理时工具链的共同作用。

## 11. Muon 与训练稳定性

### 11.1 哪些参数使用 Muon

Muon 用于真正具有二维线性映射语义的矩阵：

- QSA 的 Q/K/V 和输出投影；
- GDN 输入/输出投影；
- Routed/Shared experts 的 `fc1/fc2`；
- N-gram PLE 的 Key/Value projection。

继续使用 AdamW/Adam 的部分：

- Vocabulary Embedding 和 LM Head；
- MoE Router；
- GR 的细长低秩投影；
- N-gram 巨型查表使用无 weight decay 的 Adam。

Router 的输出维分别代表相对独立的专家分数，GR 低秩矩阵又极为细长，二者没有从正交化中稳定获益。

### 11.2 为什么 fused matrix 要先拆开

Megatron 中 QKV、SwiGLU gate/up、GDN 输入投影常物理融合成一张矩阵。若直接对拼接矩阵做 Muon 正交化：

1. 不相关子块的奇异方向会互相混合；
2. Muon 的 shape scaling 会使用错误的拼接形状。

因此官方先按真实语义拆分，再分别正交化。QKV 和 GDN 输入甚至按 head 粒度拆分；大量小 kernel 再用 CUDA Graph 捕获，避免启动开销吞掉收益。

### 11.3 Muon 具体设置

- Nesterov momentum：`0.95`；
- Newton–Schulz：8 steps；
- 使用 Polar Express 的逐步系数；
- 更新缩放：$0.2\sqrt{\max(A,B)}$；
- Frobenius normalization 稳定常数：$10^{-14}$。

8 次 NS 比更少迭代增加了优化器计算，但在压力测试中降低了 gradient-norm spike 的幅度和频率。分布式实现还需按估计的正交化 FLOPs 而不是参数元素数重新平衡 DP rank，否则不同形状矩阵会造成严重 straggler。

### 11.4 重新拟合 Scaling Law

新架构和 Muon 把近似最优点推向更大的 Batch Size 与 Learning Rate：

- 大 batch 改善大规模并行吞吐；
- 对 MoE 而言，每步更多 token 也让每个专家获得更充足、更分散的训练信号；
- 在测试的 4T token 小模型上，目标 batch 从 12.6M 提到 25.2M token 后 loss 改善约 `7.2×10^-3`。

官方还发现 Batch Size Warmup 没有改善最终 loss，却为了相同 token 预算多消耗 `18.8%` optimizer steps，因此生产训练从一开始就使用目标 batch。

### 11.5 稳定性压力测试

在 28 层 25B-A3B 模型上，官方把 Learning Rate 固定在最优值的 2× 和 4×，有意让中等规模实验复现大规模长训练才会出现的失稳：

- 4× LR 时，Qwen3.5 + AdamW 每 10K steps 出现约 183 次 loss spike；
- Muon + GR 没有跨越 gradient clipping 阈值，并记录到 0 次 loss spike；
- 单独打开 GatedNorm，在 3× LR 下把 spike rate 从 32.0/10K 降到 3.2/10K。

生产训练的全尺度验证中也未出现 loss spike 或异常 gradient norm。更合理的因果表述是：Muon、GR/GatedNorm、门控 Attention/GDN、初始化和新超参数 recipe 共同形成稳定性余量，而不是把全部收益归因于某一个 Gate。

## 12. Base Model 能力与结果边界

技术报告比较的是 Base checkpoint，与博客中完成后训练的 Agent/多模态模型不是同一口径。

| Benchmark | Flash-Next-Base | Qwen3.8-27B-Base | Qwen3.7-Plus-Base |
|---|---:|---:|---:|
| MMLU | 90.36 | 87.51 | 90.43 |
| MMLU-Pro | 73.23 | 68.60 | 70.90 |
| SuperGPQA | 51.36 | 44.86 | 48.42 |
| BBH | 90.87 | 89.56 | 89.41 |
| GPQA | 51.42 | 45.01 | 51.52 |
| GSM8K | 93.29 | 93.18 | 92.95 |
| MATH | 72.78 | 60.54 | 74.38 |
| EvalPlus | 78.76 | 76.05 | 78.06 |
| MultiPL-E | 79.09 | 74.50 | 81.68 |
| SWEBench-Pretrain | 50.99 | 41.66 | 49.24 |
| MGSM | 89.33 | 86.37 | 85.42 |
| MMMLU | 84.86 | 79.74 | 84.53 |
| INCLUDE | 78.40 | 74.37 | 78.90 |

Flash-Next-Base 在 14 项中的 8 项超过 Qwen3.7-Plus-Base，其余差距最大约 2.6 分；同时激活参数和训练 token 都约为上一代的三分之一。它并没有逐项碾压旗舰模型，优势更准确地说是 Pareto frontier 向低成本方向移动。

## 13. 与 Qwen3.8-27B / Qwen3.7-Plus 的结构对比

| 项目 | Qwen3.8-Flash-Next | Qwen3.8-27B | Qwen3.7-Plus |
|---|---:|---:|---:|
| 主模型参数 | 125B MoE | 27B Dense | 397B MoE |
| 激活参数 | 6B | 27B | 17B |
| 额外 N-gram | 51B | — | — |
| 主干层数 | 48 | 64 | 官方本次未公开完整配置 |
| Hidden size | 2560 | 5120 | — |
| Attention 排列 | 36 GDN + 12 QSA | 48 GDN + 16 Full/Gated Attn | GDN + Gated Attn Hybrid |
| Q/KV heads | 24 / 2 | 24 / 4 | — |
| FFN | 512 experts，Top-10 + shared | Dense 17408 FFN | MoE |
| Residual | 4 路 GR | 单路 Pre-Norm | 单路 Pre-Norm |
| 原生上下文 | 262K，可扩 1M | 262K，可扩 1M | 1M 服务能力 |
| Vision Tower | 27 层，1152 → 2560 | 27 层，1152 → 5120 | 原生多模态 |

Flash-Next 的核心不是在所有维度上变大，而是把容量重新分配：

- 用 512 个小专家存主干知识；
- 用 51B Host Memory 查表存局部模式；
- 用四路 residual 改善跨层信息通道；
- 用 GDN state 与 QSA Top-K 控制长上下文成本；
- 把 dense 27B 的 5120 hidden 缩到 2560，以换取更小的共享计算。

## 14. 部署时最值得关注的工程问题

### 14.1 权重容量仍然很大

虽然每 token 只激活约 6B，部署仍要容纳或分片约 125B 主模型权重；51B N-gram 表还需要约 95 GiB BF16 Host Memory。MoE 的低计算量不自动等于低权重存储量或低网络通信量。

### 14.2 高性能依赖专用 Kernel

要兑现架构收益，推理引擎至少需要高效实现：

- GDN recurrent/chunk kernel；
- QSA 压缩 Indexer、Top-K 与 sparse core attention；
- GR fused read/write 与 FP8 residual；
- MoE expert parallel / all-to-all；
- N-gram Host Memory 异步预取；
- MTP 的 QSA index reuse。

缺少其中若干项时，理论 FLOPs 很低也可能被 Python 控制流、随机内存读取或通信开销淹没。

### 14.3 1M 是扩展长度，不是默认原生长度

开源权重原生上下文为 262,144，1M 通过 YaRN 扩展。官方长上下文结果证明该模型可以工作到 1M，但实际质量和吞吐仍依赖 RoPE/YaRN 配置、QSA kernel、KV Cache dtype、chunked prefill 和显存预算。

### 14.4 “Flash”更像服务层定位

它不追求最小磁盘占用或单卡最容易跑，而是追求：

- 较大的总知识容量；
- 较低的逐 token 矩阵计算；
- 长上下文下有限的 Attention 读取；
- 高缓存命中在线服务中的吞吐；
- 编码与办公 Agent 的能力/成本比。

如果硬件只能低并发运行、Host Memory 预取无法重叠、或推理框架没有 QSA/GDN/GR 专用 kernel，Flash-Next 的结构优势可能无法完整转化成真实延迟优势。

## 15. 总结

Qwen3.8-Flash-Next 最重要的不是某个单独算子，而是一次比较完整的“容量—计算—内存—稳定性”联合设计：

1. **GDN + QSA**：36 层用固定状态压缩历史，12 层用 block-level Indexer 精确检索；
2. **QSA**：先把每 4 个 Key 压成 micro-block，使 Indexer 本身也降本，再从最多 512 个 block 展开 2048 个真实 token；
3. **GR**：四路 residual 提供不同时间尺度的跨层路径，逐元素 Read gate 改善表达与稳定性，删除昂贵的分支混合矩阵；
4. **N-gram Embedding**：用 16 个哈希头和约 51.2B 查表参数增加局部模式容量，并靠 Host Memory 预取隐藏访问延迟；
5. **Ultra-Sparse MoE**：约 120.8B routed expert 权重构成主模型容量主体，但每 token 每层只选 10 个专家；
6. **Muon + 新 Scaling Law**：更大 batch、learning rate、无 batch warmup，并用压力测试验证稳定性，而不是只按小规模 loss 选择 recipe；
7. **原生多模态与 Agent 后训练**：Vision Tower 本身延续 Qwen3-VL 路线，能力跃升更依赖语言主干、长上下文、工具使用与后训练闭环。

从公开结果看，它已经把上一代 `397B-A17B` 模型的大部分能力压缩进更低的激活和训练预算中，并在代码、办公和部分多模态 Agent 任务上反超。与此同时，它对推理系统提出了更高要求：QSA、GDN、GR、MoE、N-gram offload 和 MTP 都需要协同实现。换言之，Qwen3.8-Flash-Next 是一个以系统级优化换取极致性价比的模型，而不是仅靠模型文件就能自然获得所有加速的轻量模型。

## 参考资料

1. [Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency](https://qwen.ai/blog?id=qwen3.8-flash-next)
2. [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
3. [Qwen3.8-Flash-Next GitHub Repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)
4. [Qwen3.8-Flash-Next Hugging Face 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
5. [Transformers Qwen4Exp 参考实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
6. [Qwen3.8-Flash-Next 图像预处理配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/preprocessor_config.json)
7. [Qwen3.8-Flash-Next 视频预处理配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/video_preprocessor_config.json)
8. [Qwen3.8-27B 模型卡与公开结构](https://huggingface.co/Qwen/Qwen3.8-27B)
9. [Qwen3.8-27B Hugging Face 配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
10. [FlashQLA](https://github.com/QwenLM/FlashQLA)
11. [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d)

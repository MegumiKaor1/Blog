# 1. LLM：大语言模型

> 一切 Agent 的基石。没有 LLM，就没有后续所有技术。

## Transformer 架构

**核心论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Vaswani et al., 2017）

Transformer 架构的提出奠定了大模型时代的基础，使基于注意力机制的生成模型成为主流。

### 三种架构变体

```
┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────┐
│  Encoder-Only   │  │  Encoder-Decoder     │  │  Decoder-Only   │
│  (BERT 等)       │  │  (T5, BART 等)       │  │  (GPT, LLaMA 等) │
│  🟣 理解任务     │  │  🟢 经典生成架构      │  │  🔵 当前最主流   │
└─────────────────┘  └──────────────────────┘  └─────────────────┘
```

| 架构 | 代表模型 | 适合任务 | 当前地位 |
|------|----------|----------|----------|
| **Encoder-Only** | BERT、RoBERTa | 文本分类、NER、情感分析 | 被 Decoder-Only 替代 |
| **Encoder-Decoder** | T5、BART | 翻译、摘要 | 特定场景仍有使用 |
| **Decoder-Only** | GPT-4、Claude、LLaMA、Qwen | 一切文本任务 | **绝对主流** |

### 核心组件

**Self-Attention**：每个 token 都能「看到」序列中所有其他 token，捕捉长距离依赖。

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

**Multi-Head Attention**：多组注意力并行计算，不同头关注不同子空间（语法、语义、位置等）。

**Positional Encoding**：Transformer 本身无序，需要注入位置信息（正弦位置编码或可学习位置编码）。

**Feed-Forward Network (FFN)**：两层全连接 + 激活函数，对每个位置独立做非线性变换。

### Decoder-Only 的自回归生成

当前主流 LLM 的核心生成方式：

$$
P(y_1, y_2, ..., y_T | x) = \prod_{t=1}^{T} P(y_t | y_{<t}, x)
$$

一个 token 一个 token 地预测下一个词，直到生成结束符。

### 为什么 Decoder-Only 赢了？

1. **统一范式**：所有 NLP 任务都可以转化为「续写」—— 输入一段文字，模型续写出答案
2. **训练效率**：Causal Attention Mask 让训练并行化极高
3. **规模化友好**：Scaling Law 验证了参数、数据、算力越大效果越好
4. **涌现能力**：大到一定程度后出现推理、代码、多语言等能力

### 关键论文延伸

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)（Kaplan et al., 2020）— 量化了参数-数据-算力的关系
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)（Hoffmann et al., 2022）— Chinchilla 定律，数据量应该和参数量同步增长

---

**关键词**：Self-Attention、Multi-Head Attention、Positional Encoding、FFN、Autoregressive、Scaling Law

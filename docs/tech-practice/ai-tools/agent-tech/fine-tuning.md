# 3. Fine-tuning：微调

> 用特定数据「定制」模型的行为和能力，而不从头训练。

**核心论文**：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（Hu et al., 2021）

---

## 为什么需要微调？

预训练模型的「通识能力」很强，但：

- 不知道你的领域术语
- 不理解你的输出格式偏好
- 无法模仿你的说话/写作风格
- 缺乏特定任务的专业能力

微调就是在已有模型基础上，用特定数据再训练，让模型更适合某个具体任务或场景。

---

## 微调的方式

### 全参数微调（Full Fine-tuning）

训练模型的所有参数。效果好但极其昂贵 —— 一个 7B 模型需要 ~56GB 显存。

### 参数高效微调（PEFT：Parameter-Efficient Fine-Tuning）

只训练少量新增/修改的参数，冻结原模型权重。

| 方法 | 核心思路 | 可训练参数 |
|------|----------|------------|
| **LoRA** | 低秩分解增量矩阵 `ΔW = A·B` | < 1% |
| **QLoRA** | LoRA + 4-bit 量化 | < 1%，更省显存 |
| **Adapter** | 在 Transformer 层间插入小网络 | 3-5% |
| **Prefix Tuning** | 在输入前加可训练前缀向量 | < 0.1% |

### LoRA 的核心思想

```
全参数微调：训练所有 W（昂贵）
LoRA 微调：  W' = W + ΔW = W + A·B
             只训练 A 和 B（低秩矩阵），大幅降低训练成本
```

**为什么低秩就够了？** 研究表明确实有低「内在维度」（intrinsic dimension）—— 模型适配新任务时，权重更新实际上发生在低维子空间。

**优势**：

- 参数量减少 1000 倍以上（7B 模型只需要几 MB）
- 消费级 GPU 可微调 7B/13B 模型
- 多个 LoRA 适配器可热切换（不同任务用不同 LoRA）
- 不影响原模型，随时可卸

---

## 常见微调框架

| 框架 | 特点 |
|------|------|
| [**Hugging Face PEFT**](https://github.com/huggingface/peft) | 最主流的 PEFT 库，支持 LoRA/QLoRA/Adapter 等 |
| [**Unsloth**](https://github.com/unslothai/unsloth) | 2-5x 更快，更省显存，新手友好 |
| [**Axolotl**](https://github.com/axolotl-ai-cloud/axolotl) | YAML 配置驱动，社区活跃 |
| [**TRL**](https://github.com/huggingface/trl) | HuggingFace 出品，支持 SFT/DPO/PPO/GRPO |

---

## 微调 vs 其他方法

| 方法 | 适用场景 | 成本 | 是否改模型 |
|------|----------|------|------------|
| **Prompt Engineering** | 格式/行为微调 | 极低 | ❌ |
| **Few-shot** | 临时任务导向 | 低 | ❌ |
| **RAG** | 知识更新 | 中 | ❌ |
| **Fine-tuning** | 深层能力定制 | 高 | ✅ |

选择原则：能用 Prompt 解决就不用 Fine-tune，能用 RAG 解决知识问题就不微调。

---

## 相关阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) — 4-bit 量化 + LoRA
- [PEFT 官方文档](https://huggingface.co/docs/peft)

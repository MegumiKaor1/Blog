# 2. Prompt Engineering：提示词工程

> 不改变模型参数，只改变输入，让模型输出更好。

**参考**：[提示词工程笔记](https://www.aneasystone.com/archives/2024/01/prompt-engineering-notes.html)

---

## 什么是 Prompt

提示词（Prompt）是用来引导模型按照特定意图生成输出的输入指令。本质上，Prompt 是**人类意图与模型能力之间的翻译层**。

### 两类 Prompt

| 类型 | 作用 | 示例 |
|------|------|------|
| **系统提示词**（System Prompt） | 设定角色、规则、输出格式、行为边界 | "你是一个严谨的代码审查员，只指出安全漏洞，不讨论代码风格" |
| **用户提示词**（User Prompt） | 提出具体问题或任务 | "审查这段代码的 SQL 注入风险" |

---

## 核心技巧

### Few-shot Prompting

给模型几个「输入 → 输出」示例，让它学会格式和风格：

```
将以下句子翻译成日语：
English: Hello → こんにちは
English: Goodbye → さようなら
English: Thank you →
```

### Chain-of-Thought (CoT)

让模型「一步步思考」，适用于推理、数学、逻辑任务：

```
Q: 小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
A: 我们一步步来：
1. 小明最初有 5 个苹果
2. 给了小红 2 个 → 5 - 2 = 3 个
3. 又买了 3 个 → 3 + 3 = 6 个
答案：6 个苹果
```

**Zero-shot CoT**：只需加一句 "Let's think step by step"。

### Role Prompting

赋予角色身份，激活模型在该领域的最佳行为模式：

```markdown
你是一位有 15 年经验的机器学习安全研究员，曾在 NeurIPS 和 S&P 发表过多篇 Backdoor Attack 论文。
请评审以下论文的实验设计。
```

### Structured Output

明确要求输出格式：

```markdown
请用 JSON 格式输出，包含以下字段：
- summary: 一句话总结
- pros: 优点列表
- cons: 缺点列表
- score: 1-10 的整数评分
```

---

## 进阶技术

| 技术 | 说明 | 适用场景 |
|------|------|----------|
| **Self-Consistency** | 多次采样取多数答案 | 数学推理 |
| **Tree-of-Thought** | 分支推理树，回溯探索 | 规划、创造性任务 |
| **ReAct** | 交替推理 (Thought) 和行动 (Action) | Agent 交互 |
| **DSPy** | 程序化优化 Prompt，不再手写 | 生产级系统 |

---

## 核心原则

> Prompt Engineering 的核心不是「堆字数」，而是「精确传递意图」。

- **越具体越好**：模糊的需求 → 模糊的输出
- **先给结构再给内容**：格式要求放在规则前面
- **迭代胜过完美**：先跑一版，看输出再改 Prompt
- **模型不同 Prompt 也不同**：GPT-4 的好 Prompt 不一定适合 Claude

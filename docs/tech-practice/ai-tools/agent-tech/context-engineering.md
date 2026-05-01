# 9. Context Engineering：上下文工程

> Agent 不只是「给模型喂信息」，而是「喂对的信息」。

**参考**：

- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)（Anthropic）
- [LangChain Context Engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)

---

## 什么是 Context Engineering

Agent 运行中需要提供给 LLM 的一切相关信息都是**上下文**：

- 对话历史
- 用户输入
- 系统提示词（Persona / SOUL.md）
- 工具调用结果
- 检索到的文档
- 记忆检索结果
- 项目规范文件（AGENTS.md）

> Context Engineering 是一门工程学科：**在有限 Token 窗口内，为 LLM 组织最有利于完成任务的信息结构。**

---

## 核心操作

### 1. 筛选（Filter）

不是所有信息都值得放进上下文。

```
❌ 「把这 50 个文件全塞进去」
✅ 「先搜索相关文件，只取 Top-5 最相关的」
```

**策略**：
- 语义检索 + 关键词过滤
- 相关性评分 + 阈值截断
- 工具返回结果先摘要再传入

### 2. 压缩（Compress）

Token 有限，信息要精炼。

| 技术 | 说明 |
|------|------|
| **对话摘要** | 长对话自动压缩早期内容为摘要 |
| **工具结果截断** | 只传关键字段，丢弃冗余格式 |
| **Prompt Caching** | 不变的 Prompt 部分缓存复用（Anthropic 特有） |
| **分层上下文** | 关键信息靠前，次要信息靠后 |

### 3. 组织（Organize）

信息排列顺序和结构直接影响模型表现。

```
✅ 好的组织：
[System Prompt: 你是谁]
[关键指令: 这次要做什么]
[相关背景: 为什么做]
[工具结果: 最新的数据]
[对话历史: 最近几轮]
[长期记忆: 用户偏好]

❌ 差的组织：
把所有信息随机拼接
```

**「Lost in the Middle」效应**：LLM 对开头和结尾的信息关注度最高，中间部分容易被忽略。

---

## Context Engineering 的最佳实践

### Anthropic 的建议

1. **优先放指令**：不要让 Background 挤掉核心 Task
2. **示例比规则更有效**：Few-shot 示例放在规则前面
3. **动态上下文**：根据任务类型动态调整上下文结构
4. **主动修剪**：长对话要定期清理无关历史

### 实践清单

| 维度 | 检查项 |
|------|--------|
| **长度** | 上下文总 Token 是否在模型最佳窗口内？ |
| **相关性** | 每一段信息都与当前任务相关吗？ |
| **优先级** | 最重要的信息是否在最前面？ |
| **格式** | 结构化程度是否便于模型解析？ |
| **冗余** | 是否有重复信息可以删除？ |

---

## 类比

> Context Engineering ≈ 给 LLM 做「信息整理」
>
> 不是给越多越好，而是给**最关键**的。

就像一个好助理：不是把 100 封邮件打印出来给你，而是筛选出最紧急的 3 封，附上摘要和优先级。

---

## 延伸阅读

- [Effective Context Engineering for AI Agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [LangChain Context Engineering Guide](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [Lost in the Middle (论文)](https://arxiv.org/abs/2307.03172) — LLM 注意力分布的经典研究

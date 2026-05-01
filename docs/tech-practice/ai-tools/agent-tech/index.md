# Agent 技术串讲

从 LLM 到 Multi-Agent 的完整技术脉络。

> 本文整理自技术分享文档 **《近年 AI 应用技术串讲与优质文档分享》**。

---

## 技术栈层级

```
Layer 5   Multi-Agent    ← 多 Agent 协作
Layer 4   Agent          ← 思考-行动-观察循环
Layer 3   MCP / Tools    ← 标准化工具连接
Layer 2   RAG / Fine-tune ← 知识增强 / 能力定制
Layer 1   LLM + Prompt   ← 基础模型层
```

从下往上，每一层建立在前一层之上。理解这个层级是理解整个 AI Agent 生态的关键。

---

## 章节导航

| 章节 | 关键词 |
|------|--------|
| [1. LLM：大语言模型](llm.md) | Transformer、Self-Attention、三种架构 |
| [2. Prompt Engineering](prompt-engineering.md) | Few-shot、CoT、Role Prompting |
| [3. Fine-tuning：微调](fine-tuning.md) | LoRA、PEFT、Unsloth |
| [4. RAG：检索增强生成](rag.md) | 向量数据库、Hybrid Search、Embedding |
| [5. Function Calling](function-calling.md) | 工具调用、JSON Schema、结构化输出 |
| [6. MCP：Model Context Protocol](mcp.md) | 标准化协议、工具生态、Server/Client |
| [7. Agent：智能体](agent.md) | ReAct、Plan & Execute、Reflection |
| [8. Multi-Agent](multi-agent.md) | 分工协作、上下文隔离、Orchestrator |
| [9. Context Engineering](context-engineering.md) | 筛选、压缩、组织上下文 |

---

## 延伸阅读

| 论文 | 主题 |
|------|------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer 架构 |
| [LoRA](https://arxiv.org/abs/2106.09685) | 低秩微调 |
| [RAG](https://arxiv.org/abs/2005.11401) | 检索增强生成 |
| [ReAct](https://arxiv.org/abs/2210.03629) | Agent 推理与行动 |
| [MCP Spec](https://modelcontextprotocol.info/) | 模型上下文协议 |

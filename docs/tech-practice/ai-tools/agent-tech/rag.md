# 4. RAG：检索增强生成

> 让 LLM「查阅资料」后再回答，解决知识截止日期和幻觉问题。

**核心论文**：[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)（Lewis et al., 2020）

---

## 为什么需要 RAG？

LLM 有两个根本性问题：

1. **知识截止日期**：训练数据有截止时间，不知道之后发生的事
2. **幻觉**：不知道的事情会「编造」，且编得很自信

RAG 的解决思路：**先检索相关文档，再结合文档生成回答**。

```
用户提问 → 检索相关文档 → [文档 + 问题] → LLM → 回答
```

---

## RAG 的核心流程

### Step 1：文档处理（Offline）

```
原始文档 → 文本分割 (Chunking) → Embedding → 存入向量数据库
```

**Chunking 策略**：

| 策略 | 说明 | 适合 |
|------|------|------|
| 固定长度 | 按 token 数切分 | 通用 |
| 语义分割 | 按段落/章节切分 | 结构化文档 |
| 滑动窗口 | 有重叠地切分 | 防止切断上下文 |
| 父子文档 | 小 chunk 检索 + 大 chunk 喂给 LLM | 精确检索 + 完整上下文 |

### Step 2：检索（Online）

```
用户问题 → Embedding → 向量相似度搜索 → Top-K 相关文档
```

### Step 3：生成

```
[检索到的文档] + [用户问题] → Prompt → LLM → 带来源引用的回答
```

---

## 技术选型

| 组件 | 常用方案 |
|------|----------|
| **向量数据库** | Pinecone, Chroma, Milvus, Weaviate, Qdrant, pgvector |
| **Embedding 模型** | `text-embedding-3-small`(OpenAI), `bge-large`(BAAI), `gte-large` |
| **检索策略** | 语义检索 + 关键词混合（Hybrid Search） |
| **重排序** | Cohere Rerank, bge-reranker |
| **生成模型** | GPT-4, Claude, Qwen, DeepSeek |

---

## 进阶 RAG

| 技术 | 说明 |
|------|------|
| **Hybrid Search** | 语义 + 关键词混合检索，互补优势 |
| **Re-ranking** | 检索后再排序，提高 Top-K 精度 |
| **Self-RAG** | 模型自己判断是否需要检索、检索结果是否有用 |
| **Graph RAG** | 知识图谱 + RAG，适合关系复杂的知识 |
| **Agentic RAG** | Agent 自主决策检索策略、多轮检索 |

---

## RAG vs Fine-tuning

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时（改文档即可） | 需要重新训练 |
| 幻觉控制 | 有来源引用，可追溯 | 无明确来源 |
| 成本 | 检索 + LLM 推理 | 训练 + 推理 |
| 适合 | 知识密集型、需要溯源 | 行为/风格定制 |

> 💡 二者不是互斥的。很多生产系统同时使用 RAG（知识） + Fine-tuning（行为）。

---

## 延伸阅读

- [RAG 原始论文](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex](https://docs.llamaindex.ai/) — 专业 RAG 框架

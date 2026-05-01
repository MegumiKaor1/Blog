# 8. Multi-Agent：多智能体

> 多个 Agent 分工协作，解决单 Agent 搞不定的复杂问题。

**参考**：[Building Multi-Agent Systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)（Anthropic）

---

## 为什么需要 Multi-Agent？

### 单 Agent 的天花板

1. **上下文爆炸**：任务越复杂，上下文越长，LLM 注意力稀释
2. **能力范围有限**：一个 Agent 不可能擅长所有事
3. **单点故障**：一个环节出错全盘皆输
4. **串行瓶颈**：很多子任务本可以并行

### Multi-Agent 的解法

```
主 Agent（协调者 / Orchestrator）
    ├── Agent A：信息检索专家 → 搜索 + 读取文档
    ├── Agent B：数据分析专家 → 处理数据 + 画图
    ├── Agent C：写作专家    → 生成报告
    └── Agent D：审查专家    → 检查事实和逻辑
         ↓
    汇总结果 → 最终输出
```

---

## 核心架构模式

### 1. Orchestrator-Worker（主从模式）

最经典的架构，一个主 Agent 负责任务分解和调度。

```
Orchestrator: "这篇论文需要：1) 找相关工作 2) 分析实验 3) 写综述"
    → Worker 1: 搜 arXiv + Google Scholar
    → Worker 2: 统计实验指标
    → Worker 3: 撰写综述
Orchestrator: 整合 → 润色 → 输出
```

### 2. Peer-to-Peer（对等模式）

没有中心协调者，Agent 之间直接对话协作。

```
Agent A: "我发现了一个数据异常"
Agent B: "我检查了代码，没有 bug"
Agent A: "可能是数据预处理的问题"
Agent C: "让我重新跑一遍预处理"
```

### 3. Hierarchical（层级模式）

多层 Agent 嵌套，上层拆大任务，下层拆子任务。

```
CEO Agent
    ├── CTO Agent
    │   ├── Frontend Agent
    │   └── Backend Agent
    └── COO Agent
        ├── Data Agent
        └── Report Agent
```

---

## 关键设计原则

### Anthropic 的建议（来自 Building Multi-Agent Systems）

1. **从最简单的架构开始**：能单 Agent 就不要多 Agent
2. **每个 Agent 职责清晰**：「写代码的 Agent」比「干活的 Agent」定位更明确
3. **上下文隔离是关键优势**：子 Agent 不该看到主 Agent 的全部历史
4. **Handoff 要慎重**：什么时候交接、交接什么信息，都要明确
5. **不为了多 Agent 而多 Agent**：如果单 Agent + 好 Prompt 能搞定，就别加复杂度

---

## 需要警惕的问题

| 问题 | 说明 |
|------|------|
| **Token 消耗暴增** | 每个子 Agent 都有独立上下文，总开销是 N 倍 |
| **协作效率低** | Agent 间沟通不畅 → 重复劳动、信息丢失 |
| **系统复杂度高** | 调试困难，出问题难定位 |
| **权限失控** | 多个 Agent 同时操作文件/系统，容易冲突 |
| **幻觉传播** | 一个子 Agent 的幻觉可能污染整个流水线 |

---

## Multi-Agent 框架

| 框架 | 架构 | 特点 |
|------|------|------|
| **LangGraph** | 图结构 | 灵活编排、状态管理 |
| **CrewAI** | 角色分工 | 简单上手、适合团队模拟 |
| **AutoGen** | 对话驱动 | 微软出品、Agent 间对话 |
| **OpenAI Swarm** | 轻量 Handoff | 实验性质、简单优雅 |

---

## 延伸阅读

- [Building Multi-Agent Systems (Anthropic)](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [AutoGen](https://github.com/microsoft/autogen)

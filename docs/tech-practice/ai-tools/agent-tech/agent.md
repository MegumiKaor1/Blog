# 7. Agent：智能体

> 从「一问一答」到「自主完成复杂任务」。

**核心论文**：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)（Yao et al., 2022）

---

## 什么是 Agent

Agent 是一种能够基于目标进行「**思考 → 行动 → 观察**」循环、能够自主调用工具来完成复杂任务的智能系统。

### 与 Chatbot 的区别

| | Chatbot | Agent |
|------|---------|-------|
| 交互模式 | 一问一答 | 自主多步循环 |
| 工具使用 | 无 | 调用外部工具 |
| 任务复杂度 | 单轮对话 | 多步复杂任务 |
| 自主性 | 被动响应 | 主动规划和执行 |

---

## 核心循环：Think → Act → Observe

```
     ┌──────────┐
     │  思考    │ ← 分析当前状态，决定下一步
     │ (Think)  │
     └────┬─────┘
          ↓
     ┌──────────┐
     │  行动    │ ← 调用工具、执行操作
     │  (Act)   │
     └────┬─────┘
          ↓
     ┌──────────┐
     │  观察    │ ← 获取工具返回结果
     │(Observe) │
     └────┬─────┘
          ↓
     回到「思考」（直到任务完成或达到最大步数）
```

---

## 最小 Agent 公式

```
提示词 + LLM + Tools = Agent
```

三条缺一不可：

- **提示词**：定义目标、角色、行为规则
- **LLM**：推理和决策引擎
- **Tools**：与外部世界交互的能力

---

## Agent 设计模式

参考 [AI Agent Workflow Design Patterns](https://medium.com/binome/ai-agent-workflow-design-patterns-an-overview-cf9e1f609696)：

### ReAct（Reasoning + Acting）

交替进行推理和行动。最经典的模式。

```
Thought: 我需要知道今天天气才能建议穿衣
Action: get_weather("北京")
Observation: 晴，22°C
Thought: 晴天且 22°C，建议穿薄外套
Final Answer: 今天北京晴，22°C，建议穿薄外套
```

### Plan & Execute

先制定完整计划，再逐步执行。

```
Plan:
  1. 搜索目标公司信息
  2. 读取财报 PDF
  3. 分析关键指标
  4. 生成分析报告
Execute: 1 → 2 → 3 → 4
```

**优势**：整体规划避免跑偏；**劣势**：计划可能一开始就有误。

### Reflection

执行后反思自己的输出，自我改进。

```
生成初稿 → 自我评价 → 发现问题 → 修正 → 再评价 → 最终输出
```

### Tool Use

核心不是「什么时候用工具」，而是「用什么工具 + 怎么组合」。

### Routing

根据用户意图，动态分发给不同的处理流程：

```
用户输入 → Router → 技术问题？→ 技术 Agent
                   → 闲聊？  → 闲聊 Agent
                   → 代码？  → 代码 Agent
```

---

## Agent 框架对比

| 框架 | 特点 | 适合 |
|------|------|------|
| **LangChain / LangGraph** | 图结构编排，生态最全 | 复杂工作流 |
| **OpenAI Agents SDK** | 轻量，原生支持 Swarm | 快速原型 |
| **CrewAI** | 角色分工，简单上手 | 多 Agent 团队 |
| **Hermes Agent** | 自我进化，记忆分层 | 长期陪伴型 |
| **AutoGen** | 微软出品，对话驱动 | 多 Agent 对话 |

---

## 延伸阅读

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)（Anthropic）
- [Agent Design Patterns](https://medium.com/binome/ai-agent-workflow-design-patterns-an-overview-cf9e1f609696)

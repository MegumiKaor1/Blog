# Harness Engineering：驾驭工程

> 给 Agent 套上缰绳，让它在约束下高效奔跑。

---

## 什么是 Harness Engineering

**参考**：[OpenAI: Harness Engineering](https://openai.com/zh-Hans-CN/index/harness-engineering/)

Harness Engineering 强调通过**构建受控环境**，让 Agent 在约束下**高效、可靠地完成长周期复杂任务**。

**核心思想**：不是给 Agent 无限自由，而是给它**恰到好处的约束**。

---

## 为什么需要 Harness

裸 Agent 面临的问题：

| 问题 | 表现 |
|------|------|
| **跑偏** | 长任务中逐渐偏离原始目标 |
| **无限循环** | 工具调用死循环，不停重试 |
| **上下文爆炸** | Token 消耗失控，超出上下文窗口 |
| **幻觉** | 捏造不存在的工具调用结果 |
| **成本失控** | 一次长任务消耗大量 API 费用 |

Harness Engineering 就是针对这些问题的一系列工程实践。

---

## 核心实践

### 1. 约束机制

```
┌─────────────────────────────────────────┐
│              Harness 层                  │
│                                         │
│  ✅ 最多调用 10 次工具                    │
│  ✅ 单次响应不超过 2000 Token             │
│  ✅ 只能调用白名单中的工具                 │
│  ✅ 超时 60 秒自动终止                    │
│                                         │
│              ↓                           │
│           Agent                          │
└─────────────────────────────────────────┘
```

### 2. 反馈回路

```
Agent 行动 → 结果检查 → 符合预期？
                ├── 是 → 继续
                └── 否 → 自动修正或人工介入
```

### 3. 可靠上下文管理

| 策略 | 说明 |
|------|------|
| **关键信息提取** | 从工具返回中只保留对决策有用的部分 |
| **上下文预算** | 为不同类型信息分配 Token 配额 |
| **定期总结** | 长对话中每 N 轮进行一次压缩总结 |
| **优先级排序** | 最近、最相关的信息放在上下文最前面 |

### 4. 错误恢复

```python
# 典型的 Harness 错误处理模式
max_retries = 3
for attempt in range(max_retries):
    try:
        result = agent.execute(task)
        if validate(result):
            break
    except Exception as e:
        if attempt == max_retries - 1:
            escalate_to_human(e)
        else:
            agent.reflect_and_retry(e)
```

---

## 与 Agent 评测的关系

Harness Engineering 的名字也与 **Agent 评测框架**（如 lm-evaluation-harness）有关联：

| | Harness Engineering | Evaluation Harness |
|------|------|------|
| 目的 | 生产环境约束 | 评测模型能力 |
| 关注点 | 可靠性、成本、效率 | 准确率、性能指标 |
| 共同点 | 都在「套住」Agent 的行为 |

---

## 实践建议

| 阶段 | 建议 |
|------|------|
| **开发阶段** | 宽松约束，观察 Agent 自然行为 |
| **测试阶段** | 逐步收紧约束，找到最佳平衡点 |
| **生产阶段** | 严格约束 + 异常告警 + 人工兜底 |

---

## 延伸阅读

- [OpenAI: Harness Engineering 官方文章](https://openai.com/zh-Hans-CN/index/harness-engineering/)
- [Agent 技术串讲](agent-tech/index.md) — 理解 Agent 基础
- [Skill 机制与开发](skill-dev.md) — Skills 是 Harness 的天然搭档

# 5. Function Calling：函数调用

> 让 LLM 从「只会说话」变成「会调用工具」。

---

## 什么是 Function Calling

Function Calling 是一种机制，让大模型**按约定格式输出调用指令**，由外部系统真正去执行具体操作。

### 一个例子

```
用户：「今天北京天气怎么样？」

模型不直接编造回答，而是输出：
{
  "function": "get_weather",
  "arguments": {"city": "北京", "date": "2026-04-29"}
}

外部系统执行 get_weather("北京") → 返回 {"temp": 22, "condition": "晴"}

模型结合结果：「北京今天晴，气温 22°C。」
```

---

## 本质

> Function Calling = 结构化的意图输出，不是 LLM 真正执行函数。

模型的职责：理解用户意图 → 选择合适的函数 → 填充正确的参数 → 输出结构化 JSON。

系统的职责：解析 JSON → 执行函数 → 把结果传回模型 → 模型生成最终回复。

---

## 实现方式

### OpenAI 风格

在 API 请求中传入 `tools` 参数，定义可用的函数：

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名"},
            "date": {"type": "string", "description": "日期 YYYY-MM-DD"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

### JSON Mode / Structured Output

更简单的方式是要求模型输出 JSON：

```markdown
请以 JSON 格式回答。如果用户问天气，输出：
{"action": "get_weather", "city": "城市名"}
如果是普通对话，输出：
{"action": "chat", "reply": "回复内容"}
```

### 框架级封装

| 框架 | 用法 |
|------|------|
| **LangChain** | `@tool` 装饰器 → Agent 自动调用 |
| **DSPy** | 声明式定义工具签名 |
| **OpenAI Agents SDK** | `@function_tool` 装饰器 |

---

## Function Calling → Agent

Function Calling 是 Agent 的**技术前提**。没有它，Agent 无法与外部世界交互。

```
Function Calling：单次调用-返回
    ↓ 加上循环和决策
Agent：思考 → 调工具 → 看结果 → 再思考 → 再调工具 → ...
```

---

## 最佳实践

| 原则 | 说明 |
|------|------|
| **描述要精确** | `description` 决定了模型会不会选这个函数 |
| **参数要明确** | 类型、约束、默认值、示例都要写清楚 |
| **一个函数一个职责** | 不要一个函数做多件事 |
| **错误处理** | 工具调用失败时，给模型友好的错误信息让它重试 |
| **安全约束** | 危险操作（删文件、发消息）需要审批机制 |

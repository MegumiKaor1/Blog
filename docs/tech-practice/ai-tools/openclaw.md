# OpenClaw 入门

> 开源、高可扩展的 AI Agent 框架，构建可自定义的私人 AI 助手。

---

## 什么是 OpenClaw

OpenClaw 是一款**开源、高可扩展的 AI Agent 框架**，基于 TypeScript 开发，核心用途是构建**可自定义的私人 AI 助手**。

**参考**：
- [Nanobot](https://github.com/HKUDS/nanobot) — 精简版 OpenClaw 实现，推荐入门阅读

---

## 核心创新

OpenClaw 的重要创新之一是**扩展了 Agent 的交互入口**：

| 传统 Agent | OpenClaw |
|------------|----------|
| 仅 Web/CLI 交互 | Web + CLI + **飞书** + Telegram + Discord + ... |

这意味着用户可以**通过飞书直接与自己的 AI Agent 对话**，无需额外客户端。

---

## 架构概览

```
用户（飞书/CLI/Web）
        ↓
  ┌─────────────┐
  │  Gateway     │  ← 多平台消息接入层
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │  Agent Core  │  ← 思考-行动-观察循环
  │  (Loop)      │
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │  Tools/Skills│  ← 工具调用层
  └──────┬──────┘
         ↓
    执行结果 → 返回用户
```

**技术栈**：TypeScript + Node.js，兼容 Claude / OpenAI API。

---

## 为什么选 OpenClaw

| 优势 | 说明 |
|------|------|
| **多平台** | 飞书、Telegram、Discord 等原生接入 |
| **可扩展** | 插件式架构，轻松添加新平台和工具 |
| **开源** | 代码完全可控，可自部署 |
| **TypeScript** | 类型安全，适合工程化项目 |
| **社区活跃** | 参考实现多（如 nanobot） |

---

## 快速开始

### 1. 克隆项目

```bash
# 精简版（推荐入门）
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env 填入 API Key 和平台配置
```

### 3. 启动

```bash
npm install
npm run dev
```

---

## 与 Hermes Agent 的关系

| | OpenClaw | Hermes Agent |
|------|----------|--------------|
| 语言 | TypeScript | Python |
| 定位 | 通用 Agent 框架 | AI 编程助手 |
| 多平台 | 飞书/Telegram/Discord 等 | 飞书（主要） |
| Skill 机制 | ✅ | ✅ |
| 开源 | ✅ | ✅ |

两者理念相通——都用 **Skill + Tool** 扩展 Agent 能力，都支持多平台接入。如果你用飞书，Hermes Agent 更简单；如果需要更通用的多平台 Agent 框架，OpenClaw 更合适。

---

## 延伸阅读

- [Nanobot (精简 OpenClaw)](https://github.com/HKUDS/nanobot)
- [Agent 技术串讲](agent-tech/index.md)
- [Hermes Agent 完全指南](hermes-agent.md)

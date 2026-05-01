# 6. MCP：Model Context Protocol

> 让模型以**统一标准**连接任何外部工具和数据源。

**官方文档**：[MCP Introduction](https://modelcontextprotocol.info/docs/introduction/)

---

## 什么是 MCP

MCP（Model Context Protocol）是一种**开放标准协议**，由 Anthropic 提出，旨在让 AI 模型以统一的方式连接外部工具、数据源和服务。

### 核心问题

```
传统方式：每个工具单独写胶水代码
  LLM → 代码 A → 工具 A
  LLM → 代码 B → 工具 B    ← 每个工具都要写一遍
  LLM → 代码 C → 工具 C

MCP 方式：
  工具 A → MCP Server ↘
  工具 B → MCP Server → MCP Client → 任何 AI 应用
  工具 C → MCP Server ↗
```

**最重要的贡献**：使工具可以跨 AI 应用复用。一个 MCP Server 写好后，Claude Desktop、Hermes Agent、Cursor 都可以用。

---

## 核心概念

### MCP Server

提供工具/资源的一方。例如：

| MCP Server | 提供的工具 |
|------------|-----------|
| `@anthropic/mcp-server-filesystem` | 文件读写、目录操作 |
| `@anthropic/mcp-server-puppeteer` | 浏览器操控 |
| `@modelcontextprotocol/server-github` | GitHub Issues/PR/Repo |
| `mcp-server-feishu` | 飞书文档读写 |

### MCP Client

使用工具的一方 —— AI 应用（Claude Desktop、Hermes、Cursor 等）。

### 传输方式

| 方式 | 说明 |
|------|------|
| **stdio** | 标准输入输出，本地进程通信 |
| **HTTP + SSE** | 远程服务，支持 Server-Sent Events |

### 一个 MCP Server 的最小实现

```python
# server.py
from mcp.server import Server, stdio_server
from mcp.types import Tool, TextContent

server = Server("my-server")

@server.tool()
async def hello(name: str) -> list[TextContent]:
    return [TextContent(type="text", text=f"Hello, {name}!")]

stdio_server.run(server)
```

---

## MCP 生态

| 资源 | 链接 |
|------|------|
| 官方规范 | https://modelcontextprotocol.info/ |
| 官方 Server 列表 | https://github.com/modelcontextprotocol/servers |
| MCP Hub (社区) | https://mcp.so/ |
| Hermes MCP 集成 | 见 [Hermes Agent 完全指南](../hermes-agent.md) |

---

## MCP vs 传统 Function Calling

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 标准化 | 各家 API 格式不同 | 统一协议 |
| 可复用 | 每个应用单独集成 | 一次开发到处用 |
| 动态发现 | 工具列表写死在代码里 | Server 启动时自动注册 |
| 生态 | 各自为战 | 社区贡献 Server |

---

## 延伸阅读

- [MCP 官方文档](https://modelcontextprotocol.info/docs/)
- [Anthropic MCP Blog](https://www.anthropic.com/news/model-context-protocol)
- [Building MCP Servers](https://modelcontextprotocol.info/docs/developers/)

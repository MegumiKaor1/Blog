# Agent Skill 机制与开发

> 离职的同事终将化作温暖的 Skill。

---

## 什么是 Agent Skill

**Agent Skills** 是一种**轻量级的开放格式**，用于将一整套 Agent 能力（提示词、工具脚本、知识文件等）封装为**可复用模块**。

**参考**：
- [Equipping Agents for the Real World with Agent Skills](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills)
- [Agent Skills 官方文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Agent Skills 注册中心](https://agentskills.io/home)

---

## Agent Skill ≈ 子 Agent

一个 Skill 本质上约等于一个**子 Agent**：有自己专属的提示词、工具和知识，但被主 Agent 按需激活。

```
主 Agent（协调者）
 ├── Skill A: 代码审查（激活 → 执行 → 返回结果 → 休眠）
 ├── Skill B: 数据分析（激活 → 执行 → 返回结果 → 休眠）
 └── Skill C: 文档生成（激活 → 执行 → 返回结果 → 休眠）
```

---

## Skills 与 SOP

Agent Skill 特别适合 **SOP（标准作业流程）的沉淀和复用**。

| 场景 | 示例 |
|------|------|
| 代码审查 SOP | 检查安全漏洞 → 检查代码规范 → 生成审查报告 |
| 论文阅读 SOP | 提取核心贡献 → 分析方法论 → 评估局限性 → 写笔记 |
| 部署 SOP | 拉代码 → 跑测试 → 构建 → 部署 → 健康检查 |

**每次 SOP 都可以封装成一个 Skill**，新人来了直接加载，不用重新教。

---

## 渐进式披露（Progressive Disclosure）

Agent 在运行过程中**按需激活**不同 Skills，按需读取 Skill 文件包里内容——不是一开始就把所有 Skill 全加载进上下文。

```
工作流程：
1. 用户请求 → 分析意图
2. 匹配相关 Skill → 激活该 Skill
3. 加载 Skill 的提示词 + 相关文件
4. 执行任务 → 返回结果
5. Skill 退出上下文（释放 Token）
```

**优势**：节省 Token、提高专注度、避免上下文污染。

---

## Skill 文件结构

典型的 Skill 目录：

```
my-skill/
├── SKILL.md           # 主文件：名称、描述、触发条件、使用说明
├── references/        # 参考资料（知识文档）
│   └── api-docs.md
├── scripts/           # 可执行脚本
│   └── run_analysis.py
├── templates/         # 模板文件
│   └── report_template.md
└── assets/            # 静态资源
    └── logo.png
```

**SKILL.md 核心字段**：

```yaml
---
name: code-review
description: 执行代码审查，检查安全和代码规范
triggers:
  - "审查代码"
  - "code review"
  - "检查这段代码"
---
# 代码审查 Skill

## 工作流程
1. 安全检查（注入、XSS、敏感信息泄露）
2. 代码规范检查（命名、结构、注释）
3. 生成审查报告
...
```

---

## 与 Hermes Agent 的关系

Hermes Agent 中的 Skill 机制直接遵循这一设计理念：

```bash
~/.hermes/skills/
├── code-review/
│   └── SKILL.md
├── arxiv/
│   ├── SKILL.md
│   └── scripts/
│       └── search.py
└── obsidian/
    ├── SKILL.md
    └── references/
        └── vault-structure.md
```

Agent 在对话中检测到匹配的触发词后自动加载对应 Skill。详见 [Hermes Agent 完全指南](hermes-agent.md)。

---

## 开发一个 Skill 的步骤

### 1. 明确 SOP

先确定这个 Skill 要完成什么固定流程。

### 2. 写 SKILL.md

```markdown
---
name: my-skill
description: 一句话描述
triggers: ["触发词1", "触发词2"]
---

# Skill 名称

## 背景
...

## 工作流程
1. ...
2. ...

## 输出格式
...
```

### 3. 添加脚本（可选）

```python
# scripts/helper.py
def do_something(data):
    ...
```

### 4. 放到正确位置并测试

```bash
cp -r my-skill ~/.hermes/skills/
# 在对话中输入触发词测试
```

---

## 最佳实践

| 原则 | 说明 |
|------|------|
| **单一职责** | 一个 Skill 只做一件事 |
| **自包含** | 不依赖外部状态，所有需要的信息打包进 Skill |
| **可组合** | Skills 之间可以协作（A 的输出 → B 的输入） |
| **渐进式** | 先做最小可用版本，再迭代完善 |
| **文档化** | 写清楚触发条件和使用方式 |

---

## 延伸阅读

- [Agent Skills 注册中心](https://agentskills.io/home) — 浏览社区 Skills
- [Anthropic: Agent Skills 官方博客](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills)
- [Hermes Agent 完全指南](hermes-agent.md) — 在你的 Hermes 中创建 Skill

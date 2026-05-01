# Hermes Agent 完全指南

> "The agent that grows with you" —— 一个会自我进化的 AI 编程助手。

本文基于 **《Hermes Agent 详细讲解 & 本地部署和基于 OpenClaw 迁出的全流程》** 整理。

---

## 1. 什么是 Hermes Agent

**来源**：[Nous Research](https://github.com/nousresearch/hermes-agent) 开源的自主 AI Agent 框架，2026 年 2 月底发布，上线不到两个月 GitHub Star 近 3 万。

**核心定位**：「与你共同成长的 Agent」，最大的特点是**可以自我进化**。

**相关链接**：

| 资源 | 地址 |
|------|------|
| GitHub | https://github.com/nousresearch/hermes-agent |
| 官网 | https://hermes-agent.nousresearch.com/ |
| 官方文档 | https://hermes-agent.nousresearch.com/docs |
| 非官方使用指南 | https://github.com/alchaincyf/hermes-agent-orange-book |
| Hermes vs OpenClaw 对比 | https://lushbinary.com/blog/hermes-vs-openclaw-key-differences-comparison/ |

---

## 2. Hermes Agent vs OpenClaw

### 2.1 OpenClaw 为什么火了

**技术层面**：

1. **产品定位精准（私人数字员工）**
   - 之前的主流 AI 产品停留在对话顾问层面（ChatGPT、Kimi、豆包等），即便是 Cursor、Claude Code 这类 Coding Agent，能力也局限在 IDE 或终端内部
   - OpenClaw 拥有操作系统级权限，可以操控浏览器、收发邮件、执行系统命令、写代码、跑代码

2. **支持多平台接入（飞书、微信等）**
   - 不再局限于浏览器网页或终端命令行
   - 通过飞书、Telegram 等与 Agent 交互

3. **本地部署，隐私安全**
   - 模型调用之外的数据全部保存在本地
   - 解决金融、法律、医疗等行业的隐私刚需

**非技术层面**：多方共赢（云厂商赚钱、Mac mini 卖断货、新职业诞生）、时机完美（大模型能力已到位但缺现象级应用）。

### 2.2 OpenClaw 的问题

| 问题 | 说明 |
|------|------|
| **安全风险大** | 系统级权限失控 → 删邮件、删文件、泄密；Skills 市场中 10.8% 的插件含恶意代码 |
| **稳定性不足** | 指令理解有限、容易跑偏、经常断联 |
| **虚假低门槛** | 「一键安装」的前提是已装 Homebrew、Node.js、Git |
| **成本容易失控** | 陷入死循环时疯狂烧 Token |
| **成熟度不够** | 仍处实验阶段，每次升级引入一堆 bug |

### 2.3 Hermes Agent 的核心优势

#### 2.3.1 💡 自我进化系统（最核心）

这是 Hermes Agent 区别于所有其他开源 Agent 的**灵魂**：

1. **技能自主创建**：完成复杂任务后（≥5 次工具调用），Agent 自动把执行过程沉淀为 Skill
2. **技能自主迭代**：后续执行中发现更优解法会自动更新 Skill
3. **无需人工确认**：创建和更新都是自动的（建议定期 review 本地技能目录）

```
OpenClaw：技能靠人工编写 + 社区市场
Hermes：  技能靠自己生长

优势：
✅ Skill 贴合你的实际工作（格式偏好、命名习惯等）
✅ Skill 持续进化不会过时（自带维护机制）
✅ 规避外部 Skill 的安全风险（本地生成，不引入第三方）
```

#### 2.3.2 🔒 更强的安全性

| 防护层 | 机制 |
|--------|------|
| **防恶意 Skill 注入** | Skill 主要自己生成 + Skill 扫描机制 |
| **防 Prompt 注入** | 子 Agent 的中间推理过程不透传到父 Agent |
| **防敏感信息泄露** | 传递给子进程的环境变量经过过滤，只保留基线变量 |
| **防 Agent 失控** | 危险命令审批（manual/smart/off 三档）+ 命令级扫描 + 沙箱执行 |
| **上下文文件扫描** | AGENTS.md、SOUL.md 加入系统提示前先扫描风险 |

#### 2.3.3 📦 安装门槛低

```bash
# 真正一行命令安装
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

安装脚本自动处理所有依赖：Python 3.11、Node.js、uv 包管理器、ripgrep、ffmpeg 等。

**从 OpenClaw 无缝迁移**：

```bash
hermes claw migrate
```

一键导入你的 persona、memory、skills、configs 和 API keys。

#### 2.3.4 👀 运行过程可观测

通过 `display.tool_progress` 控制执行过程展示粒度：

| 级别 | 内容 |
|------|------|
| `off` | 只看最终结果 |
| `new` | 只看新类型工具调用（推荐日常用） |
| `all` | 看所有工具调用 |
| `verbose` | 连模型推理过程都展示（调试用） |

完整执行轨迹记录到本地 SQLite，可以随时回放追查每一步。

#### 2.3.5 💰 更谨慎地烧 Token

| 策略 | 说明 |
|------|------|
| **子任务用便宜模型** | 主对话跑强模型，委派任务跑便宜模型 |
| **上下文隔离** | 子 Agent 不继承对话历史，每次干净上下文 |
| **上下文自动压缩** | 长对话自动触发压缩，用便宜模型压缩早期对话成摘要 |
| **Prompt Caching** | 原生支持 Anthropic 缓存机制，缓存部分打折扣 |

#### 2.3.6 🧠 更优秀的记忆架构

```
Hermes 的记忆分层：

1. 短期上下文（当前会话）
   └─ 当前对话原始消息，LLM 直接看到

2. 向量检索（语义记忆）
   └─ 基于 embedding 的相似度搜索
      「你之前提过的那个想法是什么来着？」

3. FTS5 全文搜索（精确检索）
   └─ SQLite 内置全文索引，关键词精确匹配
      「搜一下我上周说的那个项目名」

4. MEMORY.md / USER.md（结构化记忆）
   └─ 跨会话持久化，每次对话自动注入系统提示

5. Honcho（用户建模）
   └─ 从行为中推理出你没说的东西
```

OpenClaw 的记忆偏存储，Hermes 的记忆架构为「长期陪伴型数字员工」设计。

#### 2.3.7 🔄 更适合长期运行

| 问题 | Hermes 的解决方案 |
|------|------------------|
| 崩溃后状态丢失 | SQLite 会话状态存储，`/resume` 恢复 |
| 长任务阻塞聊天 | `/background` 后台独立会话运行 |
| 用完就忘 | 自动沉淀经验成 Skill + 周期性 nudge |
| 想多平台在线 | `hermes gateway install` 注册系统服务 |
| 想定时自动执行 | 内置 Cron 调度器，自然语言配置 |

---

## 3. 安装与部署

### 3.1 Mac 从零部署

```bash
# 1. 安装
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 2. 重载 shell
source ~/.zshrc

# 3. 验证
hermes version

# 4. 运行配置向导
hermes setup
```

配置向导引导你完成：
1. 选择 LLM 服务商和模型（必须）
2. 配置工具集（可选）
3. 配置消息网关（可选）

也可以分步配置：
```bash
hermes model        # 单独配置模型
hermes tools        # 单独配置工具
hermes gateway setup # 单独配置消息网关
```

### 3.2 接​​入飞书

**第一步：飞书开放平台创建应用**

1. 去 https://open.feishu.cn/app 创建企业自建应用
2. 添加机器人能力
3. 开通权限：获取与发送单聊/群组消息、获取用户基本信息、读取用户发给机器人的消息
4. 事件与回调 → 添加接收消息事件
5. 发布版本
6. 拿到 App ID 和 App Secret

**第二步：在 Hermes 配置飞书网关**

```bash
hermes setup gateway
```

上下移动到飞书 → 空格选中 → 回车确认 → 输入 App ID 和 App Secret。

**第三步：飞书和 Hermes 配对**

```bash
# 在飞书中随便给机器人发一条消息，拿到配对码
# 在终端中执行（注意：是终端，不是对话内）
hermes pairing approve feishu <配对码>
# 显示 Approved 即为配对成功
```

### 3.3 Mac 从 OpenClaw 迁移

Hermes 内置一键迁移命令：

```bash
# 交互式迁移（推荐）
hermes claw migrate

# 先预览
hermes claw migrate --dry-run

# 完整迁移
hermes claw migrate --preset full --yes

# 仅迁移用户数据
hermes claw migrate --preset user-data

# 覆盖已有冲突项
hermes claw migrate --overwrite
```

**迁移内容**：人格配置（SOUL）、长期记忆、Skills、模型配置、通讯平台配置、API 密钥。

**迁移后验证**：
```bash
hermes doctor           # 检查运行状态
hermes                  # 启动对话测试
hermes skills list       # 验证 Skill 迁移
hermes memory search "关键词" # 验证记忆迁移
```

> ⚠️ 同一消息通道同一时间只能由一个进程监听，先完全关闭 OpenClaw 再启动 Hermes Gateway。

### 3.4 Windows 部署（WSL2）

```powershell
# 以管理员身份打开 PowerShell
wsl --install

# 重启后打开 Ubuntu，创建 Linux 用户名和密码
# 配置代理（如果位于中国大陆）
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890

# 一键安装
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 重载配置
source ~/.bashrc
```

此后所有步骤同 Mac。

**常见问题**：
- 找不到 `hermes` 命令 → 关闭终端重开，或 `source ~/.bashrc`
- GitHub 下载超时 → 确认代理配置正确
- WSL2 无法联网 → 检查防火墙 + 代理 Allow LAN
- Windows 原生实验性安装：`irm https://raw...install.ps1 | iex`（部分功能受限）

#### 3.4.1 WSL2 代理配置详解

WSL2 的代理配置比普通 Linux 麻烦得多——**它和 Windows 不在同一个网络里**，所以 `127.0.0.1` 在 WSL2 内指向的是 Linux 自己，而不是 Windows 宿主机。这是最常见的坑。

**WSL2 网络架构**

```
┌──────────────────────────────────────┐
│           Windows 宿主机              │
│  ┌────────────────────────────────┐  │
│  │  代理软件 (Clash / v2ray / etc) │  │
│  │  HTTP: 127.0.0.1:7890          │  │
│  │  SOCKS: 127.0.0.1:7891         │  │
│  └────────────────────────────────┘  │
│              ↕ NAT                     │
│  ┌────────────────────────────────┐  │
│  │        WSL2 虚拟机               │  │
│  │  127.0.0.1 → 指向 WSL 自己 ❌    │  │
│  │  需要访问 Windows 宿主机的 IP ✅  │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

**方案一：获取 Windows 宿主机 IP（传统方式）**

WSL2 内部可以通过 `/etc/resolv.conf` 中的 nameserver 拿到宿主机的虚拟 IP：

```bash
# 查看 Windows 宿主机在 WSL2 中的 IP
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# 通常是 172.x.x.x

# 手动设置代理
export host_ip=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
export https_proxy="http://${host_ip}:7890"
export http_proxy="http://${host_ip}:7890"
export all_proxy="socks5://${host_ip}:7891"  # 如果需要 SOCKS5
```

但 Windows 宿主 IP 每次重启都可能变化。解决方法：把上面的逻辑写进 `~/.bashrc`：

```bash
# 加入 ~/.bashrc
export host_ip=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
export https_proxy="http://${host_ip}:7890"
export http_proxy="http://${host_ip}:7890"
```

**方案二：WSL2 镜像网络模式（推荐，需 Windows 11 23H2+）**

这是微软在 2023 年底推出的新特性——让 WSL2 直接**镜像** Windows 的网络栈，两边共享同一个 `localhost`。开启后，WSL2 内的 `127.0.0.1` 直接指向 Windows 宿主机，不需要再手动查 IP。

在 Windows 用户目录下创建/编辑 `C:\Users\<你的用户名>\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

然后重启 WSL：

```powershell
wsl --shutdown
wsl
```

配置后：
- `127.0.0.1:7890` 在 WSL2 内直接指向 Windows 上的代理 ✅
- 不需要每次查宿主 IP、不需要写进 `.bashrc`
- `localhost` 在 Windows 和 WSL2 之间完全互通

> **限制**：镜像模式需要 Windows 11 23H2 或更高版本。如果你的 Windows 10 不能升级，用方案一。

**方案三：代理客户端开启 Allow LAN + 防火墙放行**

无论用方案一还是方案二，还需要确保代理客户端本身允许来自「局域网」的连接：

| 代理客户端 | 设置位置 |
|-----------|---------|
| **Clash Verge** | 设置 → 允许局域网连接（Allow LAN） |
| **v2rayN** | 参数设置 → 允许来自局域网的连接 |
| **Clash for Windows** | General → Allow LAN |
| **Quantumult X** | 设置 → 代理 → 共享代理 |

同时确保 Windows 防火墙放行代理端口——大多数代理客户端首次开启 Allow LAN 时会自动处理，但如果你手动改了端口，需要去「Windows 防火墙 → 入站规则」确认。

**方案四：一键代理脚本**

不想每次手动配？把下面这段保存为 `~/proxy.sh`：

```bash
#!/bin/bash
# WSL2 代理一键切换脚本

PROXY_PORT=${1:-7890}  # 默认 7890，可通过参数覆盖

set_proxy() {
    local host_ip=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
    export https_proxy="http://${host_ip}:${PROXY_PORT}"
    export http_proxy="http://${host_ip}:${PROXY_PORT}"
    export all_proxy="socks5://${host_ip}:7891"
    echo "✅ 代理已开启: ${host_ip}:${PROXY_PORT}"
}

unset_proxy() {
    unset https_proxy http_proxy all_proxy
    echo "✅ 代理已关闭"
}

# 使用: source ~/proxy.sh 开启, source ~/proxy.sh off 关闭
if [ "$1" = "off" ]; then
    unset_proxy
else
    set_proxy
fi
```

用法：

```bash
source ~/proxy.sh        # 开启代理（默认端口 7890）
source ~/proxy.sh 10809  # 用自定义端口
source ~/proxy.sh off    # 关闭代理
```

#### 3.4.2 WSL2 常见网络问题排查

| 现象 | 可能原因 | 排查步骤 |
|------|---------|---------|
| `curl` 超时 | 代理没开或 IP 不对 | `echo $https_proxy` 确认变量已设置；`curl -v https://google.com` 看卡在哪一步 |
| 能 ping 通百度但 git clone 失败 | 代理只配了 http 没配 https | 同时设 `http_proxy` 和 `https_proxy` |
| 每次重启 WSL 代理失效 | 宿主 IP 变了 | 用镜像模式（方案二）或自动获取 IP（方案一的 `.bashrc` 写法） |
| 代理端口拒绝连接 | 防火墙或 Allow LAN 未开启 | 检查代理客户端 Allow LAN；Windows 防火墙放行端口 |
| DNS 解析失败 | WSL2 DNS 配置问题 | `echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf`（临时）；或 `dnsTunneling=true`（永久） |
| 公司 VPN 环境下 WSL 完全无网 | VPN 接管了路由表 | 关闭 VPN 再试；或使用 `wsl --shutdown` 后重启 |

### 3.5 双系统配置（Mac + Windows）

如果你同时有 Mac（日常开发）和 Windows GPU 机器（训练），推荐**双飞书机器人架构**：

```
飞书用户（你）
    │
    ├── 日常开发、写作、研究 → Hermes-Mac（Mac 上的飞书机器人）
    │
    └── GPU 训练、大任务    → Hermes-Win（Windows 上的飞书机器人）
```

**配置步骤**：

**第 1 步：两台机器分别安装 Hermes**

| 机器 | 安装方式 |
|------|----------|
| Mac | `curl -fsSL ...install.sh \| bash`（见 3.1） |
| Windows | WSL2 → 同 Mac 安装流程（见 3.4） |

**第 2 步：分别创建两个飞书应用**

在飞书开放平台创建**两个独立的机器人应用**，分别拿到各自的 App ID 和 App Secret。

| 机器人 | 用途 | App ID |
|--------|------|--------|
| Hermes-Mac | 日常对话 | `cli_xxx_mac` |
| Hermes-Win | GPU 任务 | `cli_xxx_win` |

**第 3 步：各自配置网关**

- Mac 上：`hermes setup gateway` → 配 Mac 机器人的 App ID
- Windows 上：`hermes setup gateway` → 配 Win 机器人的 App ID

**第 4 步：飞书中分别配对**

向两个机器人都发一条消息，拿到各自的配对码，分别在各自的终端里执行：
```bash
hermes pairing approve feishu <配对码>
```

**使用策略**：

| 任务类型 | 发给 |
|----------|------|
| 写代码、读论文、写博客 | Hermes-Mac |
| GPU 训练、大数据处理 | Hermes-Win |
| 跨机器人协作 | 手动转发消息 + 共享 Obsidian Vault |

> **跨机器人共享上下文**：两台机器的 Hermes 读写同一个 Obsidian Vault（通过 iCloud / Syncthing / Git 同步），这样 Skill 和记忆可以部分共享。

---

## 4. SOUL.md 配置指南

SOUL.md（`~/.hermes/SOUL.md`）是 Hermes 的「人格文件」，类似 System Prompt，**每次对话开始时自动注入系统提示**。写得越具体，Agent 越快理解你的需求和风格。

### 4.1 基础模板

```markdown
# 我的 Hermes Agent Persona

## 关于我
- 姓名/称呼：Kaor1
- 角色：机器学习研究员 / 博士生
- 主要研究方向：ML 安全、模型遗忘、扩散模型、数据投毒

## 交流偏好
- 默认用中文回复
- 重要结论先说，再解释原因
- 对技术问题要具体、直接，不要空话
- 不确定的地方明确说要验证什么

## 工作习惯
- 用 Obsidian 记录科研想法，偏好 Markdown 输出
- 先快速形成可执行版本，再逐步迭代
- 偏好简洁方案，不要过度设计

## 讨厌的行为
- 不要用「这是一个有趣的方向」这类空话
- 不要只罗列概念不建立联系
- 不要在没有证据时做绝对判断
```

### 4.2 进阶：按场景定义行为

```markdown
## 科研任务偏好
- 讨论论文 idea：先分析研究问题、动机、威胁模型、方法设计
- 实验设计：说明目的、变量、对照组、指标、预期、可能失败原因
- 论文写作：给我可以直接放入论文或 Obsidian 的版本

## 编程任务偏好
- 优先给出可运行代码
- 审查代码重点：bug、方法一致性、实验设置正确性、数据泄漏
- 调试不要只猜测，给具体检查命令和定位步骤

## 论文阅读输出格式
1. 一句话核心贡献
2. 与我的研究方向的关系
3. 方法论结构（问题→假设→方法→设计）
4. 关键局限
5. 对我的研究的启发
```

### 4.3 核心原则

| 原则 | 说明 |
|------|------|
| **写得具体** | 不要写「帮我写代码」，写清楚语言、框架、风格要求 |
| **写你的习惯** | 命名规则、文件组织方式、偏好格式 |
| **写你的痛点** | 比如「不要过度解释基础概念，假设我已经懂了」 |
| **持续迭代** | 用一段时间后根据实际体验修改 SOUL.md |

> 💡 `~/.hermes/SOUL.md` 和 `USER.md` 的区别：SOUL.md 定义 Agent **如何回应你**（行为偏好），USER.md 记录 Agent **应该了解你的什么**（个人信息、项目背景）。两者都每次注入系统提示。

---

## 5. 日常使用技巧

Agent 执行了不满意的操作，`/rollback` 回退到上一个文件系统检查点，不需要手动 undo。

### 4.3 按任务复杂度切换模型

```bash
hermes model  # 随时切换，不需改代码
```

Hermes 也支持**智能模型路由**，自动判断当前任务是否需要更强模型。

### 4.4 Skills 管理

```bash
hermes skills list         # 查看已安装技能
hermes skills check        # 检查更新
hermes skills update       # 更新有变动的技能

# 对话中
/skills browse             # 浏览可用技能
/skills search react       # 搜索特定技能
```

### 4.5 搜索历史会话

```bash
# 对话中
/search 关键词

# 终端中
hermes memory search "关键词"
```

### 4.6 /compact 压缩上下文

对话变长后 token 消耗急剧上升，`/compact` 自动压缩中间部分对话，保留头尾关键内容。

### 4.7 hermes doctor 排查

遇到问题第一反应：
```bash
hermes doctor  # 自动检查环境、依赖、API 连通性
```

---

## 6. 命令速查

### 对话中

| 命令 | 说明 |
|------|------|
| `/new` | 新对话（清空上下文，不影响记忆和 Skill） |
| `/clear` | 清空屏幕显示 |
| `/approve` | 批准高危命令 |
| `/rollback` | 回退到上一个检查点 |
| `/compact` | 压缩上下文 |
| `/background` | 后台执行任务 |
| `/search` | 搜索历史会话 |
| `/skills browse` | 浏览可用 Skill |
| `/personality <名称>` | 切换对话人格 |

### 终端

| 命令 | 说明 |
|------|------|
| `hermes` | 启动交互式对话 |
| `hermes setup` | 运行完整配置向导 |
| `hermes model` | 选择模型 |
| `hermes tools` | 配置工具集 |
| `hermes gateway setup` | 配置消息网关 |
| `hermes claw migrate` | 从 OpenClaw 迁移 |
| `hermes skills list` | 查看已安装 Skill |
| `hermes update` | 更新到最新版本 |
| `hermes doctor` | 诊断环境和配置 |
| `hermes version` | 查看版本 |
| `hermes config show` | 查看当前配置 |

---

## 7. 使用案例

### 基础用法

```
帮我上网看看最近很火的 Hermes Agent，然后总结成文档，
注意文档里需要附上和技术概念有关的重要链接。
接着，帮我生成一个静态的博客页面，专门用来展示这些技术文档。
注意：网页要高级好看点。
完成之后，帮我本地启动前端页面，给我一个可访问的 URL。
```

### 创意用法

```
帮我做一个「测测你是海绵宝宝里的谁」互动测试网页（纯前端）。
⚠ 核心规则：主角只能是极少数天选之人才能测到的隐藏结果。
绝大多数人（80%+）测出来都应该是比奇堡的路人甲、背景鱼、龙套道具。
```

---

## 8. 卸载

### 卸载 Hermes

```bash
hermes uninstall
```

两种模式：
- **Keep data**：仅删除程序，保留 `~/.hermes/` 配置和记忆（推荐）
- **Full uninstall**：删除所有内容，不可逆

### 卸载 OpenClaw

```bash
openclaw uninstall --all --yes --non-interactive
npm uninstall -g openclaw
```

---

## 延伸阅读

博客内相关文章：

- [Agent 技术串讲](agent-tech/index.md) — 从 LLM 到 Multi-Agent 的完整技术脉络
- [Skill 机制与开发](skill-dev.md) — Skill 原理、SOP 沉淀与开发实践
- [OpenClaw 入门](openclaw.md) — OpenClaw 框架入门与对比
- [Harness Engineering](harness.md) — 给 Agent 套上缰绳

---

## 参考资料与文档

- <a href="https://hermes-agent.nousresearch.com/docs" target="_blank" rel="noopener">Hermes Agent 官方文档</a>
- <a href="https://hermes-agent.nousresearch.com/docs/developer-guide/architecture" target="_blank" rel="noopener">Hermes Agent 架构设计</a>
- <a href="https://oigi8odzc5w.feishu.cn/wiki/NtOrwxnyWirN3ykWSoncFRr5nGc" target="_blank" rel="noopener">Hermes Agent 详细讲解 & 部署全流程（飞书文档）</a>
- <a href="https://lushbinary.com/blog/hermes-vs-openclaw-key-differences-comparison/" target="_blank" rel="noopener">Hermes vs OpenClaw 关键差异对比</a>
- <a href="https://github.com/alchaincyf/hermes-agent-orange-book" target="_blank" rel="noopener">非官方使用指南（橙皮书）</a>
- <a href="https://github.com/nousresearch/hermes-agent" target="_blank" rel="noopener">Hermes Agent GitHub</a>

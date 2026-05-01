# Kaor1's World — Agent 项目约定

> **AI Agent 速查手册**：下次在这个项目里工作的 AI（无论哪个模型、哪个平台），先读这个。

---

## 项目概览

- **名称**: Kaor1's World
- **类型**: 个人博客 / 知识库，使用 MkDocs + Material for MkDocs
- **仓库**: `github.com/MegumiKaor1/Blog`
- **本地路径**: `~/blog`
- **线上地址**: `https://megumikaor1.github.io/Blog`

---

## 环境约束（重要！）

- **Python 版本**: 3.9.6
- **mkdocs 调用方式**: 必须用 `python3 -m mkdocs`，不能直接用 `mkdocs`（不在 PATH 中）
- **Material for MkDocs 版本**: 9.7.6

### 本地预览命令

```bash
cd ~/blog
python3 -m mkdocs serve --dev-addr=0.0.0.0:8000
# 访问: http://localhost:8000/blog/
# 注意: 带 /blog/ 前缀，因为 site_url = https://megumikaor1.github.io/Blog
```

### 部署命令

```bash
cd ~/blog
mkdocs gh-deploy --force
```

---

## 目录结构

```
~/blog/
├── mkdocs.yml              # 全站配置
├── README.md               # 人类可读的 README
├── CLAUDE.md               # ← 本文件（AI agent 入口）
├── overrides/main.html     # Giscus 评论区模板覆盖
└── docs/
    ├── index.md            # 首页
    ├── ml-security/        # ML Security 笔记 (7 篇)
    ├── gen-models/         # Generative Models 笔记 (3 篇)
    ├── ai-math/            # AI 数学基础 (5 篇)
    ├── tech-practice/      # 技术实践
    │   ├── setup/          # 环境配置 (4 篇)
    │   ├── ai-tools/       # AI 工具链 (7 篇 + agent-tech 子目录 9 篇)
    │   ├── infra/          # AI Infra (4 篇)
    │   └── docs-share/     # 文档分享 (1 篇)
    ├── papers/             # 论文阅读 (1 篇索引)
    ├── life/               # 生活博客 (2 篇)
    └── assets/             # JS/CSS 资源
```

---

## 导航结构

博客有 8 个顶级导航项（见 `mkdocs.yml` 的 `nav:` 部分）:

1. 🏠 首页 → `index.md`
2. 🔒 ML Security → `ml-security/`（6 个话题 + 索引）
3. 🧠 Generative Models → `gen-models/`（3 个话题 + 索引）
4. 📐 AI 数学基础 → `ai-math/`（5 个话题 + 索引）
5. 🛠️ 技术实践 → `tech-practice/`（4 个子板块）
6. 📝 论文阅读 → `papers/`
7. 📸 生活 → `life/`

---

## 关键配置细节

### 评论区
- **方案**: Giscus（GitHub OAuth 登录）
- **配置位置**: `overrides/main.html`
- **仓库**: `MegumiKaor1/Blog`，category: `Announcements`

### Mermaid 图表
- 需要 `mermaid2` 插件（在 `mkdocs.yml` 的 `plugins:` 中）
- 需要 CDN 加载 `mermaid.min.js`（在 `extra_javascript:` 中）
- 代码块用 ` ```mermaid ` 标记

### KaTeX 数学
- 通过 CDN 加载（`extra_css` + `extra_javascript`）
- 行内公式: `$...$`，块级公式: `$$...$$`
- 自定义渲染脚本: `docs/assets/katex-render.js`

---

## 内容风格约定

- 用户是中文母语者，内容以中文为主
- 技术页面会加入「💭 个人理解」章节，不是复述论文，而是自己的洞察
- 要简洁、有主线，不要过度堆砌概念
- 涉及论文时，按：核心贡献 → 与研究方向关系 → 方法论结构 → 局限 → 启发

---

## 红线

- 不要修改 `site/` 目录下的文件（构建产物，由 `mkdocs build` 生成）
- 新增页面后，记得在 `mkdocs.yml` 的 `nav:` 中添加导航条目
- 部署前先在本地 `python3 -m mkdocs serve` 检查渲染效果
- `site_url` 不要改：`https://megumikaor1.github.io/Blog`（注意大写 B）

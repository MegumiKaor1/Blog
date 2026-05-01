# 🌲 Kaor1's World

> **ML Security · Generative Models · 动漫 · 摄影 · 旅游**

个人知识库 & 博客，基于 [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 构建，部署于 GitHub Pages。

---

## 🧭 内容结构

| 版块 | 内容 |
|------|------|
| 🔒 **ML Security** | Machine Unlearning, Backdoor Attacks, Data Poisoning, Model Editing, Adversarial Attacks, Robust Evaluation |
| 🧠 **Generative Models** | Diffusion Models, LLMs, Autoregressive Models |
| 📐 **AI 数学基础** | 线性代数、微积分、概率论、最优化、信息论 |
| 🛠️ **技术实践** | 环境配置 (Python/CUDA/PyTorch)、AI 工具链 (Agent/MCP/Skill)、AI Infra |
| 📝 **论文阅读** | 论文阅读笔记归档 |
| 📝 **随笔** | 动漫感想、摄影记录、旅游见闻、碎碎念 |

---

## 🚀 本地运行

```bash
# 安装依赖
pip install mkdocs-material mkdocs-mermaid2-plugin

# 启动本地预览
python3 -m mkdocs serve --dev-addr=0.0.0.0:8000
# → 打开 http://localhost:8000/blog/
```

> **注意**：Python 版本 3.9.6，`mkdocs` 不在 PATH 中，需要用 `python3 -m` 调用。访问路径带 `/blog/` 前缀是因为 `site_url` 设置为 `https://megumikaor1.github.io/Blog`。

---

## 📦 部署

```bash
mkdocs gh-deploy --force
```

推送到 `gh-pages` 分支后，GitHub Pages 自动更新。

---

## 📂 目录结构

```
Blog/
├── mkdocs.yml                 # 全站配置
├── README.md
├── overrides/
│   └── main.html              # 评论区 (Giscus) 模板覆盖
├── docs/
│   ├── index.md               # 首页
│   ├── ml-security/           # ML 安全笔记
│   ├── gen-models/            # 生成式模型笔记
│   ├── ai-math/               # AI 数学基础
│   ├── tech-practice/         # 技术实践
│   │   ├── setup/             #   环境基础
│   │   ├── ai-tools/          #   AI 工具链
│   │   ├── infra/             #   AI Infra
│   │   └── docs-share/        #   优质文档分享
│   ├── papers/                # 论文阅读
│   ├── life/                  # 随笔
│   └── assets/                # JS/CSS 资源
└── site/                      # 构建输出 (gitignored)
```

---

## 🛠 技术栈

- **静态站点**: [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 9.7.6
- **数学渲染**: KaTeX
- **图表**: Mermaid（通过 `mermaid2` 插件 + CDN 加载 `mermaid.min.js`）
- **评论区**: Giscus（GitHub OAuth 登录）
- **托管**: GitHub Pages

---

## 📝 许可

内容为个人学习笔记，转载请注明出处。

# 🌲 Kaor1's World

> **ML Security · Generative Models · 动漫 · 摄影 · 旅游**

个人知识库 & 博客，基于 [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 构建，部署于 GitHub Pages。

---

## 🧭 内容结构

| 版块 | 内容 |
|------|------|
| 🔒 **ML Security** | Machine Unlearning, Backdoor Attacks, Data Poisoning, Model Editing, Adversarial Attacks, Robust Evaluation |
| 🧠 **Generative Models** | Diffusion Models (基于 Luo 2022 的统一视角笔记), LLMs, Autoregressive Models |
| 📝 **论文阅读** | 论文阅读笔记归档 |
| 📸 **生活** | 动漫感想、摄影记录、旅游见闻、碎碎念 |

---

## 🚀 本地运行

```bash
# 安装依赖
pip install mkdocs-material mkdocs-mermaid2-plugin

# 启动本地预览
mkdocs serve
# → 打开 http://localhost:8000/
```

---

## 📦 部署

```bash
mkdocs gh-deploy
```

推送到 `gh-pages` 分支后，GitHub Pages 自动更新。

---

## 📂 目录结构

```
Blog/
├── mkdocs.yml              # 全站配置
├── docs/
│   ├── index.md            # 首页
│   ├── ml-security/        # ML 安全笔记
│   ├── gen-models/         # 生成式模型笔记
│   ├── papers/             # 论文阅读
│   ├── life/               # 生活博客
│   └── assets/             # JS/CSS 资源
└── .github/workflows/      # CI/CD（如有）
```

---

## 🛠 技术栈

- **静态站点**: [MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- **数学渲染**: KaTeX
- **图表**: Mermaid
- **托管**: GitHub Pages

---

## 📝 许可

内容为个人学习笔记，转载请注明出处。

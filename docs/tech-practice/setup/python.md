# Python 环境配置

AI 开发的入口——配好 Python 环境，后面全是坦途。

---

## 为什么 AI 开发要会配环境

- 不同项目依赖不同的 Python 版本和包版本，全局装一起必炸
- GPU 框架（PyTorch/CUDA）对 Python 版本敏感
- 虚拟环境让你每个项目独立，互不污染

---

## 前置概念

| 概念 | 一句话 |
|------|--------|
| **Python 版本** | 目前 AI 开发推荐 Python 3.10~3.12 |
| **虚拟环境** | 项目级别的隔离 Python 运行空间 |
| **venv** | Python 自带，轻量虚拟环境（本篇） |
| **Anaconda/Conda** | 第三方发行版，自带虚拟环境和科学计算包 → [Anaconda 配置](anaconda.md) |
| **包管理器** | pip / conda / uv ——安装和管理第三方库 |

> **选哪个？** 新手或数据科学入门 → Anaconda。轻量开发、已有 Python → venv。不冲突，可以共存。详见 [Anaconda 配置](anaconda.md)。

---

## macOS

### 1. 安装 Python

```bash
# 推荐用 Homebrew
brew install python@3.11

# 验证
python3.11 --version
```

### 2. 创建虚拟环境

```bash
# 到项目目录
cd my-project

# 创建 venv
python3.11 -m venv .venv

# 激活
source .venv/bin/activate

# 此时终端前面出现 (.venv) 标识
```

### 3. 安装依赖

```bash
pip install numpy torch torchvision
```

### 4. 导出 / 还原环境

```bash
# 导出
pip freeze > requirements.txt

# 还原（别人拿到你的项目后）
pip install -r requirements.txt
```

### Mac 特殊注意事项

- Apple Silicon (M1/M2/M3) 使用 **MPS 后端**跑 PyTorch，无需 CUDA
- Homebrew 是 Mac 上最方便的包管理器，先装它：`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

---

## Windows

### 1. 安装 Python

推荐方式：**Microsoft Store 直接装**

1. 打开 Microsoft Store
2. 搜 `Python 3.11` → 安装
3. 或者去 https://www.python.org/downloads/ 下载安装包

> ⚠️ 安装时**勾选 "Add Python to PATH"**！不然后面敲 `python` 命令都找不到。

### 2. 创建虚拟环境

```powershell
# PowerShell 或 CMD
cd my-project
python -m venv .venv

# 激活（注意 Windows 路径不同）
.venv\Scripts\activate
```

### 3. 日常操作同 macOS

```powershell
pip install numpy torch
pip freeze > requirements.txt
```

### Windows 特殊注意事项

- 路径用反斜杠 `\`，建议用 PowerShell 而不是 CMD
- 建议装 **Windows Terminal**（Microsoft Store 免费），比自带的 CMD 好用太多
- CUDA 安装需要额外步骤（见 CUDA 篇）

---

## Linux (Ubuntu/Debian)

### 1. 安装 Python

```bash
# 更新包列表
sudo apt update

# 安装 Python 3.11 和 venv 支持
sudo apt install python3.11 python3.11-venv python3-pip

# 验证
python3.11 --version
```

### 2. 创建虚拟环境

```bash
cd my-project
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. 日常操作

```bash
pip install numpy torch
pip freeze > requirements.txt
```

### Linux 特殊注意事项

- 服务器版 Linux 通常没有 GUI，全命令行操作
- 建议用 `screen` 或 `tmux` 保持会话，防止 SSH 断开后训练中断
- GPU 服务器需要额外装 NVIDIA 驱动和 CUDA（见 CUDA 篇）

---

## 进阶：用 `uv` 提速（全平台通用）

`uv` 是 Rust 写的 Python 包管理器，比 pip 快 10-100 倍：

```bash
# 安装 uv（三平台通用）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv

# 激活（和之前一样）
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# 装包
uv pip install torch numpy

# 一键创建项目
uv init my-project
```

---

## 检查清单

| 项目 | 命令 |
|------|------|
| Python 装好了吗 | `python --version` |
| venv 激活了吗 | 看终端前缀有没有 `(.venv)` |
| 包装对了吗 | `pip list` |

---

## 下一步

- 配好 Python 后 → [CUDA 与 GPU 环境](cuda.md)
- 装 CUDA 后 → [PyTorch 安装与入门](pytorch.md)

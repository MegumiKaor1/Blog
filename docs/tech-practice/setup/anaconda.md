# Anaconda / Miniconda 环境配置

Anaconda 是 Python 发行版 + 虚拟环境管理器 + 预装科学计算包的合体方案，AI 入门的经典选择。

---

## Anaconda vs Miniconda

| | Anaconda | Miniconda |
|------|----------|-----------|
| 大小 | ~3 GB | ~50 MB |
| 预装包 | numpy, scipy, pandas, jupyter 等 250+ 包 | 只装 conda + Python |
| 适合 | 新手、数据科学入门 | 有经验的开发者 |
| **推荐** | 🔰 新手 | ✅ 开发者 |

> Miniconda 更轻量，需要什么自己装。推荐用它。

---

## macOS

### 安装 Miniconda

```bash
# 下载安装脚本
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# 运行（Apple Silicon）
bash Miniconda3-latest-MacOSX-arm64.sh

# Intel Mac 用这个：
# bash Miniconda3-latest-MacOSX-x86_64.sh
```

按提示回车，`yes` 同意协议，最后选 `yes` 让 conda 自动初始化 shell。

装完后**关掉终端重开**，或执行：

```bash
source ~/.zshrc
```

验证：

```bash
conda --version
# conda 24.x.x
```

### 日常操作

```bash
# 创建环境（指定 Python 版本）
conda create -n myproject python=3.11

# 激活
conda activate myproject

# 装包（优先 conda，不行再 pip）
conda install numpy pandas matplotlib
pip install torch torchvision

# 导出环境
conda env export > environment.yml

# 还原环境
conda env create -f environment.yml

# 列出所有环境
conda env list

# 退出环境
conda deactivate

# 删除环境
conda env remove -n myproject
```

---

## Windows

### 安装 Miniconda

1. 打开 https://docs.anaconda.com/miniconda/
2. 下载 **Windows 64-bit** 安装包
3. 双击运行 → 一路 Next
4. 安装选项选 **Just Me**（不要选 All Users）
5. ⚠️ **不要勾选 "Add to PATH"**（安装程序会警告说最好不勾——听它的）
6. 装完后从**开始菜单**打开 **Anaconda Prompt (Miniconda3)**

> 始终用 Anaconda Prompt 操作，不要用普通 CMD 或 PowerShell——它配置好了 conda 的环境变量。

### 日常操作（和 macOS 完全相同）

```bash
conda create -n myproject python=3.11
conda activate myproject
conda install numpy pandas
pip install torch
```

---

## Linux

### 安装 Miniconda

```bash
# 下载
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 初始化（安装脚本最后会问，选 yes）
# 手动初始化：
~/miniconda3/bin/conda init bash

# 重载配置
source ~/.bashrc
```

### 服务器无 GUI 安装

```bash
# 静默安装，跳过所有交互
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 日常操作

```bash
conda create -n myproject python=3.11
conda activate myproject
conda install numpy pandas
pip install torch
```

---

## Conda vs venv 怎么选

| 场景 | 推荐 |
|------|------|
| 刚学 Python，装东西总报错 | **Anaconda**——自带常用包，开箱即用 |
| 已经装了 Python，只想隔离项目 | **venv**——轻量够用 |
| 数据科学 / 科学计算 | **Anaconda**——conda 处理二进制依赖更稳 |
| 服务器部署 / Docker | **venv**——更小更快 |
| PyTorch / CUDA 项目 | **都可以**——conda 能装 cudatoolkit，但 pip 装 torch 也自带 CUDA |
| 两者混用 | **可以共存**——但不要在同一个项目里 conda 和 venv 混用 |

---

## Conda 常用命令速查

| 操作 | 命令 |
|------|------|
| 创建环境 | `conda create -n 名字 python=3.11` |
| 激活 | `conda activate 名字` |
| 退出 | `conda deactivate` |
| 装包 | `conda install 包名` |
| pip 装包 | `pip install 包名`（在激活的 conda 环境里） |
| 列出环境 | `conda env list` |
| 导出 | `conda env export > environment.yml` |
| 删除环境 | `conda env remove -n 名字` |
| 更新 conda | `conda update conda` |

---

## 常见坑

| 问题 | 解决 |
|------|------|
| `conda` 命令找不到 | 没初始化 shell：跑 `conda init` 然后重开终端 |
| `conda activate` 报错 | Windows 用 Anaconda Prompt，别用普通 CMD |
| 装包很慢 | 换清华源：`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/` |
| conda 和 pip 混着装冲突 | 尽量用 conda 装，pip 只作为补充。先 conda 后 pip |
| 环境太多占空间 | `conda clean --all` 清理缓存 |

---

## 下一步

- 环境配好了 → [PyTorch 安装与入门](pytorch.md)
- 更喜欢轻量方案 → [Python 环境配置 (venv)](python.md)

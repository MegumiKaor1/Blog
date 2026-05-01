# CUDA 与 GPU 环境配置

深度学习不跑 GPU 就跟开跑车挂一档一样。这篇从零把你的显卡配到能跑 PyTorch。

---

## 前置知识

| 概念 | 是什么 |
|------|--------|
| **NVIDIA 驱动** | 操作系统和显卡之间的桥梁，识别你的 GPU |
| **CUDA Toolkit** | NVIDIA 的并行计算平台，PyTorch 靠它调用 GPU |
| **cuDNN** | NVIDIA 的深度神经网络加速库，训练/推理更快 |
| **PyTorch CUDA 版本** | PyTorch 内置了精简版 CUDA，不必单独装 |

> **重要**：现代 PyTorch（≥ 2.0）自带 CUDA 运行时，**不需要**单独安装 CUDA Toolkit。 只装显卡驱动就行。

---

## macOS

Apple Silicon 用户看这里。

### Apple Silicon (M1/M2/M3)

**不需要 CUDA，不需要 NVIDIA 驱动。** PyTorch 直接用 MPS 后端：

```python
import torch
print(torch.backends.mps.is_available())  # 应输出 True

# 把 tensor 放到 "GPU" 上
device = torch.device("mps")
x = torch.randn(3, 3).to(device)
```

安装 PyTorch 时选 MPS 版本（见 [PyTorch 篇](pytorch.md)）。

### Intel Mac（有独显的旧款）

少数 Intel Mac 有 AMD 独显，**不支持 CUDA**，只能用 CPU 跑或外接 eGPU（极其不推荐）。

---

## Windows

Windows 配 GPU 环境最折腾，按顺序来。

### 第 1 步：确认你有 NVIDIA 显卡

```powershell
# PowerShell
nvidia-smi
```

如果显示显卡型号和驱动版本 → ✅ 驱动已装。
如果报错 → 继续下一步。

### 第 2 步：装 NVIDIA 驱动

1. 打开 https://www.nvidia.com/download/index.aspx
2. 选你的显卡型号 → 下载 Game Ready 驱动 → 安装
3. 重启 → 再跑 `nvidia-smi` → 应该看到类似：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 555.xx       Driver Version: 555.xx       CUDA Version: 12.5     |
+-----------------------------------------------------------------------------+
```

### 第 3 步：装 PyTorch（自带 CUDA）

**不要**单独装 CUDA Toolkit。直接在 PyTorch 官网选对应版本：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 第 4 步：验证

```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # 你的显卡型号
```

### Windows 常见坑

| 问题 | 解决 |
|------|------|
| `nvidia-smi` 不识别 | 驱动没装好，重装 |
| PyTorch 说 CUDA 不可用 | PyTorch 版本装错了，确认你装的 `cu124` 而不是 `cpu` |
| 显存不足 | 减小 batch_size 或用 `torch.cuda.empty_cache()` |

---

## Linux (Ubuntu/Debian)

### 第 1 步：装 NVIDIA 驱动

```bash
# 查看推荐驱动版本
ubuntu-drivers devices

# 自动安装推荐版本
sudo ubuntu-drivers autoinstall

# 重启
sudo reboot

# 验证
nvidia-smi
```

> 如果 `ubuntu-drivers` 不可用：`sudo apt install ubuntu-drivers-common`

### 第 2 步：装 PyTorch

```bash
pip install torch torchvision torchaudio
```

PyTorch 自动带 CUDA，不需要额外步骤。

### 第 3 步：验证

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())  # 多卡服务器会显示 >1
```

### 服务器常用操作

```bash
# 查看 GPU 状态
nvidia-smi

# 实时监控（每秒刷新）
watch -n 1 nvidia-smi

# 指定用哪张卡
CUDA_VISIBLE_DEVICES=0,1 python train.py  # 只用 0 号和 1 号卡
```

---

## 三平台对比

| 平台 | GPU 后端 | 需要单独装的东西 |
|------|----------|------------------|
| macOS (Apple Silicon) | MPS | 无，PyTorch 直接用 |
| macOS (Intel) | 无 | 无，只能用 CPU |
| Windows (NVIDIA) | CUDA | NVIDIA 驱动 |
| Linux (NVIDIA) | CUDA | NVIDIA 驱动 |

---

## 验证一切都配好了

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")

# macOS MPS
if torch.backends.mps.is_available():
    print("✅ MPS 可用（Apple GPU）")
    x = torch.randn(10, 10).to("mps")

# CUDA
elif torch.cuda.is_available():
    print(f"✅ CUDA 可用 ({torch.cuda.get_device_name(0)})")
    x = torch.randn(10, 10).cuda()

else:
    print("⚠️ 只用 CPU，速度会比较慢")
    x = torch.randn(10, 10)

print(x)
```

---

## 下一步

- GPU 配好了 → 装 PyTorch 跑第一个模型：[PyTorch 安装与入门](pytorch.md)

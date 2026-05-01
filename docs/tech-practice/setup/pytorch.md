# PyTorch 安装与入门

从安装到跑出第一个训练结果。

---

## 安装

根据你的平台选一条命令：

### macOS (Apple Silicon)

```bash
pip install torch torchvision torchaudio
```

PyTorch 在 Apple Silicon 上直接用 MPS 加速，不需要额外配置。

### Windows / Linux (NVIDIA GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> `cu124` 表示 CUDA 12.4 版本。去 https://pytorch.org/get-started/locally/ 选择你的配置获取最新命令。

### CPU Only（无 GPU 或 macOS Intel）

```bash
pip install torch torchvision torchaudio
```

---

## 验证安装

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"Python 版本: {torch.__version__}")

# 检查 GPU
if torch.cuda.is_available():
    print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("✅ MPS (Apple GPU)")
else:
    print("⚠️ CPU only")
```

---

## 核心概念速览

### Tensor（张量）

Tensor 就是多维数组。PyTorch 里一切都是 Tensor。

```python
# 创建
x = torch.tensor([[1, 2], [3, 4]])  # 2x2 矩阵
z = torch.zeros(3, 3)                # 全零
r = torch.randn(2, 5)                # 随机数

# 属性
print(x.shape)   # torch.Size([2, 2])
print(x.dtype)   # torch.int64
print(x.device)  # cpu

# 移到 GPU
if torch.cuda.is_available():
    x = x.cuda()
```

### 自动求导（Autograd）

神经网络的核心——自动算梯度，不需要手推。

```python
# requires_grad=True 开启梯度追踪
w = torch.randn(3, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# 前向计算
x = torch.randn(3)
y = (x @ w + b).sum()

# 反向传播（自动算 dw, db）
y.backward()

print(w.grad)  # ∂y/∂w
print(b.grad)  # ∂y/∂b
```

> `@` 是矩阵乘法，等价于 `torch.matmul()`。

---

## 三件套简介

| 库 | 用途 |
|----|------|
| `torch` | 核心：Tensor + Autograd + 神经网络层 |
| `torchvision` | 图像数据集、预处理、预训练模型 |
| `torchaudio` | 音频处理 |

---

## 第一个完整例子：线性回归

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ===== 1. 造数据 =====
# 真实关系: y = 3x + 2
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.3  # 加点噪声

# ===== 2. 定义模型 =====
model = nn.Linear(1, 1)  # 输入1维 → 输出1维（就是 w*x + b）

# ===== 3. 损失函数 & 优化器 =====
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===== 4. 训练 =====
for epoch in range(200):
    # 前向
    pred = model(X)
    loss = criterion(pred, y)

    # 反向
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# ===== 5. 结果 =====
w, b = model.weight.item(), model.bias.item()
print(f"真实: y = 3x + 2")
print(f"学出: y = {w:.2f}x + {b:.2f}")
```

输出大致：
```
Epoch 0: loss = 12.3456
Epoch 50: loss = 0.2345
Epoch 100: loss = 0.0987
Epoch 150: loss = 0.0912
真实: y = 3x + 2
学出: y = 2.98x + 2.04
```

---

## 训练流程模板（记住这个就够了）

```
1. 模型定义     → nn.Module
2. 数据准备     → Dataset + DataLoader
3. 损失函数     → 衡量预测和真实的差距
4. 优化器       → 更新参数
5. 训练循环     → for epoch: 前向 → 算 loss → 反向 → 更新
```

```python
# 模板代码
model = MyModel()
criterion = nn.MSELoss()          # 回归用 MSE
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

---

## 常见问题

| 问题 | 诊断 | 解决 |
|------|------|------|
| CUDA out of memory | 显存爆了 | 减小 batch_size，加 `torch.cuda.empty_cache()` |
| Tensor 在 CPU 不在 GPU | 忘了 `.to(device)` | 统一用 `device` 变量 |
| loss 不下降 | 学习率不对 / 数据有问题 | 先在小数据集上过拟合测试 |

---

## 下一步

- 想深入 → 官方教程 https://pytorch.org/tutorials/
- 配 GPU 环境 → [CUDA 与 GPU 环境](cuda.md)
- 开始做项目 → 去 [Hermes Agent 配置指南](../ai-tools/hermes-agent.md) 把开发环境串起来

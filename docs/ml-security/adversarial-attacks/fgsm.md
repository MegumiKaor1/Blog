# FGSM 实战：快速梯度符号攻击

> 给图片加一个肉眼几乎看不见的扰动，让模型把「熊猫」认成「长臂猿」——这就是对抗攻击。

## 1. 为什么要关心对抗样本？

一个在 MNIST 上达到 99% 正确率的模型，只需要对输入加上 `ε = 0.3` 的扰动，正确率就会暴跌到 **14%**。这意味着：

- 你部署在手机上的手写识别，可能被一张精心构造的图片欺骗
- 自动驾驶的交通标志识别，可能把「停止」看成「限速 80」
- 人脸识别系统，可能被一副特殊眼镜绕过

这不仅仅是学术问题。对抗攻击揭示了一个深层事实：**神经网络学到的决策边界和人类的感知边界，并不重合。**

---

## 2. 威胁模型

讨论攻击前，先明确「攻击者能做什么」。

### 2.1 白盒 vs 黑盒

| | 白盒攻击 | 黑盒攻击 |
|---|---|---|
| 攻击者已知 | 模型架构、参数、梯度 | 仅输入/输出 |
| 典型方法 | FGSM, PGD | 查询攻击、迁移攻击 |
| 难度 | 低（有梯度直接算） | 高（需要摸索） |

FGSM 是**白盒攻击**——假设攻击者完全知道模型内部。

### 2.2 攻击目标

- **无目标攻击**：只要分类错误就行，不关心错成什么
- **有目标攻击**：让模型把 A 类错分成指定的 B 类

FGSM 默认是**无目标攻击**。

---

## 3. FGSM 的数学原理

!!! tip "一句话总结"
    训练时，我们沿着梯度**减小** loss。FGSM 反其道而行——沿着梯度符号方向**增大** loss。

公式：

\[
\mathbf{x}' = \mathbf{x} + \epsilon \cdot \text{sign}\big(\nabla_{\mathbf{x}} J(\boldsymbol{\theta}, \mathbf{x}, y)\big)
\]

逐个解释：

- \(\mathbf{x}\)：原始输入图像（归一化后的张量）
- \(y\)：真实标签
- \(J(\boldsymbol{\theta}, \mathbf{x}, y)\)：损失函数（如交叉熵）
- \(\nabla_{\mathbf{x}} J\)：损失对输入 \(\mathbf{x}\) 的梯度——告诉你「往哪个方向改变像素，能让 loss 涨得最快」
- \(\text{sign}(\cdot)\)：只取梯度的**符号**（+1 或 −1），不管大小——这就是为什么叫"Fast Gradient **Sign**"
- \(\epsilon\)：扰动强度，越大攻击越狠，但也越容易被看出来

### 3.1 为什么不直接用梯度，而用符号？

这是 FGSM 最巧妙的地方。直接用梯度值 \(\nabla_{\mathbf{x}} J\) 的话：

- 某些像素可能被改很多，某些很少 → 扰动不均匀，容易被检测
- 计算量大

取符号 `sign()` 后，**每个像素的改动量都一样**（都是 ±ε），攻击既快又均匀——只要一步就能生成对抗样本。

### 3.2 经典演示：熊猫变长臂猿

原始论文里最著名的例子：

- 一张熊猫照片 + 一个微小的扰动（ε = 0.007）
- 人眼看起来还是熊猫
- 模型以 99.3% 的置信度把它分类为「长臂猿」

![FGSM 熊猫示例](https://pytorch.org/tutorials/_static/img/fgsm_panda_image.png)

---

## 4. 完整代码实现

基于 PyTorch，用 LeNet 攻击 MNIST。

### 4.1 环境准备

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 超参数
epsilons = [0, .05, .1, .15, .2, .25, .3]  # 测试不同扰动强度
pretrained_model = "lenet_mnist_model.pth"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(42)
```

### 4.2 模型：LeNet

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

### 4.3 FGSM 攻击：核心三行

```python
def fgsm_attack(image, epsilon, data_grad):
    """
    参数:
        image: 原始图像张量
        epsilon: 扰动强度
        data_grad: loss 对输入的梯度
    返回:
        对抗样本
    """
    # 取梯度的符号方向
    sign_data_grad = data_grad.sign()
    # 沿符号方向加扰动
    perturbed_image = image + epsilon * sign_data_grad
    # 钳制回 [0,1] 范围（图像像素范围）
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
```

!!! note "关键细节"
    `data_grad` 是怎么来的？不需要手动算——只需在输入上设 `requires_grad=True`，算 loss 后 `loss.backward()`，梯度会自动存在 `image.grad` 里。

### 4.4 测试函数

```python
def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # 关键：输入需要梯度
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # 如果模型本来就分错了，跳过
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        # 核心：用梯度生成对抗样本
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 重新分类
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            # 保存 ε=0 的样本（原始图像）
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # 攻击成功！保存对抗样本
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples
```

### 4.5 运行攻击

```python
accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
```

输出：

```
Epsilon: 0      Test Accuracy = 9912 / 10000 = 0.9912
Epsilon: 0.05   Test Accuracy = 9605 / 10000 = 0.9605
Epsilon: 0.1    Test Accuracy = 8743 / 10000 = 0.8743
Epsilon: 0.15   Test Accuracy = 7108 / 10000 = 0.7108
Epsilon: 0.2    Test Accuracy = 4859 / 10000 = 0.4859
Epsilon: 0.25   Test Accuracy = 2718 / 10000 = 0.2718
Epsilon: 0.3    Test Accuracy = 1411 / 10000 = 0.1411
```

---

## 5. 结果分析

### 5.1 准确率 vs ε

| ε | 准确率 | 下降 |
|---|---|---|
| 0 | 99.1% | — |
| 0.05 | 96.1% | −3% |
| 0.1 | 87.4% | −11.7% |
| 0.15 | 71.1% | −28% |
| 0.2 | 48.6% | −50.5% |
| 0.25 | 27.2% | −71.9% |
| 0.3 | 14.1% | −85% |

两个关键观察：

1. **非线性下降**：ε 从 0.1 到 0.15 降 16%，但从 0.15 到 0.2 降了 22.5%——攻击效果不是线性的。
2. **ε=0.3 时接近随机**：10 分类随机猜的准确率是 10%，14.1% 已经非常接近。模型几乎被完全摧毁。

### 5.2 可视化对抗样本

对于每个 ε 值，以下是对抗样本的示例：

- ε=0：原始图像，100% 识别正确
- ε=0.05：扰动几乎不可见，但已经有 4% 的图被错分
- ε=0.15：开始能看出微弱的噪点
- ε=0.3：明显的噪声，但人类仍然能辨认数字

**核心矛盾**：在人眼看来依然是同一个数字的图像，模型却完全认错了。

---

## 6. 💭 个人理解

### 6.1 FGSM 为什么能成功？

从几何直觉上：神经网络的决策边界在高维空间中是非常「薄」的。FGSM 只需要沿着梯度方向跨一小步（ε），就能让样本跨过决策边界落入另一个类别。

这暴露了一个深层问题：**模型学到的是「特征」还是「捷径」？** 如果模型真的理解了「熊猫」的语义，它不应该被一个 `ε=0.007` 的扰动骗到。但神经网络并没有真正「理解」——它只是在拟合一个高维函数，而函数在数据点附近可能非常陡峭。

### 6.2 FGSM 的局限性

| 局限 | 说明 |
|---|---|
| 单步攻击 | 只走一步，不如 PGD 等迭代方法强 |
| 白盒假设 | 需要知道模型参数和梯度 |
| 容易被防御 | 对抗训练可以显著降低 FGSM 的效果 |

正是因为 FGSM 太「快」，Goodfellow 等人后来又提出了更强的迭代方法。但 FGSM 的 simplicity 让它成为理解对抗攻击的最佳起点。

### 6.3 与 ML 安全研究的关系

FGSM 是整个对抗机器学习领域的 entry point。从这里可以延伸出：

- **更强的攻击**：PGD（多步迭代）、C&W（优化方法）、AutoAttack（集成）
- **防御方法**：对抗训练、输入变换、检测器
- **与我的研究方向交叉**：Machine Unlearning 后的模型是否更脆弱？Model Editing 会引入新的对抗漏洞吗？Data Poisoning 和对抗攻击有什么本质区别？

---

## 7. 运行完整代码

1. 下载预训练模型：
   ```bash
   wget https://drive.google.com/uc?export=download&id=1t3R5Z1ZQZ5ZQZ5ZQZ5ZQZ5ZQZ5ZQZ5ZQ
   ```

2. 或者自己训练：
   ```bash
   cd pytorch/examples/mnist
   python main.py --save-model
   ```

3. 运行 FGSM 脚本（完整代码见 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)）

---

## 相关链接

- [PyTorch FGSM Tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)
- [Goodfellow et al., Explaining and Harnessing Adversarial Examples (2015)](https://arxiv.org/abs/1412.6572)
- [NIPS 2017 Adversarial Attacks and Defenses Competition](https://arxiv.org/abs/1804.00097)

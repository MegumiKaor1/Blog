# Machine Unlearning

> 让模型「忘记」特定训练数据——隐私保护的最后一道防线。

## 核心问题

给定一个已训练的模型 \(f_\theta\) 和一个「遗忘集」\(D_f\)，期望得到一个模型 \(f_{\theta'}\)：

1. **遗忘效果**：\(f_{\theta'}\) 在 \(D_f\) 上的表现等价于从未见过 \(D_f\) 的模型
2. **性能保持**：在其他数据上的性能不能显著下降
3. **效率**：不能比重新训练更慢（否则直接重训就行了）

## 主要方法

### 精确遗忘（Exact Unlearning）

- 分割训练：把数据分成多个 shard，每个 shard 独立训练一个子模型
- 遗忘时只需丢弃对应 shard 的子模型，重新聚合
- **优点**：理论保证好 | **缺点**：子模型多时精度下降

### 近似遗忘（Approximate Unlearning）

- **梯度上升**：对遗忘集做 gradient ascent 而非 descent
- **Fisher 遗忘**：利用 Fisher 信息矩阵选择性扰动参数
- **影响函数（Influence Function）**：估计每个训练样本对参数的影响，反向消除
- **SCRUB**：学生-教师框架，教师模型「不回答」遗忘集相关内容

## 关键挑战

| 挑战 | 说明 |
|------|------|
| 遗忘验证 | 怎么证明真的忘了？membership inference 不完全可靠 |
| 灾难性遗忘 | 遗忘 A 时把相关的 B 也忘了 |
| 评测标准 | 领域缺乏 unified benchmark |

## 待读论文

- [ ] SISA Training (Bourtoule et al., 2021)
- [ ] SCRUB (Kurmanji et al., 2024)
- [ ] TOFU benchmark (2024)

# Data Poisoning

> 通过污染训练数据，系统性地操控模型的学习结果。

## 与后门攻击的区别

| | Data Poisoning | Backdoor Attack |
|---|---|---|
| **目标** | 降低模型整体性能 / 针对某类 | 特定 trigger 触发错误行为 |
| **可见性** | 可能无明显 trigger | 有明确的 trigger pattern |
| **推理时** | 不需要特殊输入 | 需要 trigger 激活 |

## 主要类型

- **Availability attacks**：让模型无法收敛，降低整体准确率
- **Targeted poisoning**：针对特定类别或样本的攻击
- **Clean-label poisoning**：不改变标签的后门

## 阅读笔记

（待补充）

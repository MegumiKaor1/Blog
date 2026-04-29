# Model Editing

> 精准修改模型中的一个知识点，而不影响其他知识。

## 与 Machine Unlearning 的关系

| | Model Editing | Machine Unlearning |
|---|---|---|
| **操作** | 改一个知识点 | 删一个知识点 |
| **目标** | 新知识正确 + 旧知识保持 | 目标知识遗忘 + 其他保持 |
| **共同挑战** | 灾难性遗忘、精准定位参数 | 同左 |

## 主要方法

- **ROME**（Rank-One Model Editing）：用因果追踪定位关键 MLP 层，秩一更新修改
- **MEND**：训练一个超网络来生成参数更新
- **MEMIT**：批量编辑，一次改多个知识点

## 待读论文

- [ ] ROME (Meng et al., 2022)
- [ ] MEMIT (Meng et al., 2023)

# Backdoor Attacks

> 训练时植入后门：输入带特定 trigger 时模型输出攻击者指定的错误结果。

## 威胁模型

1. **数据投毒**：攻击者向训练数据注入带 trigger 的样本，标签改为 target label
2. **推理时触发**：模型正常输入表现正常；带 trigger 的输入输出攻击者指定的结果

## 经典攻击

| 攻击 | 特点 |
|------|------|
| **BadNets** | 简单像素 patch 作为 trigger |
| **Blended** | 将 trigger 以低透明度混合到图像中 |
| **Invisible** | 人眼不可见的 trigger |
| **Clean-label** | 不改变标签的后门攻击 |

## 后门与 Unlearning 的关系

- 后门本质上是模型「记住」了一段恶意关联
- Machine unlearning 可以作为一种**后门移除**手段
- 关键问题：unlearning 能否精准移除后门而不损害正常能力？

## 待读论文

- [ ] BadNets (Gu et al., 2017)
- [ ] Neural Cleanse (Wang et al., 2019)
- [ ] Fine-Pruning (Liu et al., 2018)

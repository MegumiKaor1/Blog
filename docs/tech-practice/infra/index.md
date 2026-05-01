# 🏗️ AI Infra

> AI Infrastructure：模型训练、推理、部署的底层工程体系。

AI Infra 是连接「算法研究」与「生产落地」的桥梁。做 ML 安全实验时你会频繁遇到：多卡训练怎么配、显存爆了怎么调、怎么高效跑 100 次鲁棒评测。

---

## 板块结构

| 章节 | 内容 |
|------|------|
| [分布式训练](distributed-training.md) | 多卡并行（DP/DDP/FSDP）、混合精度、DeepSpeed |
| [推理优化](inference-optimization.md) | vLLM、量化（GPTQ/AWQ）、KV Cache |
| [模型服务化](model-serving.md) | API 部署、负载均衡、监控 |
| [存储与数据管线](data-pipeline.md) | 数据加载优化、向量数据库 |

---

## 为什么学 AI Infra？

```
你现在的状态：
「这个攻击方法要跑 50 组对照实验，一张卡跑了两天还没出完」

学了 Infra 之后：
「配好 4 卡 FSDP，半小时跑完 50 组实验，结果已经画好图了」
```

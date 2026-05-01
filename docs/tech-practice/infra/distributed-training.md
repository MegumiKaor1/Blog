# 分布式训练

> 一张 GPU 不够用？把模型和数据拆到多张卡上并行训练。

---

## 为什么需要分布式训练

现代模型越来越大，单卡装不下了：

| 模型规模 | 参数 | 显存需求（FP16） | 需要几张 A100 (80G) |
|----------|------|-----------------|---------------------|
| LLaMA-7B | 7B | ~14 GB | 1 |
| LLaMA-13B | 13B | ~26 GB | 1 |
| LLaMA-70B | 70B | ~140 GB | 2+ |
| GPT-3 | 175B | ~350 GB | 5+ |
| 自己微调 7B | 7B | ~56 GB（含优化器） | 1 勉强 |

---

## 三种并行策略

### 1. 数据并行（Data Parallelism, DP/DDP）

> 每张卡上有完整模型副本，数据拆分成 N 份，各自算梯度，最后汇总。

```
GPU 0: 模型副本 + 数据 batch 0 → 梯度 0 ↘
GPU 1: 模型副本 + 数据 batch 1 → 梯度 1 → AllReduce → 平均梯度 → 更新所有副本
GPU 2: 模型副本 + 数据 batch 2 → 梯度 2 ↗
GPU 3: 模型副本 + 数据 batch 3 → 梯度 3 ↗
```

| 框架 | 特点 |
|------|------|
| **DP** (`nn.DataParallel`) | PyTorch 原生，单机多卡，效率一般 |
| **DDP** (`nn.DistributedDataParallel`) | 多进程，通信更高效，**推荐** |

```python
# DDP 最简示例
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

### 2. 模型并行（Tensor Parallelism）

> 模型太大单卡装不下？把模型切开，每张卡负责一部分层或一部分矩阵运算。

```
Transformer 层切分示例：
GPU 0: Attention Head 0-7
GPU 1: Attention Head 8-15   ← 同层内部切开
GPU 2: FFN 层
GPU 3: Output 层
```

**[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** 是 Tensor Parallelism 的代表实现。

### 3. 流水线并行（Pipeline Parallelism）

> 按层切开，GPU 0 处理前几层，GPU 1 处理中间层，GPU 2 处理后几层 —— 像工厂流水线。

```
Batch 1: GPU 0 (层1-8)  → GPU 1 (层9-16)  → GPU 2 (层17-24)
Batch 2:                 → GPU 0 (层1-8)   → GPU 1 (层9-16) → GPU 2 (层17-24)
```

**问题**：流水线有空泡（bubble），GPU 会闲着等上游。

### 混合：3D 并行

实际训练大模型是三种并行的组合：

```
数据并行 × 张量并行 × 流水线并行 = 3D 并行
```

---

## DeepSpeed：一站式分布式训练

> 微软出品，"PyTorch 的分布式训练外挂"。核心是 **ZeRO** 优化器。

### ZeRO 的三个阶段

| 阶段 | 分片内容 | 显存节省 |
|------|---------|---------|
| **ZeRO-1** | 优化器状态（Adam 的 m, v） | 4× |
| **ZeRO-2** | + 梯度 | 8× |
| **ZeRO-3** | + 模型参数 | N×（N = GPU 数量） |

**只需改 3 行代码**：

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config='ds_config.json'  # ZeRO stage 配置
)
# 训练循环几乎不变
```

### DeepSpeed 配置示例

```json
{
  "train_batch_size": 32,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"}
  },
  "fp16": {"enabled": true}
}
```

---

## 混合精度训练（Mixed Precision）

> FP32（32 位浮点）训练慢、占显存。用 FP16 加速，但精度敏感部分保留 FP32。

```
前向/反向传播：FP16（快，省显存）
权重更新：FP32（保证精度）
```

PyTorch 一行启用：

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
```

**NVIDIA APEX** 和 **torch.cuda.amp** 都支持。

---

## FSDP：PyTorch 原生替代

PyTorch 2.0 后推荐用 `FullyShardedDataParallel` 替代 DDP，理念类似 ZeRO-3：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

---

## 框架选型速查

| 场景 | 推荐 |
|------|------|
| 单机 2-4 卡，模型能装下 | DDP + 混合精度 |
| 单机 4-8 卡，模型较大 | DeepSpeed ZeRO-2 / FSDP |
| 大模型（70B+） | DeepSpeed ZeRO-3 + CPU Offload |
| 超大模型（175B+） | Megatron-LM 3D 并行 + DeepSpeed |

---

## 延伸阅读

- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

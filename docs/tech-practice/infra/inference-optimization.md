# 推理优化

> 模型训好了，怎么让它跑得更快、更省显存？

---

## 核心思路

推理优化的目标三角：

```
        速度快
         /\
        /  \
       /    \
      /      \
     /________\
  省显存    精度高

三者之间存在 trade-off，不同场景侧重不同。
```

---

## 量化（Quantization）

> 把模型参数从高精度（FP16/FP32）压到低精度（INT8/INT4），降低显存和计算量。

### 两个阶段

| 方法 | 原理 | 代表 |
|------|------|------|
| **PTQ**（训练后量化） | 不重新训练，直接压缩 | GPTQ、AWQ |
| **QAT**（训练时量化） | 训练时就模拟量化 | 一般大模型不用 |

### 常用量化方案

| 方案 | 精度 | 显存（7B 模型） | 速度 | 适合 |
|------|------|----------------|------|------|
| FP16 | 16-bit | ~14 GB | 基准 | 有 A100 时 |
| INT8 | 8-bit | ~7 GB | 1.5-2× | 大多数 GPU |
| INT4 (GPTQ) | 4-bit | ~4 GB | 2-3× | 消费级 GPU |
| INT4 (AWQ) | 4-bit | ~4 GB | 2-3× | 比 GPTQ 精度略好 |
| Q4_K_M (GGUF) | 混合 4-bit | ~4 GB | 2-3× | llama.cpp 生态 |

### GPTQ 示例

```python
from transformers import AutoModelForCausalLM, GPTQConfig

quant_config = GPTQConfig(bits=4, dataset="c4", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quant_config,
    device_map="auto"
)
```

---

## KV Cache 优化

> LLM 每生成一个 token 都要重新计算所有之前 token 的 Key/Value 矩阵 —— 这是巨大的计算浪费。KV Cache 把算过的 K、V 存起来复用。

```
无 KV Cache： 每个新 token 都要算全部历史 → O(n²)
有 KV Cache： 只算新 token，历史查缓存   → O(n)
```

### KV Cache 的显存问题

长序列下 KV Cache 本身就很占显存。于是有了：

| 技术 | 思路 |
|------|------|
| **Multi-Query Attention (MQA)** | 所有 head 共享一组 K、V |
| **Grouped-Query Attention (GQA)** | 分组共享 K、V（LLaMA 2/3 用的） |
| **PagedAttention (vLLM)** | 把 KV Cache 分页管理，类似 OS 虚拟内存 |
| **Sliding Window** | 只保留最近 N 个 token 的 KV |

---

## vLLM：高性能推理引擎

> UC Berkeley 出品，**PagedAttention** 算法大幅提升吞吐量。

```
传统推理：KV Cache 连续分配 → 碎片化 → 浪费显存 → 并发低
vLLM：    页式 KV Cache   → 近乎零浪费 → 显存利用率翻倍 → 并发大幅提升
```

### 一键部署

```bash
pip install vllm

# 启动 OpenAI 兼容 API
vllm serve meta-llama/Llama-2-7b
```

### Python 调用

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B")
outputs = llm.generate(["什么是机器学习？"], SamplingParams(temperature=0.7))
print(outputs[0].outputs[0].text)
```

---

## 其他推理加速技术

| 技术 | 说明 |
|------|------|
| **TensorRT-LLM** | NVIDIA 官方，极致优化（需编译） |
| **Flash Attention** | IO-aware 注意力算法，2-4× 加速 |
| **Speculative Decoding** | 小模型「猜」token，大模型验证，提速 2-3× |
| **Continuous Batching** | 动态组 batch，不等所有请求完成 |

---

## 选型建议

| 场景 | 推荐方案 |
|------|---------|
| 本地跑 7B 模型玩玩 | llama.cpp + GGUF (Q4_K_M) |
| 单 GPU 做推理服务 | vLLM + AWQ 量化 |
| 生产高并发 API | vLLM + 多 GPU 张量并行 |
| 极致性能（NVIDIA 卡） | TensorRT-LLM |
| CPU 推理 | llama.cpp（GGUF）+ 内存 |

---

## 延伸阅读

- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

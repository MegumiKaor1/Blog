# 存储与数据管线

> 数据喂得慢，GPU 就闲着。让数据加载跟上 GPU 的速度。

---

## 为什么数据管线重要

```
GPU 训练一 batch 时间分配：
  ████████ 前向+反向传播：80%
  ██ 数据加载：20% ← 如果优化不好，这个是瓶颈

GPU 利用率 100% 时：
  ████████████████████ 计算：100%

GPU 利用率 60% 时：
  ████ 空闲等数据
  ████████████ 计算

结论：数据加载慢了，多好的 GPU 都在摸鱼。
```

---

## PyTorch DataLoader 优化

### 基本用法

```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # 多进程加载
    pin_memory=True,    # 锁页内存，加速 CPU→GPU 传输
    prefetch_factor=2   # 每个 worker 预取 2 个 batch
)
```

### 关键参数调优

| 参数 | 建议值 | 作用 |
|------|--------|------|
| `num_workers` | 4-8（CPU 核心数/4） | 并行加载进程数 |
| `pin_memory` | `True` | 数据放锁页内存，GPU 拷贝更快 |
| `prefetch_factor` | 2 | 预取 batch，减少等待 |
| `persistent_workers` | `True` | worker 进程不销毁重建 |

---

## 数据存储格式

### 图片/视频数据：不要用 PNG/JPG 直接读

```python
# ❌ 慢：每 epoch 都从磁盘解码图片
for img_path in paths:
    img = Image.open(img_path).convert('RGB')

# ✅ 快：用 WebDataset / LMDB / HDF5 预打包
import webdataset as wds
dataset = wds.WebDataset("data-{0000..0099}.tar").decode("pil")
```

| 格式 | 随机读取 | 顺序读取 | 适合 |
|------|---------|---------|------|
| **LMDB** | 极快 | 快 | 小文件多 |
| **HDF5** | 快 | 最快 | 数组/表格 |
| **WebDataset** | 快 | 快 | 大规模图片 |
| **TFRecord** | 中 | 快 | TensorFlow 生态 |
| **原始文件** | 慢 | 中 | 数据量小 |

### 文本数据：避免逐行读 JSON

```python
# ❌ 慢：每 epoch 都解析 JSON
import json
for line in open('data.jsonl'):
    item = json.loads(line)

# ✅ 快：用 HuggingFace Datasets（memory-mapped）
from datasets import load_dataset
dataset = load_dataset('json', data_files='data.jsonl')
```

---

## 向量数据库

> 向量不再是研究实验的产物，RAG 和语义搜索需要向量数据库。

| 数据库 | 类型 | 特点 |
|--------|------|------|
| **Chroma** | 嵌入式 | 最轻量，适合原型 |
| **Qdrant** | 独立服务 | Rust 实现，性能好 |
| **Milvus** | 分布式 | 十亿级向量，云原生 |
| **Weaviate** | 独立服务 | GraphQL API，Hybrid Search |
| **Pinecone** | 托管云服务 | 零运维，按量付费 |
| **pgvector** | PostgreSQL 插件 | 和业务数据库一体 |

---

## 数据版本管理

> 跑了 3 个月实验后：「这个实验用的是哪个版本的数据来着？」

| 工具 | 特点 |
|------|------|
| **DVC**（Data Version Control） | Git-like，S3/GCS/OSS 后端 |
| **LakeFS** | Git-like，S3 上做分支 |
| **HuggingFace Datasets** | 社区驱动，自带版本 |
| **W&B Artifacts** | 和实验追踪集成 |

---

## 实践清单

| 问题 | 检查 |
|------|------|
| GPU 利用率 < 80%？ | 增大 `num_workers`、启用 `pin_memory` |
| 磁盘 IO 占满？ | 换 LMDB/WebDataset、用 NVMe SSD |
| 数据重复加载？ | 加缓存层、预计算静态特征 |
| 多机共享数据？ | NFS / S3 + FUSE / 对象存储 |

---

## 延伸阅读

- [PyTorch DataLoader 最佳实践](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [WebDataset](https://github.com/webdataset/webdataset)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

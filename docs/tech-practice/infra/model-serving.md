# 模型服务化

> 把模型从「实验室脚本」变成「生产 API」。

---

## 为什么需要模型服务化

```
实验室：python run.py → 看终端输出 → 满意
生产：  用户发请求 → API 返回结果 → 99.9% 可用 → 毫秒级延迟 → 能扛 1000 QPS
```

差距在于：**可靠性、延迟、并发、监控、版本管理**。

---

## 主流服务框架

### vLLM（推荐）

> 高性能推理引擎 + API 服务，最主流的选择。

```bash
# 启动 OpenAI 兼容 API
vllm serve Qwen/Qwen2.5-7B --host 0.0.0.0 --port 8000

# 调用（和 OpenAI API 完全兼容）
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B", "prompt": "Hello,", "max_tokens": 50}'
```

**特点**：
- PagedAttention，吞吐量高
- OpenAI API 兼容，已有代码零改动切换
- 支持连续批处理（Continuous Batching）
- 支持多 GPU 张量并行

### Text Generation Inference (TGI)

> HuggingFace 出品，深度集成 HF 生态。

```bash
docker run -p 8080:80 \
  -e MODEL=Qwen/Qwen2.5-7B \
  ghcr.io/huggingface/text-generation-inference
```

**特点**：
- 量化支持好（GPTQ/AWQ/bitsandbytes）
- 水印功能（Watermarking）
- HuggingFace Hub 模型直接拉取
- 与 HF 生态无缝集成

### Ollama

> 主打「一键运行」，适合本地开发。

```bash
ollama pull llama3
ollama serve
```

**特点**：简单到极致，但不适合高并发生产环境。

---

## API 设计模式

### OpenAI 兼容 API（事实标准）

```
POST /v1/chat/completions   → 对话补全
POST /v1/completions        → 文本补全
POST /v1/embeddings         → 文本嵌入
GET  /v1/models             → 模型列表
```

### 自定义 API 网关

当需要负载均衡、认证、限流时：

```
客户端 → API 网关 (Nginx/FastAPI) → 模型服务 1 (vLLM)
                                   → 模型服务 2 (vLLM)
                                   → 模型服务 3 (vLLM)
```

---

## 关键考量

### 延迟 vs 吞吐量

| | 延迟优先 | 吞吐量优先 |
|------|----------|------------|
| Batch Size | 小（1-4） | 大（32-256） |
| 适用 | 实时对话 | 批量处理 |
| 优化方向 | KV Cache 复用 | Continuous Batching |

### 显存管理

```python
# vLLM 的 GPU 显存利用率配置
vllm serve model \
  --gpu-memory-utilization 0.90  # 用 90% 显存
  --max-model-len 4096           # 最大序列长度
  --max-num-seqs 256             # 最大并发请求数
```

### 健康检查

```python
# 给 vLLM 加一个健康检查端点
@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}
```

---

## 监控与运维

| 维度 | 指标 | 工具 |
|------|------|------|
| **延迟** | P50/P95/P99 延迟 | Prometheus + Grafana |
| **吞吐量** | Token/s、Request/s | vLLM 内置 metrics |
| **显存** | GPU 显存使用率 | `nvidia-smi` / DCGM |
| **错误率** | 4xx/5xx 比例 | 日志分析 |
| **利用率** | GPU 利用率 | `nvidia-smi` |

---

## 延伸阅读

- [vLLM Serving Guide](https://docs.vllm.ai/en/latest/serving/)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference/)
- [Ollama](https://ollama.com/)

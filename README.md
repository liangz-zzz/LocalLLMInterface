# Local LLM Interface

OpenAI兼容的本地LLM推理API服务，支持多种模型类型的自动切换和智能资源管理。

## 特性

- 🔄 **自动模型切换**: 根据API请求自动加载对应模型
- 🚀 **高性能推理**: vLLM + transformers混合架构
- 🔌 **OpenAI兼容**: 客户端只需修改URL，参数完全一致
- 📦 **Docker部署**: 基于NVIDIA PyTorch镜像的容器化部署
- 🎯 **多模型支持**: Chat、Embedding、Reranker模型统一管理
- 🔍 **智能发现**: 自动扫描和识别HuggingFace模型
- 🧠 **智能部署**: GPU+内存混合部署，适配8GB显存限制
- ⚡ **资源优化**: 大模型GPU+内存offloading，小模型纯GPU部署

## 支持的模型类型

| 模型类型 | 推理引擎 | API端点 | 示例模型 |
|---------|---------|---------|----------|
| Chat | vLLM | `/v1/chat/completions` | Qwen2.5-7B-Instruct |
| Embedding | transformers | `/v1/embeddings` | Qwen3-Embedding-0.6B |
| Reranker | transformers | `/v1/rerank` | Qwen3-Reranker-0.6B |

## 快速开始

### 1. 环境准备

确保系统安装了：
- Docker & Docker Compose
- NVIDIA Container Toolkit
- CUDA 12.1+

### 2. 模型准备

将HuggingFace模型下载到 `/mnt/workspace/Source/LLM/` 目录：

```bash
# 示例：下载Qwen模型
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ /mnt/workspace/Source/LLM/Qwen2.5-7B-Instruct-AWQ
```

### 3. 启动服务

```bash
# 使用docker-compose启动
docker-compose up -d

# 或直接使用docker
docker run -d \
  --name local-llm-api \
  --gpus all \
  -p 15530:15530 \
  -v /mnt/workspace/Source/LLM:/models:ro \
  -v .:/app \
  -e LLM_GPU_MEMORY_UTILIZATION=0.8 \
  -e LLM_ENABLE_GPU_MEMORY_OFFLOAD=true \
  -e LLM_GPU_MEMORY_OFFLOAD_THRESHOLD_GB=4.0 \
  local-llm-interface
```

### 4. 验证部署

```bash
# 检查服务状态
curl http://localhost:15530/v1/health

# 查看可用模型
curl http://localhost:15530/v1/models
```

## API使用示例

### Chat Completion

```python
import openai

# 配置客户端（只需修改base_url）
client = openai.OpenAI(
    base_url="http://localhost:15530/v1",
    api_key="not-needed"  # 本地服务无需密钥
)

# 使用Chat模型
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct-AWQ",  # 自动切换到此模型
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Embeddings

```python
# 使用Embedding模型
response = client.embeddings.create(
    model="Qwen3-Embedding-0.6B",  # 自动切换到此模型
    input=["文本1", "文本2", "文本3"]
)

embeddings = [data.embedding for data in response.data]
```

### Reranking

```python
import requests

# 文档重排序
response = requests.post("http://localhost:15530/v1/rerank", json={
    "model": "Qwen3-Reranker-0.6B",  # 自动切换到此模型
    "query": "人工智能的发展",
    "documents": [
        "人工智能是计算机科学的一个分支",
        "机器学习是实现人工智能的方法",
        "深度学习是机器学习的子集"
    ],
    "top_k": 2
})

ranked_docs = response.json()["results"]
```

## 配置选项

通过环境变量配置服务：

```bash
# 模型目录
LLM_MODELS_DIR=/models

# 服务器配置
LLM_HOST=0.0.0.0
LLM_PORT=15530

# GPU配置
LLM_GPU_MEMORY_UTILIZATION=0.8
LLM_CUDA_VISIBLE_DEVICES=0

# 智能部署配置
LLM_ENABLE_GPU_MEMORY_OFFLOAD=true
LLM_GPU_MEMORY_OFFLOAD_THRESHOLD_GB=4.0
LLM_AVAILABLE_GPU_MEMORY_GB=8.0

# vLLM配置
LLM_VLLM_TENSOR_PARALLEL_SIZE=1
LLM_VLLM_DTYPE=auto
LLM_VLLM_MAX_MODEL_LEN=16384

# 模型管理
LLM_AUTO_UNLOAD_MODELS=true
LLM_MODEL_CACHE_SIZE=1

# 日志级别
LLM_LOG_LEVEL=INFO
```

## 项目架构

```
┌─────────────────────────────────────────┐
│           FastAPI服务器                   │
├─────────────────────────────────────────┤
│  OpenAI兼容路由层                        │
│  /v1/chat/completions                   │
│  /v1/embeddings                         │
│  /v1/models                             │
│  /v1/rerank                             │
├─────────────────────────────────────────┤
│           模型管理器                      │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ vLLM引擎    │  │ transformers引擎 │   │
│  │ Chat模型    │  │ Embedding模型   │   │
│  │ 并发推理    │  │ Reranker模型    │   │
│  └─────────────┘  └─────────────────┘   │
├─────────────────────────────────────────┤
│         模型发现和加载层                  │
│  扫描/models目录 → 识别模型类型 → 路由   │
└─────────────────────────────────────────┘
```

## 智能资源管理

本项目实现了针对8GB显存的智能部署策略：

### 部署策略
- **大模型 (>4GB)**: GPU+内存混合部署，避免显存不足
- **小模型 (<4GB)**: 纯GPU部署，充分利用GPU性能
- **自动切换**: 根据模型大小自动选择最优部署方式

### 当前部署状态
- `Qwen2.5-7B-Instruct-AWQ` (10.38GB): GPU+内存混合部署
- `Qwen3-Embedding-0.6B` (1.11GB): 纯GPU部署
- `Qwen3-Reranker-0.6B` (1.11GB): 纯GPU部署

## API文档

- **Swagger UI**: http://localhost:15530/docs
- **ReDoc**: http://localhost:15530/redoc
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)

## 常见问题

### Q: 如何添加新模型？
A: 将HuggingFace模型下载到 `/mnt/workspace/Source/LLM/` 目录，服务会自动发现并支持推理。

### Q: 显存不够怎么办？
A: 系统会自动检测模型大小，大模型使用GPU+内存混合部署，无需手动配置。

### Q: 支持多模型并行吗？
A: 为优化8GB显存使用，同类型一次只加载一个模型，但会自动切换。不同类型可并行。

### Q: 如何切换模型？
A: 无需手动切换，在API请求中指定model参数即可自动切换。

### Q: 支持streaming吗？
A: 支持，在chat/completions请求中设置 `"stream": true`。

### Q: 模型切换会很慢吗？
A: 首次加载需要几秒，之后会缓存在内存中，切换相对快速。

## 开发和扩展

详细的开发指南请参考 [CLAUDE.md](CLAUDE.md)。
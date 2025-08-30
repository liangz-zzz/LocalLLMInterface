# Local LLM Interface API Reference

## Base URL
```
http://localhost:15530
```

## Endpoints

### 1. Chat Completions
```
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "Qwen2.5-7B-Instruct-AWQ",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1734567890,
  "model": "Qwen2.5-7B-Instruct-AWQ",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么可以帮助您的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 12,
    "total_tokens": 20
  }
}
```

### 2. Embeddings
```
POST /v1/embeddings
```

**Request Body:**
```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": "文本向量化测试"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, -0.2, 0.3, ...]
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

### 3. Reranking
```
POST /v1/rerank
```

**Request Body:**
```json
{
  "model": "Qwen3-Reranker-0.6B",
  "query": "什么是人工智能？",
  "documents": [
    "人工智能是计算机科学的一个分支",
    "机器学习是AI的重要组成部分"
  ],
  "top_k": 2
}
```

**Response:**
```json
{
  "model": "Qwen3-Reranker-0.6B",
  "results": [
    {
      "index": 0,
      "score": 0.709,
      "document": "人工智能是计算机科学的一个分支"
    },
    {
      "index": 1,
      "score": 0.623,
      "document": "机器学习是AI的重要组成部分"
    }
  ]
}
```

## Available Models

### Chat Models
- `Qwen2.5-7B-Instruct-AWQ` (10.38GB) - 使用GPU+内存混合部署

### Embedding Models  
- `Qwen3-Embedding-0.6B` (1.11GB) - 使用纯GPU部署

### Reranker Models
- `Qwen3-Reranker-0.6B` (1.11GB) - 使用纯GPU部署

## Model Management

### List Models
```
GET /v1/models
```

### Health Check
```
GET /health
```

## OpenAI SDK兼容

可以直接使用OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:15530/v1",
    api_key="dummy"  # 不需要真实key
)

# Chat
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct-AWQ",
    messages=[{"role": "user", "content": "你好"}]
)

# Embeddings  
embedding = client.embeddings.create(
    model="Qwen3-Embedding-0.6B",
    input="测试文本"
)
```

## 注意事项

1. **自动模型切换**: 第一次调用新模型时会自动加载，可能需要等待几秒
2. **资源管理**: 大模型使用GPU+内存混合部署，小模型使用纯GPU
3. **并发限制**: 同时只能加载一个同类型模型
4. **Reranker**: 单文档重排序正常，多文档可能有padding token警告
# 本地LLM推理服务架构设计

## 项目概述
本项目旨在搭建一个**本地LLM推理API服务**，提供OpenAI兼容的本地推理能力。

## 核心需求

### 1. 容器化部署
- **基础镜像**: `nvcr.io/nvidia/pytorch:24.10-py3`
- **GPU支持**: NVIDIA Docker运行时
- **文件挂载**: `/mnt/workspace/Source/LLM` → 容器内模型目录

### 2. API兼容性设计
- **OpenAI格式兼容**: 客户端只需修改URL，其他参数保持一致
- **容器间通信**: 支持Docker客户端调用
- **自动模型切换**: 根据API请求中的model参数自动切换模型

### 3. 多模型支持架构
- **模型来源**: Hugging Face下载的权重文件
- **动态发现**: 扫描LLM目录，自动识别可用模型
- **统一框架**: transformers生态统一处理

### 4. 智能资源管理
- **模型热切换**: 自动卸载当前模型→加载新模型
- **GPU内存优化**: 一次只运行一个模型
- **并发就绪**: 框架支持未来并发推理

## 技术架构

### 推理框架选择：vLLM + transformers 混合架构

**vLLM引擎** (处理Chat模型)
- ✅ 支持大模型（70B+）和高性能推理
- ✅ 原生多模型并行和并发请求支持  
- ✅ PagedAttention内存优化
- ✅ 适合Qwen2.5-7B-Instruct类型

**transformers引擎** (处理其他模型)
- ✅ 完整HuggingFace生态兼容
- ✅ 支持Embedding、Reranker模型
- ✅ 非标准模型灵活接入
- ✅ 自定义接口扩展能力

### 系统架构图

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

## 项目结构

```
local-llm-interface/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI应用入口
│   ├── config.py              # 配置管理
│   └── api/
│       ├── __init__.py
│       ├── chat.py           # Chat API路由
│       ├── embeddings.py     # Embedding API路由
│       ├── models.py         # 模型列表API
│       └── rerank.py         # Reranker API路由
├── engines/
│   ├── __init__.py
│   ├── base.py               # 推理引擎基类
│   ├── vllm_engine.py        # vLLM推理引擎
│   └── transformers_engine.py # transformers推理引擎
├── models/
│   ├── __init__.py
│   ├── manager.py            # 模型管理器
│   ├── discovery.py          # 模型发现
│   └── types.py              # 模型类型定义
├── utils/
│   ├── __init__.py
│   ├── openai_compat.py      # OpenAI格式转换
│   └── logging.py            # 日志配置
├── Dockerfile
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 核心组件设计

### 1. 模型管理器 (`models/manager.py`)
负责模型的自动发现、加载、切换和资源管理。

```python
class ModelManager:
    def __init__(self):
        self.current_chat_model = None
        self.current_embedding_model = None
        self.current_reranker_model = None
        
    async def switch_model(self, model_name: str, model_type: str):
        """自动模型切换逻辑"""
        # 检查当前加载的模型
        # 如果不匹配：卸载当前 → 加载目标 → 更新状态
        # 返回切换结果
```

### 2. 双引擎架构 (`engines/`)

**vLLM引擎**：
- 处理Chat类型模型（文件名包含"Instruct"/"Chat"）
- 高性能推理和并发支持
- PagedAttention内存优化

**transformers引擎**：
- 处理Embedding、Reranker模型
- 支持非标准模型接入
- 完整HuggingFace兼容性

### 3. OpenAI兼容层 (`api/`)
每个API端点的工作流程：
1. 接收OpenAI格式请求
2. 检查请求中的model参数
3. 调用模型管理器进行自动切换
4. 执行推理
5. 返回OpenAI格式响应

### 4. 模型发现机制 (`models/discovery.py`)
```python
def discover_models(models_dir: str) -> List[ModelInfo]:
    """扫描模型目录，识别模型类型"""
    # 文件名包含"Instruct"/"Chat" → vLLM引擎
    # 文件名包含"Embedding" → transformers引擎  
    # 文件名包含"Reranker" → transformers引擎
    # 其他 → 可配置引擎选择
```

## API接口设计

### 自动模型切换流程
```
客户端请求 → 检查model参数 → 自动切换模型 → 执行推理 → 返回结果
```

### API端点
- `GET /v1/models` - 返回可用模型列表（只读）
- `POST /v1/chat/completions` - 自动切换Chat模型并推理
- `POST /v1/embeddings` - 自动切换Embedding模型  
- `POST /v1/rerank` - 自动切换Reranker模型
- `GET /health` - 健康检查

### Docker部署配置
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
VOLUME ["/models"]           # 挂载/mnt/workspace/Source/LLM
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 关键特性

### ✅ 透明模型切换
- 客户端无感知，完全OpenAI兼容
- 不需要额外的模型管理调用
- 按需加载，自动释放GPU内存

### ✅ 多模型类型支持
- Chat模型：vLLM高性能推理
- Embedding/Reranker：transformers完整兼容
- 非标准模型：灵活扩展接口

### ✅ 可扩展架构
- 支持未来大模型（70B+）
- 多模型并行就绪
- 并发推理框架支持

### ✅ 简化部署
- Docker标准化部署
- 即插即用：下载模型→自动支持推理
- 最小化配置复杂度
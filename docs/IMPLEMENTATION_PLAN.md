# 视觉模型集成实施计划

## 阶段1: 基础扩展 (第1-2天)

### 1.1 扩展模型类型系统
**文件**: `models/types.py`
```python
# 新增模型类型
class ModelType(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VISION = "vision"           # 新增
    MULTIMODAL = "multimodal"   # 新增
    UNKNOWN = "unknown"

# 新增请求/响应模型
class ImageInput(BaseModel):
    """图像输入格式"""
    data: str  # base64编码的图像数据
    url: Optional[str] = None  # 或者图像URL
    path: Optional[str] = None  # 或者本地路径

class VisionEncodeRequest(BaseModel):
    """视觉编码请求"""
    model: str
    images: List[Union[str, ImageInput]]
    normalize: bool = True
    return_tensors: bool = False

class VisionEncodeResponse(BaseModel):
    """视觉编码响应"""
    model: str
    embeddings: List[List[float]]
    dimensions: int
    processing_time: float
```

### 1.2 更新模型发现
**文件**: `models/discovery.py`
- 修改 `_determine_model_type()` 方法以识别CLIP和DINOv3
- 添加视觉模型特定的配置读取逻辑

## 阶段2: Vision引擎实现 (第3-4天)

### 2.1 创建Vision引擎基类
**新文件**: `engines/vision_engine.py`

```python
class VisionEngine(BaseEngine):
    """视觉模型推理引擎"""
    
    def __init__(self, model_info: ModelInfo, use_gpu: bool = True):
        super().__init__(model_info)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.processor = None
        self.model = None
    
    async def load_model(self) -> None:
        """加载模型和处理器"""
        # 实现模型加载逻辑
        pass
    
    async def encode_image(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        """编码单个图像"""
        pass
    
    async def encode_text(self, text: str) -> np.ndarray:
        """编码文本(仅CLIP)"""
        pass
    
    async def encode_batch(self, inputs: List[Any]) -> List[np.ndarray]:
        """批量编码"""
        pass
```

### 2.2 实现具体模型支持
**子类实现**:
- `CLIPEngine`: 支持图像-文本联合编码
- `DINOv3Engine`: 专注于图像特征提取

## 阶段3: API层实现 (第5-6天)

### 3.1 视觉API端点
**新文件**: `api/vision.py`
- `/v1/vision/encode` - 图像编码
- `/v1/vision/similarity` - 相似度计算
- `/v1/vision/batch` - 批量处理

### 3.2 多模态API端点
**新文件**: `api/multimodal.py`
- `/v1/multimodal/encode` - 图像/文本编码
- `/v1/multimodal/match` - 图像-文本匹配
- `/v1/multimodal/search` - 跨模态搜索

### 3.3 集成到主应用
**文件**: `app/main.py`
```python
# 添加新路由
app.include_router(vision.router)
app.include_router(multimodal.router)
```

## 阶段4: 模型管理器更新 (第7天)

### 4.1 扩展ModelManager
**文件**: `models/manager.py`
- 添加视觉模型的加载逻辑
- 实现视觉引擎的生命周期管理
- 更新引擎选择逻辑

```python
def _select_engine(self, model_info: ModelInfo) -> BaseEngine:
    """根据模型类型选择引擎"""
    if model_info.type in [ModelType.VISION, ModelType.MULTIMODAL]:
        return VisionEngine(model_info, use_gpu=self.use_gpu)
    # ... 原有逻辑
```

## 阶段5: 测试与优化 (第8-9天)

### 5.1 单元测试
**新文件**: `tests/test_vision.py`
- 测试图像编码
- 测试文本编码(CLIP)
- 测试相似度计算
- 测试批处理

### 5.2 集成测试
**新文件**: `tests/test_integration_vision.py`
- 端到端API测试
- 性能基准测试
- 内存使用测试

### 5.3 性能优化
- 实现图像预处理缓存
- 添加批处理优化
- GPU内存管理优化

## 阶段6: 文档与部署 (第10天)

### 6.1 更新文档
- 更新README.md
- 创建VISION_API_REFERENCE.md
- 添加使用示例

### 6.2 Docker配置
**文件**: `docker-compose.yml`, `Dockerfile`
- 添加视觉处理依赖
- 配置环境变量
- 优化镜像大小

### 6.3 部署测试
- 容器化测试
- 资源监控
- 性能验证

## 实施时间表

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|----------|--------|
| 1 | 基础扩展 | 2天 | 高 |
| 2 | Vision引擎 | 2天 | 高 |
| 3 | API实现 | 2天 | 高 |
| 4 | 管理器更新 | 1天 | 中 |
| 5 | 测试优化 | 2天 | 中 |
| 6 | 文档部署 | 1天 | 低 |

## 关键技术点

### 1. 图像处理
```python
def process_image_input(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
    """统一图像输入处理"""
    if isinstance(image_input, str):
        if image_input.startswith('http'):
            # URL
            response = requests.get(image_input)
            image = Image.open(BytesIO(response.content))
        elif image_input.startswith('data:image'):
            # base64 data URL
            base64_str = image_input.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(base64_str)))
        elif os.path.exists(image_input):
            # 文件路径
            image = Image.open(image_input)
        else:
            # 纯base64
            image = Image.open(BytesIO(base64.b64decode(image_input)))
    elif isinstance(image_input, bytes):
        image = Image.open(BytesIO(image_input))
    else:
        image = image_input
    
    return image.convert('RGB')
```

### 2. 内存优化
```python
@torch.no_grad()
def encode_with_memory_optimization(self, inputs):
    """内存优化的编码"""
    if self.use_gpu:
        # 使用mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs)
    else:
        outputs = self.model(**inputs)
    
    # 立即转移到CPU并清理GPU内存
    result = outputs.cpu().numpy()
    if self.use_gpu:
        torch.cuda.empty_cache()
    
    return result
```

### 3. 批处理优化
```python
async def process_batch(self, items: List[Any], batch_size: int = 32):
    """批量处理优化"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # 并行预处理
        processed = await asyncio.gather(*[
            self.preprocess(item) for item in batch
        ])
        
        # 批量推理
        batch_results = await self.encode_batch(processed)
        results.extend(batch_results)
    
    return results
```

## 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| GPU内存不足 | 高 | 实现动态批大小调整 |
| 模型加载慢 | 中 | 预加载常用模型 |
| API兼容性 | 低 | 保持独立的视觉API |
| 依赖冲突 | 中 | 使用独立的虚拟环境 |

## 成功标准

1. ✅ CLIP和DINOv3模型能成功加载和推理
2. ✅ API响应时间 < 100ms (单图像)
3. ✅ 批处理吞吐量 > 100 images/s
4. ✅ GPU内存使用 < 4GB
5. ✅ 完整的API文档和示例
6. ✅ 通过所有测试用例

## 后续扩展

1. **更多模型支持**: SAM, DINO, BLIP等
2. **高级功能**: 图像生成、视频理解
3. **优化技术**: TensorRT、ONNX加速
4. **应用集成**: RAG系统、搜索引擎
# 视觉模型集成设计方案

## 1. 概述

本文档描述了如何将CLIP和DINOv3等视觉模型集成到现有的LocalLLMInterface框架中。

### 支持的模型
- **CLIP (openai/clip-vit-base-patch32)**: 视觉-语言对齐模型，支持图像和文本的联合编码
- **DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m)**: 自监督视觉Transformer模型，专注于图像特征提取

## 2. 系统架构扩展

### 2.1 新增模型类型

```python
# models/types.py 扩展
class ModelType(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VISION = "vision"           # 新增：纯视觉模型
    MULTIMODAL = "multimodal"   # 新增：多模态模型
    UNKNOWN = "unknown"
```

### 2.2 新增API端点

#### 视觉编码API
```
POST /v1/vision/encode
功能：将图像编码为特征向量
输入：图像(base64/URL/路径)
输出：特征向量
```

#### 图像相似度API
```
POST /v1/vision/similarity
功能：计算图像之间的相似度
输入：多个图像
输出：相似度矩阵
```

#### 多模态编码API
```
POST /v1/multimodal/encode
功能：同时编码图像和文本(CLIP专用)
输入：图像和/或文本
输出：统一的特征向量
```

#### 图像-文本匹配API
```
POST /v1/multimodal/match
功能：计算图像和文本的匹配度
输入：图像列表和文本列表
输出：匹配分数矩阵
```

## 3. 实现方案

### 3.1 模型发现增强

```python
# models/discovery.py 修改
def _determine_model_type(self, model_id: str, config: Optional[Dict]) -> ModelType:
    """增强模型类型识别，支持视觉模型"""
    model_id_lower = model_id.lower()
    
    # 检查视觉模型标识
    if 'clip' in model_id_lower:
        return ModelType.MULTIMODAL
    elif any(keyword in model_id_lower for keyword in ['dino', 'vit', 'vision']):
        return ModelType.VISION
    
    # 检查架构
    if config:
        architectures = config.get("architectures", [])
        if architectures:
            arch_str = " ".join(architectures).lower()
            if 'clip' in arch_str:
                return ModelType.MULTIMODAL
            elif any(keyword in arch_str for keyword in ['vit', 'dino', 'vision']):
                return ModelType.VISION
    
    # ... 原有逻辑
```

### 3.2 新增Vision引擎

```python
# engines/vision_engine.py
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import base64
from io import BytesIO

class VisionEngine(BaseEngine):
    """视觉模型推理引擎"""
    
    async def load_model(self) -> None:
        """加载视觉模型"""
        if self.model_info.type == ModelType.MULTIMODAL:
            # CLIP模型需要processor和model
            self.processor = AutoProcessor.from_pretrained(self.model_info.path)
            self.model = AutoModel.from_pretrained(
                self.model_info.path,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else "cpu"
            )
        elif self.model_info.type == ModelType.VISION:
            # DINOv3等纯视觉模型
            self.processor = AutoProcessor.from_pretrained(self.model_info.path)
            self.model = AutoModel.from_pretrained(
                self.model_info.path,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else "cpu"
            )
    
    async def encode_image(self, image_input: Union[str, bytes, Image.Image]) -> List[float]:
        """编码单个图像"""
        image = self._process_image_input(image_input)
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            if self.model_info.type == ModelType.MULTIMODAL:
                # CLIP: 获取图像特征
                outputs = self.model.get_image_features(**inputs)
            else:
                # DINOv3: 获取CLS token或平均池化
                outputs = self.model(**inputs)
                outputs = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return outputs.cpu().numpy().tolist()[0]
    
    async def encode_text(self, text: str) -> List[float]:
        """编码文本(仅CLIP支持)"""
        if self.model_info.type != ModelType.MULTIMODAL:
            raise ValueError(f"Model {self.model_info.id} does not support text encoding")
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        
        return outputs.cpu().numpy().tolist()[0]
    
    async def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """计算特征相似度"""
        # 归一化
        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        
        # 余弦相似度
        similarity = (features1 @ features2.T).item()
        return similarity
```

### 3.3 API实现

```python
# api/vision.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union

router = APIRouter(prefix="/v1/vision")

class VisionEncodeRequest(BaseModel):
    model: str
    images: List[str]  # base64编码或URL或文件路径
    normalize: bool = True

class VisionEncodeResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    dimensions: int

@router.post("/encode")
async def encode_images(request: VisionEncodeRequest) -> VisionEncodeResponse:
    """编码图像为特征向量"""
    manager = get_model_manager()
    
    # 确保模型已加载
    await manager.ensure_model_loaded(request.model)
    
    # 获取引擎
    engine = manager.get_engine(request.model)
    if not isinstance(engine, VisionEngine):
        raise HTTPException(400, f"Model {request.model} is not a vision model")
    
    # 编码所有图像
    embeddings = []
    for image_input in request.images:
        embedding = await engine.encode_image(image_input)
        if request.normalize:
            # 归一化
            norm = np.linalg.norm(embedding)
            embedding = (embedding / norm).tolist()
        embeddings.append(embedding)
    
    return VisionEncodeResponse(
        model=request.model,
        embeddings=embeddings,
        dimensions=len(embeddings[0])
    )

# api/multimodal.py
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/v1/multimodal")

class MultimodalEncodeRequest(BaseModel):
    model: str
    images: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    normalize: bool = True

@router.post("/encode")
async def encode_multimodal(request: MultimodalEncodeRequest):
    """编码图像和文本到统一的特征空间"""
    manager = get_model_manager()
    await manager.ensure_model_loaded(request.model)
    
    engine = manager.get_engine(request.model)
    if not isinstance(engine, VisionEngine) or engine.model_info.type != ModelType.MULTIMODAL:
        raise HTTPException(400, f"Model {request.model} is not a multimodal model")
    
    results = {"model": request.model}
    
    # 编码图像
    if request.images:
        image_embeddings = []
        for image in request.images:
            embedding = await engine.encode_image(image)
            if request.normalize:
                embedding = normalize_vector(embedding)
            image_embeddings.append(embedding)
        results["image_embeddings"] = image_embeddings
    
    # 编码文本
    if request.texts:
        text_embeddings = []
        for text in request.texts:
            embedding = await engine.encode_text(text)
            if request.normalize:
                embedding = normalize_vector(embedding)
            text_embeddings.append(embedding)
        results["text_embeddings"] = text_embeddings
    
    return results

class MultimodalMatchRequest(BaseModel):
    model: str
    images: List[str]
    texts: List[str]

@router.post("/match")
async def match_image_text(request: MultimodalMatchRequest):
    """计算图像和文本的匹配分数"""
    manager = get_model_manager()
    await manager.ensure_model_loaded(request.model)
    
    engine = manager.get_engine(request.model)
    
    # 编码所有图像和文本
    image_features = []
    for image in request.images:
        features = await engine.encode_image(image)
        image_features.append(torch.tensor(features))
    
    text_features = []
    for text in request.texts:
        features = await engine.encode_text(text)
        text_features.append(torch.tensor(features))
    
    # 计算相似度矩阵
    image_features = torch.stack(image_features)
    text_features = torch.stack(text_features)
    
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarity_matrix = (image_features @ text_features.T).tolist()
    
    return {
        "model": request.model,
        "similarity_matrix": similarity_matrix,
        "shape": [len(request.images), len(request.texts)]
    }
```

## 4. 配置更新

### 4.1 Docker环境变量
```yaml
# docker-compose.yml 新增
environment:
  # 视觉模型配置
  - LLM_VISION_BATCH_SIZE=32
  - LLM_VISION_MAX_IMAGE_SIZE=1024
  - LLM_VISION_CACHE_SIZE=100
  - LLM_ENABLE_VISION_MODELS=true
```

### 4.2 依赖更新
```txt
# requirements.txt 新增
Pillow>=10.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
```

## 5. 使用示例

### 5.1 CLIP模型使用

```python
import requests
import base64

# 编码图像
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:15530/v1/multimodal/encode",
    json={
        "model": "clip-vit-base-patch32",
        "images": [image_base64],
        "texts": ["a photo of a cat", "a photo of a dog"]
    }
)

# 获取相似度
similarity = response.json()
```

### 5.2 DINOv3模型使用

```python
# 提取图像特征
response = requests.post(
    "http://localhost:15530/v1/vision/encode",
    json={
        "model": "dinov3-vitl16-pretrain-lvd1689m",
        "images": [image_base64]
    }
)

features = response.json()["embeddings"][0]
```

## 6. 实现步骤

1. **扩展模型类型枚举** - 添加VISION和MULTIMODAL类型
2. **更新模型发现逻辑** - 识别视觉模型
3. **实现VisionEngine** - 处理图像和文本编码
4. **添加API端点** - 实现视觉和多模态API
5. **更新依赖** - 添加图像处理库
6. **测试集成** - 验证CLIP和DINOv3功能

## 7. 性能优化建议

1. **批处理**: 支持批量图像编码以提高吞吐量
2. **图像预处理缓存**: 缓存处理后的图像张量
3. **模型量化**: 使用INT8量化减少内存占用
4. **异步处理**: 使用异步队列处理大批量请求
5. **GPU优化**: 使用mixed precision和torch.compile

## 8. 注意事项

1. **内存管理**: 视觉模型通常比文本模型占用更多内存
2. **图像格式**: 支持多种输入格式(base64, URL, 文件路径)
3. **错误处理**: 处理损坏的图像和不支持的格式
4. **安全性**: 验证图像大小和来源，防止恶意输入
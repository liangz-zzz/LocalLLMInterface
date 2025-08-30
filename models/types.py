"""Model type definitions"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class ModelType(str, Enum):
    """Model type enumeration"""
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    UNKNOWN = "unknown"


class EngineType(str, Enum):
    """Inference engine type"""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


class ModelStatus(str, Enum):
    """Model loading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelInfo(BaseModel):
    """Model information"""
    id: str = Field(description="Model identifier (directory name)")
    name: str = Field(description="Model display name")
    type: ModelType = Field(description="Model type")
    engine: EngineType = Field(description="Recommended inference engine")
    path: str = Field(description="Full path to model directory")
    size_gb: Optional[float] = Field(default=None, description="Model size in GB")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Model config.json content")
    status: ModelStatus = Field(default=ModelStatus.UNLOADED, description="Current loading status")
    loaded_at: Optional[datetime] = Field(default=None, description="When model was loaded")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelListResponse(BaseModel):
    """Response for /v1/models endpoint"""
    object: str = "list"
    data: List[Dict[str, Any]] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """Chat message format"""
    role: str = Field(description="Message role (system, user, assistant)")
    content: str = Field(description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(description="Model name to use")
    messages: List[ChatMessage] = Field(description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0, le=1, description="Top-p sampling")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Stream response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    n: int = Field(default=1, ge=1, description="Number of completions")
    presence_penalty: float = Field(default=0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: float = Field(default=0, ge=-2, le=2, description="Frequency penalty")
    user: Optional[str] = Field(default=None, description="User identifier")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    model: str = Field(description="Model name to use")
    input: List[str] = Field(description="Input texts to embed")
    encoding_format: str = Field(default="float", description="Encoding format")


class EmbeddingData(BaseModel):
    """Embedding data"""
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


class RerankRequest(BaseModel):
    """Reranking request"""
    model: str = Field(description="Model name to use")
    query: str = Field(description="Query text")
    documents: List[str] = Field(description="Documents to rerank")
    top_k: Optional[int] = Field(default=None, description="Number of top documents to return")


class RerankResult(BaseModel):
    """Single rerank result"""
    index: int = Field(description="Original document index")
    score: float = Field(description="Relevance score")
    document: str = Field(description="Document text")


class RerankResponse(BaseModel):
    """Reranking response"""
    model: str
    results: List[RerankResult]
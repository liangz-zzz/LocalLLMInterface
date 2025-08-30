"""Base inference engine interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from models.types import (
    ModelInfo, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse
)


class InferenceEngine(ABC):
    """Abstract base class for inference engines"""
    
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load model into memory"""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model from memory"""
        pass
    
    @abstractmethod
    async def generate_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate chat completion"""
        pass
    
    async def generate_chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Generate streaming chat completion"""
        raise NotImplementedError("Streaming not implemented for this engine")
    
    async def generate_embeddings(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Generate embeddings"""
        raise NotImplementedError("Embeddings not supported by this engine")
    
    async def rerank_documents(
        self,
        request: RerankRequest
    ) -> RerankResponse:
        """Rerank documents"""
        raise NotImplementedError("Reranking not supported by this engine")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                "gpu_memory_free": (torch.cuda.get_device_properties(0).total_memory - 
                                   torch.cuda.memory_allocated()) / 1024**3       # GB
            }
        return {
            "gpu_memory_allocated": 0,
            "gpu_memory_reserved": 0,
            "gpu_memory_free": 0
        }
    
    def clear_cache(self) -> None:
        """Clear GPU cache"""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
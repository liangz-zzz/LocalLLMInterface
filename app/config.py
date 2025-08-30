"""Configuration management for Local LLM Interface"""

from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=15530, description="Server port")
    
    # Model paths
    models_dir: str = Field(
        default="/models",
        description="Directory containing model files (mapped from /mnt/workspace/Source/LLM)"
    )
    
    # GPU configuration
    gpu_memory_utilization: float = Field(
        default=0.9,
        description="Fraction of GPU memory to use for model loading"
    )
    cuda_visible_devices: Optional[str] = Field(
        default=None,
        description="CUDA devices to use (e.g., '0,1')"
    )
    
    # vLLM configuration
    vllm_max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum sequence length for vLLM models"
    )
    vllm_tensor_parallel_size: int = Field(
        default=1,
        description="Number of GPUs for tensor parallelism"
    )
    vllm_dtype: str = Field(
        default="auto",
        description="Data type for vLLM models (auto, float16, bfloat16)"
    )
    
    # Transformers configuration
    transformers_device: str = Field(
        default="cuda",
        description="Device for transformers models (cuda, cpu)"
    )
    transformers_torch_dtype: str = Field(
        default="auto",
        description="Torch dtype for transformers models"
    )
    
    # API configuration
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for authentication"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    # Model switching
    auto_unload_models: bool = Field(
        default=True,
        description="Automatically unload models when switching"
    )
    model_cache_size: int = Field(
        default=1,
        description="Number of models to keep in memory (1 = only current model)"
    )
    
    # Hybrid GPU+Memory deployment for large models
    enable_gpu_memory_offload: bool = Field(
        default=True,
        description="Enable GPU+System Memory offloading for large models"
    )
    gpu_memory_offload_threshold_gb: float = Field(
        default=4.0,
        description="Model size threshold (GB) for GPU+Memory offloading"
    )
    available_gpu_memory_gb: float = Field(
        default=8.0,
        description="Available GPU memory in GB (used for intelligent allocation)"
    )
    
    class Config:
        env_prefix = "LLM_"
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
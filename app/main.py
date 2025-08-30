"""FastAPI application entry point"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.config import settings
from app.api import models, chat, embeddings, rerank
from models.manager import model_manager


# Configure logging
logger.remove()

# Console logging (colored)
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# File logging (plain text)
logger.add(
    "/app/logs/app.log",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="100 MB",
    retention="7 days",
    compression="zip"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Local LLM Interface")
    logger.info(f"Models directory: {settings.models_dir}")
    logger.info(f"GPU memory utilization: {settings.gpu_memory_utilization}")
    
    # Refresh models on startup
    available_models = model_manager.get_available_models()
    logger.info(f"Found {len(available_models)} available models")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local LLM Interface")
    await model_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Local LLM Interface",
    description="""
    **OpenAI-compatible API for local LLM inference**
    
    支持Chat、Embedding和Reranker三种模型类型的自动切换和智能资源管理。
    专为8GB显存环境优化，实现GPU+内存混合部署策略。
    
    ## 主要特性
    
    - 🚀 **完全兼容OpenAI API** - 无缝替换，只需修改URL
    - 🧠 **智能模型切换** - 自动卸载和加载，优化内存使用
    - ⚡ **混合部署策略** - 大模型GPU+内存，小模型纯GPU
    - 📊 **实时监控** - GPU内存使用、模型状态一目了然
    
    ## 支持的端点
    
    - **Chat Completions** - `/v1/chat/completions`
    - **Embeddings** - `/v1/embeddings` 
    - **Reranking** - `/v1/rerank`
    - **Models** - `/v1/models`
    - **Health Check** - `/v1/health`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(embeddings.router)
app.include_router(rerank.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Local LLM Interface - OpenAI Compatible API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/v1/health",
        "models_url": "/v1/models"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )
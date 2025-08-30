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
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
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
    description="OpenAI-compatible API for local LLM inference",
    version="0.1.0",
    lifespan=lifespan
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
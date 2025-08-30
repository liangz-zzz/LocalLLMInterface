"""Models API endpoints - OpenAI compatible"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from loguru import logger

from app.config import settings
from models.manager import model_manager
from models.types import ModelListResponse


router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List all available models - OpenAI compatible endpoint"""
    try:
        available_models = model_manager.get_available_models()
        
        # Convert to OpenAI format
        model_data = []
        for model_id, model_info in available_models.items():
            model_data.append({
                "id": model_id,
                "object": "model",
                "created": int(model_info.loaded_at.timestamp()) if model_info.loaded_at else 0,
                "owned_by": "local",
                "permission": [
                    {
                        "id": f"modelperm-{model_id}",
                        "object": "model_permission",
                        "created": int(model_info.loaded_at.timestamp()) if model_info.loaded_at else 0,
                        "allow_create_engine": True,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": model_id,
                "parent": None,
                # Custom fields for our service
                "type": model_info.type.value,
                "engine": model_info.engine.value,
                "size_gb": model_info.size_gb,
                "status": model_info.status.value,
                "path": model_info.path
            })
        
        return {
            "object": "list",
            "data": model_data
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {e}")


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model"""
    try:
        available_models = model_manager.get_available_models()
        
        if model_id not in available_models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        model_info = available_models[model_id]
        
        return {
            "id": model_id,
            "object": "model",
            "created": int(model_info.loaded_at.timestamp()) if model_info.loaded_at else 0,
            "owned_by": "local",
            "permission": [
                {
                    "id": f"modelperm-{model_id}",
                    "object": "model_permission",
                    "created": int(model_info.loaded_at.timestamp()) if model_info.loaded_at else 0,
                    "allow_create_engine": True,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ],
            "root": model_id,
            "parent": None,
            "type": model_info.type.value,
            "engine": model_info.engine.value,
            "size_gb": model_info.size_gb,
            "status": model_info.status.value,
            "path": model_info.path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model: {e}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Get memory usage
        total_memory = 0
        used_memory = 0
        
        for engine in model_manager.engines.values():
            memory_info = engine.get_memory_usage()
            used_memory += memory_info.get("gpu_memory_allocated", 0)
        
        # Get loaded models count
        loaded_models = sum(1 for engine in model_manager.engines.values() if engine.is_loaded)
        available_models = len(model_manager.get_available_models())
        
        return {
            "status": "healthy",
            "models": {
                "available": available_models,
                "loaded": loaded_models,
                "current_chat": model_manager.get_current_model("chat"),
                "current_embedding": model_manager.get_current_model("embedding"),
                "current_reranker": model_manager.get_current_model("reranker")
            },
            "memory": {
                "gpu_memory_used_gb": round(used_memory, 2),
                "gpu_memory_utilization": settings.gpu_memory_utilization
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")
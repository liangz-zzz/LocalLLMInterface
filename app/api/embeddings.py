"""Embeddings API - OpenAI compatible"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from models.manager import model_manager
from models.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelType
)


router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings - OpenAI compatible endpoint"""
    try:
        logger.info(f"Embedding request for model: {request.model}, {len(request.input)} texts")
        
        # Ensure model is loaded (automatic switching)
        model_info = await model_manager.ensure_model_loaded(request.model)
        
        # Verify it's an embedding model
        if model_info.type != ModelType.EMBEDDING:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not an embedding model (type: {model_info.type})"
            )
        
        # Get engine
        engine = model_manager.get_engine(request.model)
        if not engine:
            raise HTTPException(
                status_code=500,
                detail=f"Engine not found for model '{request.model}'"
            )
        
        # Generate embeddings
        response = await engine.generate_embeddings(request)
        
        logger.info(f"Embeddings generated successfully for model: {request.model}")
        return response
        
    except ValueError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {e}")
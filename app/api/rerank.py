"""Rerank API endpoints"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from models.manager import model_manager
from models.types import (
    RerankRequest,
    RerankResponse,
    ModelType
)


router = APIRouter(prefix="/v1", tags=["rerank"])


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on relevance to query"""
    try:
        logger.info(f"Rerank request for model: {request.model}, {len(request.documents)} documents")
        
        # Ensure model is loaded (automatic switching)
        model_info = await model_manager.ensure_model_loaded(request.model)
        
        # Verify it's a reranker model
        if model_info.type != ModelType.RERANKER:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a reranker model (type: {model_info.type})"
            )
        
        # Get engine
        engine = model_manager.get_engine(request.model)
        if not engine:
            raise HTTPException(
                status_code=500,
                detail=f"Engine not found for model '{request.model}'"
            )
        
        # Rerank documents
        response = await engine.rerank_documents(request)
        
        logger.info(f"Documents reranked successfully for model: {request.model}")
        return response
        
    except ValueError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error reranking documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error reranking documents: {e}")
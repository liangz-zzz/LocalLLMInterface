"""Embeddings API - OpenAI compatible

This endpoint supports optional prompt controls for embedding models:
- `prompt_name`: Use a named, model-provided template (e.g., "query").
- `prompt`: Provide a custom instruction string (or list of strings),
  which is passed through to SentenceTransformers or emulated when
  using raw Transformers fallback.

Examples:
- Named prompt: {"model": "Qwen3-Embedding-0.6B", "input": ["..."], "prompt_name": "query"}
- Custom prompt: {"model": "Qwen3-Embedding-0.6B", "input": ["..."],
                 "prompt": "Given a web search query, retrieve relevant passages that answer the query"}
"""

from fastapi import APIRouter, HTTPException, Body
from loguru import logger

from models.manager import model_manager
from models.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelType
)


router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest = Body(
        ...,
        examples={
            "basic": {
                "summary": "Basic embedding",
                "value": {
                    "model": "Qwen3-Embedding-0.6B",
                    "input": ["测试文本"]
                },
            },
            "with_prompt_name": {
                "summary": "Use named prompt (query)",
                "value": {
                    "model": "Qwen3-Embedding-0.6B",
                    "input": [
                        "What is the capital of China?",
                        "Explain gravity"
                    ],
                    "prompt_name": "query"
                },
            },
            "with_custom_prompt": {
                "summary": "Use custom prompt string",
                "value": {
                    "model": "Qwen3-Embedding-0.6B",
                    "input": [
                        "What is the capital of China?",
                        "Explain gravity"
                    ],
                    "prompt": "Given a web search query, retrieve relevant passages that answer the query"
                },
            },
            "per_input_prompts": {
                "summary": "Per-input custom prompts",
                "value": {
                    "model": "Qwen3-Embedding-0.6B",
                    "input": [
                        "What is the capital of China?",
                        "Explain gravity"
                    ],
                    "prompt": [
                        "Represent this as a search query",
                        "Represent this as a search query"
                    ]
                },
            },
        },
    )
):
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

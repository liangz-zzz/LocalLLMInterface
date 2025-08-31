"""Vision API endpoints for image encoding and processing"""

import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from models.types import (
    VisionEncodeRequest,
    VisionEncodeResponse,
    ModelType
)
from models.manager import model_manager
from engines.vision_engine import VisionEngine

router = APIRouter(prefix="/v1/vision", tags=["vision"])


@router.post("/encode", response_model=VisionEncodeResponse)
async def encode_images(request: VisionEncodeRequest):
    """Encode images to feature vectors
    
    This endpoint encodes one or more images into dense feature vectors
    using vision models like DINOv3 or the vision component of multimodal models.
    
    Args:
        request: Vision encoding request containing model name and images
        
    Returns:
        VisionEncodeResponse with embeddings and metadata
        
    Raises:
        HTTPException: If model is not found, not a vision model, or encoding fails
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        logger.info(f"Vision encode request for model: {request.model}")
        # Validate model exists and ensure it's loaded
        await manager.ensure_model_loaded(request.model)
        logger.info(f"Model loaded successfully: {request.model}")
        
        # Get the engine
        engine = manager.get_engine(request.model)
        if not isinstance(engine, VisionEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a vision or multimodal model"
            )
        
        # Encode images
        if len(request.images) == 1:
            # Single image
            features = await engine.encode_image(request.images[0])
            embeddings = [features.tolist()]
        else:
            # Batch encoding for multiple images
            features_list = await engine.encode_batch_images(request.images)
            embeddings = [features.tolist() for features in features_list]
        
        
        # Get dimensions from first embedding
        dimensions = len(embeddings[0]) if embeddings else 0
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Vision encoding completed: model={request.model}, "
            f"images={len(request.images)}, dimensions={dimensions}, "
            f"time={processing_time:.3f}s"
        )
        
        return VisionEncodeResponse(
            model=request.model,
            embeddings=embeddings,
            dimensions=dimensions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vision encoding: {e}")
        raise HTTPException(status_code=500, detail=f"Vision encoding failed: {str(e)}")


@router.post("/similarity")
async def compute_similarity(request: dict):
    """Compute similarity between two images
    
    Args:
        request: JSON request containing model, image1, and image2
        
    Returns:
        Dictionary containing similarity score, original features, and metadata
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        # Extract parameters from request
        model = request.get("model")
        image1 = request.get("image1")
        image2 = request.get("image2")
        
        if not all([model, image1, image2]):
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: model, image1, image2"
            )
        
        # Ensure model is loaded
        await manager.ensure_model_loaded(model)
        
        # Get the engine
        engine = manager.get_engine(model)
        if not isinstance(engine, VisionEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not a vision or multimodal model"
            )
        
        # Encode both images
        features1 = await engine.encode_image(image1)
        features2 = await engine.encode_image(image2)
        
        # Compute similarity
        similarity = engine.compute_similarity(features1, features2)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Similarity computation completed: model={model}, "
            f"similarity={similarity:.4f}, time={processing_time:.3f}s"
        )
        
        return {
            "model": model,
            "similarity": float(similarity),
            "image1_features": features1.tolist(),
            "image2_features": features2.tolist(),
            "feature_dimensions": len(features1),
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in similarity computation: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")


@router.post("/batch_similarity")
async def compute_batch_similarity(
    model: str,
    images: List[str]
):
    """Compute similarity matrix for multiple images
    
    Args:
        model: Vision model name to use
        images: List of images (base64/URL/path)
        
    Returns:
        Similarity matrix and metadata
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        if len(images) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images required")
        
        # Ensure model is loaded
        await manager.ensure_model_loaded(model)
        
        # Get the engine
        engine = manager.get_engine(model)
        if not isinstance(engine, VisionEngine):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not a vision or multimodal model"
            )
        
        # Encode all images
        features_list = await engine.encode_batch_images(images)
        
        # Compute similarity matrix
        import numpy as np
        n = len(features_list)
        similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = engine.compute_similarity(features_list[i], features_list[j])
                    similarity_matrix[i][j] = float(sim)
                    similarity_matrix[j][i] = float(sim)  # Symmetric
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Batch similarity completed: model={model}, "
            f"images={len(images)}, time={processing_time:.3f}s"
        )
        
        return {
            "model": model,
            "similarity_matrix": similarity_matrix,
            "shape": [n, n],
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Batch similarity failed: {str(e)}")


@router.get("/models")
async def list_vision_models():
    """List available vision and multimodal models
    
    Returns:
        List of available vision models with their types and status
    """
    manager = model_manager
    
    try:
        all_models = manager.get_available_models()
        
        # Filter vision and multimodal models
        vision_models = []
        for model_id, model_info in all_models.items():
            if model_info and model_info.type in [ModelType.VISION, ModelType.MULTIMODAL]:
                vision_models.append({
                    "id": model_id,
                    "name": model_info.name,
                    "type": model_info.type.value,
                    "engine": model_info.engine.value,
                    "size_gb": model_info.size_gb,
                    "status": model_info.status.value,
                    "capabilities": {
                        "image_encoding": True,
                        "text_encoding": model_info.type == ModelType.MULTIMODAL,
                        "similarity_computation": True
                    }
                })
        
        return {
            "object": "list",
            "data": vision_models
        }
        
    except Exception as e:
        logger.error(f"Error listing vision models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list vision models: {str(e)}")
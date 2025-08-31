"""Multimodal API endpoints for image-text processing"""

import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from loguru import logger

from models.types import (
    MultimodalEncodeRequest,
    MultimodalEncodeResponse,
    MultimodalMatchRequest,
    MultimodalMatchResponse,
    ModelType
)
from models.manager import model_manager
from engines.vision_engine import VisionEngine

router = APIRouter(prefix="/v1/multimodal", tags=["multimodal"])


@router.post("/encode", response_model=MultimodalEncodeResponse)
async def encode_multimodal(request: MultimodalEncodeRequest):
    """Encode images and/or texts into unified embedding space
    
    This endpoint encodes images and texts using multimodal models like CLIP,
    projecting them into a shared embedding space for cross-modal tasks.
    
    Args:
        request: Multimodal encoding request containing model name, images and/or texts
        
    Returns:
        MultimodalEncodeResponse with separate embeddings for images and texts
        
    Raises:
        HTTPException: If model is not multimodal or encoding fails
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        # Validate inputs
        if not request.images and not request.texts:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'images' or 'texts' must be provided"
            )
        
        # Ensure model is loaded
        await manager.ensure_model_loaded(request.model)
        
        # Get and validate engine
        engine = manager.get_engine(request.model)
        if not isinstance(engine, VisionEngine) or not engine.is_multimodal:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a multimodal model"
            )
        
        image_embeddings = None
        text_embeddings = None
        dimensions = 0
        
        # Encode images if provided
        if request.images:
            if len(request.images) == 1:
                features = await engine.encode_image(request.images[0])
                image_embeddings = [features.tolist()]
            else:
                features_list = await engine.encode_batch_images(request.images)
                image_embeddings = [features.tolist() for features in features_list]
            
            
            dimensions = len(image_embeddings[0]) if image_embeddings else 0
        
        # Encode texts if provided
        if request.texts:
            if len(request.texts) == 1:
                features = await engine.encode_text(request.texts[0])
                text_embeddings = [features.tolist()]
            else:
                features_list = await engine.encode_batch_texts(request.texts)
                text_embeddings = [features.tolist() for features in features_list]
            
            
            # Set dimensions if not set from images
            if dimensions == 0 and text_embeddings:
                dimensions = len(text_embeddings[0])
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Multimodal encoding completed: model={request.model}, "
            f"images={len(request.images) if request.images else 0}, "
            f"texts={len(request.texts) if request.texts else 0}, "
            f"dimensions={dimensions}, time={processing_time:.3f}s"
        )
        
        return MultimodalEncodeResponse(
            model=request.model,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            dimensions=dimensions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multimodal encoding: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal encoding failed: {str(e)}")


@router.post("/match", response_model=MultimodalMatchResponse)
async def match_images_texts(request: MultimodalMatchRequest):
    """Compute similarity matrix between images and texts
    
    This endpoint computes cross-modal similarity scores between all provided
    images and texts, useful for image-text retrieval tasks.
    
    Args:
        request: Matching request containing model name, images and texts
        
    Returns:
        MultimodalMatchResponse with similarity matrix and metadata
        
    Raises:
        HTTPException: If model is not multimodal or matching fails
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        # Validate inputs
        if not request.images or not request.texts:
            raise HTTPException(
                status_code=400,
                detail="Both 'images' and 'texts' must be provided"
            )
        
        # Ensure model is loaded
        await manager.ensure_model_loaded(request.model)
        
        # Get and validate engine
        engine = manager.get_engine(request.model)
        if not isinstance(engine, VisionEngine) or not engine.is_multimodal:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a multimodal model"
            )
        
        # Encode images and texts
        image_features = await engine.encode_batch_images(request.images)
        text_features = await engine.encode_batch_texts(request.texts)
        
        # Compute similarity matrix
        import numpy as np
        
        # Normalize features
        image_features_norm = []
        for feat in image_features:
            norm = np.linalg.norm(feat)
            if norm > 0:
                image_features_norm.append(feat / norm)
            else:
                image_features_norm.append(feat)
        
        text_features_norm = []
        for feat in text_features:
            norm = np.linalg.norm(feat)
            if norm > 0:
                text_features_norm.append(feat / norm)
            else:
                text_features_norm.append(feat)
        
        # Compute cosine similarity matrix
        similarity_matrix = []
        for img_feat in image_features_norm:
            row = []
            for txt_feat in text_features_norm:
                similarity = float(np.dot(img_feat, txt_feat))
                row.append(similarity)
            similarity_matrix.append(row)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Image-text matching completed: model={request.model}, "
            f"images={len(request.images)}, texts={len(request.texts)}, "
            f"time={processing_time:.3f}s"
        )
        
        return MultimodalMatchResponse(
            model=request.model,
            similarity_matrix=similarity_matrix,
            shape=[len(request.images), len(request.texts)]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image-text matching: {e}")
        raise HTTPException(status_code=500, detail=f"Image-text matching failed: {str(e)}")


@router.post("/search")
async def search_cross_modal(
    model: str,
    query: str,
    images: List[str],
    top_k: int = 5
):
    """Search for most relevant images given a text query
    
    Args:
        model: Multimodal model name to use
        query: Text query
        images: List of images to search through
        top_k: Number of top results to return
        
    Returns:
        Ranked list of images with similarity scores
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        # Validate inputs
        if not images:
            raise HTTPException(status_code=400, detail="Images list cannot be empty")
        
        if top_k <= 0 or top_k > len(images):
            top_k = len(images)
        
        # Ensure model is loaded
        await manager.ensure_model_loaded(model)
        
        # Get and validate engine
        engine = manager.get_engine(model)
        if not isinstance(engine, VisionEngine) or not engine.is_multimodal:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not a multimodal model"
            )
        
        # Encode query and images
        query_features = await engine.encode_text(query)
        image_features = await engine.encode_batch_images(images)
        
        # Compute similarities
        import numpy as np
        
        # Normalize features
        query_norm = np.linalg.norm(query_features)
        if query_norm > 0:
            query_features = query_features / query_norm
        
        similarities = []
        for i, img_feat in enumerate(image_features):
            img_norm = np.linalg.norm(img_feat)
            if img_norm > 0:
                img_feat_norm = img_feat / img_norm
                similarity = float(np.dot(query_features, img_feat_norm))
            else:
                similarity = 0.0
            
            similarities.append({
                "index": i,
                "image": images[i],
                "similarity": similarity
            })
        
        # Sort by similarity (descending) and get top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = similarities[:top_k]
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Cross-modal search completed: model={model}, "
            f"query_length={len(query)}, images={len(images)}, "
            f"top_k={top_k}, time={processing_time:.3f}s"
        )
        
        return {
            "model": model,
            "query": query,
            "results": top_results,
            "total_images": len(images),
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cross-modal search: {e}")
        raise HTTPException(status_code=500, detail=f"Cross-modal search failed: {str(e)}")


@router.post("/search_text")
async def search_texts_by_image(
    model: str,
    image: str,
    texts: List[str],
    top_k: int = 5
):
    """Search for most relevant texts given an image query
    
    Args:
        model: Multimodal model name to use
        image: Image query (base64/URL/path)
        texts: List of texts to search through
        top_k: Number of top results to return
        
    Returns:
        Ranked list of texts with similarity scores
    """
    start_time = time.time()
    manager = model_manager
    
    try:
        # Validate inputs
        if not texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        if top_k <= 0 or top_k > len(texts):
            top_k = len(texts)
        
        # Ensure model is loaded
        await manager.ensure_model_loaded(model)
        
        # Get and validate engine
        engine = manager.get_engine(model)
        if not isinstance(engine, VisionEngine) or not engine.is_multimodal:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not a multimodal model"
            )
        
        # Encode image and texts
        image_features = await engine.encode_image(image)
        text_features = await engine.encode_batch_texts(texts)
        
        # Compute similarities
        import numpy as np
        
        # Normalize features
        img_norm = np.linalg.norm(image_features)
        if img_norm > 0:
            image_features = image_features / img_norm
        
        similarities = []
        for i, txt_feat in enumerate(text_features):
            txt_norm = np.linalg.norm(txt_feat)
            if txt_norm > 0:
                txt_feat_norm = txt_feat / txt_norm
                similarity = float(np.dot(image_features, txt_feat_norm))
            else:
                similarity = 0.0
            
            similarities.append({
                "index": i,
                "text": texts[i],
                "similarity": similarity
            })
        
        # Sort by similarity (descending) and get top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = similarities[:top_k]
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Text search by image completed: model={model}, "
            f"texts={len(texts)}, top_k={top_k}, time={processing_time:.3f}s"
        )
        
        return {
            "model": model,
            "results": top_results,
            "total_texts": len(texts),
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text search by image: {e}")
        raise HTTPException(status_code=500, detail=f"Text search by image failed: {str(e)}")


@router.get("/models")
async def list_multimodal_models():
    """List available multimodal models
    
    Returns:
        List of available multimodal models with capabilities
    """
    manager = model_manager
    
    try:
        all_models = manager.get_available_models()
        
        # Filter multimodal models
        multimodal_models = []
        for model_id, model_info in all_models.items():
            if model_info and model_info.type == ModelType.MULTIMODAL:
                multimodal_models.append({
                    "id": model_id,
                    "name": model_info.name,
                    "type": model_info.type.value,
                    "engine": model_info.engine.value,
                    "size_gb": model_info.size_gb,
                    "status": model_info.status.value,
                    "capabilities": {
                        "image_encoding": True,
                        "text_encoding": True,
                        "cross_modal_search": True,
                        "image_text_matching": True
                    }
                })
        
        return {
            "object": "list",
            "data": multimodal_models
        }
        
    except Exception as e:
        logger.error(f"Error listing multimodal models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list multimodal models: {str(e)}")
"""Vision model inference engine for CLIP and DINOv3"""

import os
import base64
import requests
from io import BytesIO
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import asyncio
from loguru import logger

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor
)

from engines.base import InferenceEngine
from models.types import ModelInfo, ModelType, ModelStatus
from app.config import settings


class VisionEngine(InferenceEngine):
    """Vision and multimodal model inference engine"""
    
    def __init__(self, model_info: ModelInfo, use_gpu: bool = True):
        """Initialize vision engine
        
        Args:
            model_info: Model information
            use_gpu: Whether to use GPU if available
        """
        super().__init__(model_info)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.processor = None
        self.model = None
        self.is_multimodal = model_info.type == ModelType.MULTIMODAL
        
        logger.info(f"Initializing VisionEngine for {model_info.id} on {self.device}")
    
    async def load_model(self) -> None:
        """Load vision or multimodal model"""
        try:
            self.model_info.status = ModelStatus.LOADING
            logger.info(f"Loading vision model: {self.model_info.id}")
            
            # Determine model loading strategy based on type
            if self.is_multimodal:
                await self._load_multimodal_model()
            else:
                await self._load_vision_model()
            
            self.model_info.status = ModelStatus.LOADED
            logger.info(f"Successfully loaded model: {self.model_info.id}")
            
        except Exception as e:
            self.model_info.status = ModelStatus.ERROR
            logger.error(f"Failed to load model {self.model_info.id}: {e}")
            raise
    
    async def _load_multimodal_model(self) -> None:
        """Load multimodal model (e.g., CLIP)"""
        model_path = self.model_info.path
        
        # Try to load as CLIP model first
        if 'clip' in self.model_info.id.lower():
            self.processor = CLIPProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = CLIPModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                trust_remote_code=True
            )
        else:
            # Generic multimodal model loading
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                trust_remote_code=True
            )
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    async def _load_vision_model(self) -> None:
        """Load pure vision model (e.g., DINOv3)"""
        model_path = self.model_info.path
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.use_gpu else torch.float32,
            trust_remote_code=True
        )
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    async def unload_model(self) -> None:
        """Unload model and free resources"""
        logger.info(f"Unloading model: {self.model_info.id}")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        self.model_info.status = ModelStatus.UNLOADED
        logger.info(f"Model unloaded: {self.model_info.id}")
    
    def _process_image_input(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Process various image input formats into PIL Image
        
        Args:
            image_input: Image as base64 string, URL, file path, bytes, or PIL Image
            
        Returns:
            PIL Image object
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        
        if isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input)).convert('RGB')
        
        if isinstance(image_input, str):
            # Check if it's a URL
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            
            # Check if it's a data URL
            if image_input.startswith('data:image'):
                # Extract base64 data from data URL
                base64_str = image_input.split(',')[1]
                image_data = base64.b64decode(base64_str)
                return Image.open(BytesIO(image_data)).convert('RGB')
            
            # Check if it's a file path
            if os.path.exists(image_input):
                return Image.open(image_input).convert('RGB')
            
            # Assume it's base64 encoded string
            try:
                image_data = base64.b64decode(image_input)
                return Image.open(BytesIO(image_data)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Unable to process image input: {e}")
        
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    @torch.no_grad()
    async def encode_image(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        """Encode single image to feature vector
        
        Args:
            image_input: Image in various formats
            
        Returns:
            Feature vector as numpy array
        """
        if self.model is None:
            raise RuntimeError(f"Model {self.model_info.id} is not loaded")
        
        # Process image input
        image = self._process_image_input(image_input)
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to device
        if self.use_gpu:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Get features based on model type
        if self.is_multimodal and hasattr(self.model, 'get_image_features'):
            # CLIP-style model
            features = self.model.get_image_features(**inputs)
        else:
            # Generic vision model - use CLS token or pooled output
            outputs = self.model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Use CLS token (first token)
                features = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Unable to extract features from model output")
        
        # Convert to numpy and return
        result = features.detach().cpu().numpy()[0]
        return result
    
    @torch.no_grad()
    async def encode_text(self, text: str) -> np.ndarray:
        """Encode text to feature vector (only for multimodal models)
        
        Args:
            text: Input text
            
        Returns:
            Feature vector as numpy array
        """
        if not self.is_multimodal:
            raise ValueError(f"Model {self.model_info.id} does not support text encoding")
        
        if self.model is None:
            raise RuntimeError(f"Model {self.model_info.id} is not loaded")
        
        # Prepare text inputs
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        if self.use_gpu:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Get text features
        if hasattr(self.model, 'get_text_features'):
            # CLIP-style model
            features = self.model.get_text_features(**inputs)
        else:
            # Generic multimodal model
            outputs = self.model(**inputs)
            if hasattr(outputs, 'text_embeds'):
                features = outputs.text_embeds
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            else:
                raise ValueError(f"Unable to extract text features from model output")
        
        # Convert to numpy and return
        result = features.detach().cpu().numpy()[0]
        return result
    
    async def encode_batch_images(self, image_inputs: List[Union[str, bytes, Image.Image]]) -> List[np.ndarray]:
        """Encode multiple images in batch
        
        Args:
            image_inputs: List of images in various formats
            
        Returns:
            List of feature vectors
        """
        if self.model is None:
            raise RuntimeError(f"Model {self.model_info.id} is not loaded")
        
        # Process all images
        images = [self._process_image_input(img) for img in image_inputs]
        
        # Batch process
        batch_size = min(32, len(images))  # Limit batch size for memory
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Prepare batch inputs
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            
            # Move to device
            if self.use_gpu:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get features
                if self.is_multimodal and hasattr(self.model, 'get_image_features'):
                    features = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state[:, 0, :]
                    else:
                        raise ValueError(f"Unable to extract features from model output")
                
                # Convert to numpy
                batch_features = features.detach().cpu().numpy()
                results.extend([feat for feat in batch_features])
        
        return results
    
    async def encode_batch_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts in batch (only for multimodal models)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of feature vectors
        """
        if not self.is_multimodal:
            raise ValueError(f"Model {self.model_info.id} does not support text encoding")
        
        if self.model is None:
            raise RuntimeError(f"Model {self.model_info.id} is not loaded")
        
        # Batch process
        batch_size = min(32, len(texts))
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Prepare batch inputs
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            
            # Move to device
            if self.use_gpu:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get text features
                if hasattr(self.model, 'get_text_features'):
                    features = self.model.get_text_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'text_embeds'):
                        features = outputs.text_embeds
                    elif hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    else:
                        raise ValueError(f"Unable to extract text features from model output")
                
                # Convert to numpy
                batch_features = features.detach().cpu().numpy()
                results.extend([feat for feat in batch_features])
        
        return results
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        features1_norm = features1 / norm1
        features2_norm = features2 / norm2
        
        # Compute cosine similarity
        similarity = np.dot(features1_norm, features2_norm)
        
        return float(similarity)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector to unit length
        
        Args:
            features: Feature vector
            
        Returns:
            Normalized feature vector
        """
        norm = np.linalg.norm(features)
        if norm == 0:
            return features
        return features / norm
    
    # Required abstract methods implementation
    async def generate_chat_completion(self, request) -> None:
        """Vision models don't support chat completion"""
        raise NotImplementedError("Vision models do not support chat completion")
    
    async def generate_embeddings(self, request) -> None:
        """Vision models don't support text embeddings via this interface"""
        raise NotImplementedError("Vision models use specialized vision APIs")
    
    async def generate_rerank(self, request) -> None:
        """Vision models don't support reranking"""
        raise NotImplementedError("Vision models do not support reranking")
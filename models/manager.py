"""Model manager for loading and switching models"""

import asyncio
from typing import Dict, Optional, Any
from datetime import datetime
from loguru import logger

from app.config import settings
from models.types import ModelInfo, ModelType, EngineType, ModelStatus
from models.discovery import ModelDiscovery
from engines.base import InferenceEngine


class ModelManager:
    """Manages model loading, switching, and resource allocation"""
    
    def __init__(self):
        self.discovery = ModelDiscovery(settings.models_dir)
        self.engines: Dict[str, InferenceEngine] = {}
        self.current_models: Dict[ModelType, Optional[str]] = {
            ModelType.CHAT: None,
            ModelType.EMBEDDING: None,
            ModelType.RERANKER: None
        }
        self._lock = asyncio.Lock()
        self._available_models: Dict[str, ModelInfo] = {}
        self._refresh_models()
    
    def _refresh_models(self) -> None:
        """Refresh the list of available models"""
        models = self.discovery.scan_models()
        self._available_models = {model.id: model for model in models}
        logger.info(f"Found {len(models)} models in {settings.models_dir}")
        for model in models:
            logger.info(f"  - {model.id}: {model.type} ({model.engine}, {model.size_gb}GB)")
    
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get all available models"""
        return self._available_models.copy()
    
    def get_current_model(self, model_type: ModelType) -> Optional[str]:
        """Get currently loaded model for a type"""
        return self.current_models.get(model_type)
    
    def _determine_deployment_strategy(self, model_info: ModelInfo) -> dict:
        """Determine optimal deployment strategy for a model"""
        strategy = {
            "device": "cuda",
            "use_cpu_offload": False,
            "reason": "default GPU deployment"
        }
        
        if not settings.enable_gpu_memory_offload:
            return strategy
        
        # Large models that exceed threshold need GPU+Memory offloading
        if model_info.size_gb > settings.gpu_memory_offload_threshold_gb:
            strategy.update({
                "device": "cuda",
                "use_cpu_offload": True,
                "reason": f"Large model ({model_info.size_gb}GB) using GPU+Memory offloading"
            })
            logger.info(f"Model {model_info.id}: {strategy['reason']}")
            return strategy
        
        # Small models can use pure GPU (since we unload between switches)
        strategy.update({
            "device": "cuda", 
            "use_cpu_offload": False,
            "reason": f"Small model ({model_info.size_gb}GB) using pure GPU"
        })
        logger.info(f"Model {model_info.id}: {strategy['reason']}")
        return strategy
    
    async def ensure_model_loaded(self, model_name: str) -> ModelInfo:
        """Ensure a model is loaded, switching if necessary"""
        async with self._lock:
            # Check if model exists
            if model_name not in self._available_models:
                # Try refreshing model list
                self._refresh_models()
                if model_name not in self._available_models:
                    raise ValueError(f"Model '{model_name}' not found")
            
            model_info = self._available_models[model_name]
            model_type = model_info.type
            
            # Check if model is already loaded
            current_model = self.current_models.get(model_type)
            if current_model == model_name:
                if model_name in self.engines and self.engines[model_name].is_loaded:
                    logger.info(f"Model '{model_name}' already loaded")
                    return model_info
            
            # Need to switch models
            logger.info(f"Switching from '{current_model}' to '{model_name}'")
            
            # Unload current model if auto_unload is enabled
            if current_model and settings.auto_unload_models:
                await self._unload_model(current_model)
            
            # Load new model
            await self._load_model(model_info)
            
            # Update current model
            self.current_models[model_type] = model_name
            
            return model_info
    
    async def _load_model(self, model_info: ModelInfo) -> None:
        """Load a model into memory"""
        try:
            model_info.status = ModelStatus.LOADING
            logger.info(f"Loading model '{model_info.id}' with {model_info.engine} engine...")
            
            # Create appropriate engine
            engine = self._create_engine(model_info)
            
            # Load the model
            await engine.load_model()
            
            # Store engine
            self.engines[model_info.id] = engine
            
            # Update model status
            model_info.status = ModelStatus.LOADED
            model_info.loaded_at = datetime.now()
            
            # Log memory usage
            memory_info = engine.get_memory_usage()
            logger.info(f"Model '{model_info.id}' loaded successfully")
            logger.info(f"GPU Memory: {memory_info['gpu_memory_allocated']:.2f}GB allocated, "
                       f"{memory_info['gpu_memory_free']:.2f}GB free")
            
        except Exception as e:
            model_info.status = ModelStatus.ERROR
            logger.error(f"Failed to load model '{model_info.id}': {e}")
            raise
    
    async def _unload_model(self, model_name: str) -> None:
        """Unload a model from memory"""
        if model_name not in self.engines:
            return
        
        try:
            logger.info(f"Unloading model '{model_name}'...")
            engine = self.engines[model_name]
            
            # Unload the model
            await engine.unload_model()
            
            # Clear cache
            engine.clear_cache()
            
            # Remove engine
            del self.engines[model_name]
            
            # Update model status
            if model_name in self._available_models:
                self._available_models[model_name].status = ModelStatus.UNLOADED
                self._available_models[model_name].loaded_at = None
            
            # Clear from current models
            for model_type, current in self.current_models.items():
                if current == model_name:
                    self.current_models[model_type] = None
            
            logger.info(f"Model '{model_name}' unloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to unload model '{model_name}': {e}")
            raise
    
    def _create_engine(self, model_info: ModelInfo) -> InferenceEngine:
        """Create appropriate inference engine for model"""
        # Determine optimal deployment strategy
        strategy = self._determine_deployment_strategy(model_info)
        
        if model_info.engine == EngineType.VLLM:
            from engines.vllm_engine import VLLMEngine
            # For large models, vLLM can use CPU offloading
            return VLLMEngine(model_info, use_cpu_offload=strategy["use_cpu_offload"])
        elif model_info.engine == EngineType.TRANSFORMERS:
            from engines.transformers_engine import TransformersEngine
            return TransformersEngine(model_info, device=strategy["device"])
        else:
            raise ValueError(f"Unknown engine type: {model_info.engine}")
    
    def get_engine(self, model_name: str) -> Optional[InferenceEngine]:
        """Get engine for a loaded model"""
        return self.engines.get(model_name)
    
    async def cleanup(self) -> None:
        """Clean up all loaded models"""
        logger.info("Cleaning up model manager...")
        
        # Unload all models
        model_names = list(self.engines.keys())
        for model_name in model_names:
            try:
                await self._unload_model(model_name)
            except Exception as e:
                logger.error(f"Error unloading model '{model_name}': {e}")
        
        logger.info("Model manager cleanup complete")


# Global model manager instance
model_manager = ModelManager()
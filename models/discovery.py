"""Model discovery and identification"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from models.types import ModelInfo, ModelType, EngineType, ModelStatus


class ModelDiscovery:
    """Discover and identify models in the models directory"""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {models_dir}")
            self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_models(self) -> List[ModelInfo]:
        """Scan models directory and return list of available models"""
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return models
        
        # Scan each subdirectory in models_dir
        for model_path in self.models_dir.iterdir():
            if not model_path.is_dir():
                continue
            
            try:
                model_info = self._identify_model(model_path)
                if model_info:
                    models.append(model_info)
                    logger.info(f"Discovered model: {model_info.name} ({model_info.type})")
            except Exception as e:
                logger.error(f"Error identifying model at {model_path}: {e}")
        
        return models
    
    def _identify_model(self, model_path: Path) -> Optional[ModelInfo]:
        """Identify model type and properties from directory"""
        model_id = model_path.name
        
        # Read config.json if exists
        config = self._read_model_config(model_path)
        
        # Determine model type from directory name
        model_type = self._determine_model_type(model_id, config)
        
        # Determine inference engine
        engine = self._determine_engine(model_type, model_id, config)
        
        # Calculate model size
        size_gb = self._calculate_model_size(model_path)
        
        # Extract model name
        model_name = config.get("model_type", model_id) if config else model_id
        
        return ModelInfo(
            id=model_id,
            name=model_name,
            type=model_type,
            engine=engine,
            path=str(model_path),
            size_gb=size_gb,
            config=config,
            status=ModelStatus.UNLOADED
        )
    
    def _read_model_config(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """Read config.json from model directory"""
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read config.json at {config_path}: {e}")
            return None
    
    def _determine_model_type(self, model_id: str, config: Optional[Dict]) -> ModelType:
        """Determine model type from name and config"""
        model_id_lower = model_id.lower()
        
        # Check directory name patterns
        if any(keyword in model_id_lower for keyword in ['instruct', 'chat', 'conversation']):
            return ModelType.CHAT
        elif 'embedding' in model_id_lower:
            return ModelType.EMBEDDING
        elif any(keyword in model_id_lower for keyword in ['rerank', 'reranker', 'cross-encoder']):
            return ModelType.RERANKER
        
        # Check config for model type hints
        if config:
            # Check architectures field
            architectures = config.get("architectures", [])
            if architectures:
                arch_str = " ".join(architectures).lower()
                if any(keyword in arch_str for keyword in ['causal', 'gpt', 'llama', 'qwen']):
                    return ModelType.CHAT
                elif 'bert' in arch_str and 'embedding' in model_id_lower:
                    return ModelType.EMBEDDING
                elif 'cross' in arch_str or 'rerank' in arch_str:
                    return ModelType.RERANKER
            
            # Check task-specific fields
            if config.get("task_type") == "feature-extraction":
                return ModelType.EMBEDDING
            
            # Check for sentence_transformers config
            modules_path = model_path / "modules.json"
            if modules_path.exists():
                return ModelType.EMBEDDING
        
        # Default to CHAT if uncertain (most common case)
        logger.warning(f"Could not determine type for model {model_id}, defaulting to CHAT")
        return ModelType.CHAT
    
    def _determine_engine(self, model_type: ModelType, model_id: str, config: Optional[Dict]) -> EngineType:
        """Determine which inference engine to use"""
        # vLLM is optimal for chat models
        if model_type == ModelType.CHAT:
            return EngineType.VLLM
        
        # Transformers for embedding and reranker models
        # vLLM doesn't support these model types well
        if model_type in [ModelType.EMBEDDING, ModelType.RERANKER]:
            return EngineType.TRANSFORMERS
        
        # Default to transformers for unknown types
        return EngineType.TRANSFORMERS
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate total size of model files in GB"""
        total_size = 0
        
        # Common model file patterns
        patterns = [
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.h5",
            "*.msgpack"
        ]
        
        for pattern in patterns:
            for file_path in model_path.glob(pattern):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        # Check for sharded models
        for file_path in model_path.glob("*.safetensors.index.json"):
            # This is an index file for sharded models
            # Read it to get actual shard files
            try:
                with open(file_path, 'r') as f:
                    index_data = json.load(f)
                    weight_map = index_data.get("weight_map", {})
                    # Get unique shard files
                    shard_files = set(weight_map.values())
                    for shard_file in shard_files:
                        shard_path = model_path / shard_file
                        if shard_path.exists():
                            total_size += shard_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to read index file {file_path}: {e}")
        
        # Convert to GB
        return round(total_size / (1024 ** 3), 2)
    
    def find_model(self, model_name: str) -> Optional[ModelInfo]:
        """Find a specific model by name or ID"""
        models = self.scan_models()
        
        for model in models:
            if model.id == model_name or model.name == model_name:
                return model
        
        # Try case-insensitive match
        model_name_lower = model_name.lower()
        for model in models:
            if model.id.lower() == model_name_lower or model.name.lower() == model_name_lower:
                return model
        
        return None
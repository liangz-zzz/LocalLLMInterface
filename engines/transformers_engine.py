"""Transformers inference engine for embedding and reranker models"""

import uuid
import time
import torch
import numpy as np
from typing import List, Optional, Union, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModel, AutoTokenizer, AutoConfig

from app.config import settings
from engines.base import InferenceEngine
from models.types import (
    ModelInfo,
    ModelType,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
    ChatCompletionRequest,
    ChatCompletionResponse
)


class TransformersEngine(InferenceEngine):
    """Transformers-based inference engine for embedding and reranker models"""
    
    def __init__(self, model_info: ModelInfo, device: str = None):
        super().__init__(model_info)
        # Use provided device or fallback to settings
        device = device or settings.transformers_device
        self.device = torch.device(device)
        self.torch_dtype = getattr(torch, settings.transformers_torch_dtype) if settings.transformers_torch_dtype != "auto" else None
        logger.info(f"TransformersEngine initialized for {model_info.id} on device: {self.device}")
    
    async def load_model(self) -> None:
        """Load model with transformers/sentence-transformers"""
        try:
            logger.info(f"Loading transformers model from {self.model_info.path}")
            
            if self.model_info.type == ModelType.EMBEDDING:
                await self._load_embedding_model()
            elif self.model_info.type == ModelType.RERANKER:
                await self._load_reranker_model()
            else:
                raise ValueError(f"Unsupported model type for transformers engine: {self.model_info.type}")
            
            self.is_loaded = True
            logger.info(f"Transformers model '{self.model_info.id}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load transformers model '{self.model_info.id}': {e}")
            self.is_loaded = False
            raise
    
    async def _load_embedding_model(self) -> None:
        """Load embedding model"""
        try:
            # Try sentence-transformers first (more optimized for embeddings)
            self.model = SentenceTransformer(
                self.model_info.path,
                device=self.device,
                trust_remote_code=True
            )
            logger.info(f"Loaded as SentenceTransformer model")
            
        except Exception as e:
            logger.warning(f"Failed to load as SentenceTransformer: {e}")
            
            # Fallback to raw transformers
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_info.path,
                    trust_remote_code=True
                )
                
                self.model = AutoModel.from_pretrained(
                    self.model_info.path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    device_map="auto"
                )
                
                self.model.eval()
                logger.info(f"Loaded as raw transformers model")
                
            except Exception as e2:
                logger.error(f"Failed to load with transformers: {e2}")
                raise e2
    
    async def _load_reranker_model(self) -> None:
        """Load reranker model"""
        try:
            # Try CrossEncoder first (optimized for reranking)
            self.model = CrossEncoder(
                self.model_info.path,
                device=self.device,
                trust_remote_code=True
            )
            logger.info(f"Loaded as CrossEncoder model")
            
        except Exception as e:
            logger.warning(f"Failed to load as CrossEncoder: {e}")
            
            # Fallback to sentence-transformers for similarity-based reranking
            try:
                self.model = SentenceTransformer(
                    self.model_info.path,
                    device=self.device,
                    trust_remote_code=True
                )
                logger.info(f"Loaded as SentenceTransformer for similarity reranking")
                
            except Exception as e2:
                logger.error(f"Failed to load reranker model: {e2}")
                raise e2
    
    async def unload_model(self) -> None:
        """Unload transformers model"""
        try:
            if self.model:
                # Move model to CPU and delete reference
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')
                
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            self.clear_cache()
            self.is_loaded = False
            logger.info(f"Transformers model '{self.model_info.id}' unloaded")
            
        except Exception as e:
            logger.error(f"Error unloading transformers model: {e}")
            raise
    
    async def generate_embeddings(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Generate embeddings using transformers"""
        if not self.is_loaded or not self.model:
            raise RuntimeError(f"Model '{self.model_info.id}' is not loaded")
        
        if self.model_info.type != ModelType.EMBEDDING:
            raise ValueError(f"Model '{self.model_info.id}' is not an embedding model")
        
        try:
            texts = list(request.input)
            logger.debug(f"Generating embeddings for {len(texts)} texts")

            prepared_texts, encode_kwargs = self._prepare_embedding_inputs(
                texts,
                request.prompt,
                request.prompt_name
            )

            # Generate embeddings
            if isinstance(self.model, SentenceTransformer):
                # Use sentence-transformers
                embeddings = self.model.encode(
                    prepared_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=32,
                    **encode_kwargs
                )
            else:
                # Use raw transformers
                embeddings = await self._generate_raw_embeddings(prepared_texts)

            # Convert to list format
            embedding_data = []
            for i, embedding in enumerate(embeddings):
                embedding_data.append(EmbeddingData(
                    index=i,
                    embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                ))
            
            # Calculate approximate token usage
            total_tokens = sum(len(text.split()) for text in prepared_texts)  # Rough approximation
            
            return EmbeddingResponse(
                data=embedding_data,
                model=request.model,
                usage=Usage(
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens
                )
            )
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def _generate_raw_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using raw transformers model"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get hidden states
                outputs = self.model(**inputs)
                
                # Mean pooling
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                embeddings.append(embedding.cpu().numpy()[0])
        
        return np.array(embeddings)

    def _prepare_embedding_inputs(
        self,
        texts: List[str],
        prompt: Optional[Union[str, List[str]]],
        prompt_name: Optional[str]
    ) -> Tuple[List[str], dict]:
        """Apply instruction-style prompts based on model capabilities"""
        if isinstance(self.model, SentenceTransformer):
            encode_kwargs = {}
            if prompt is not None:
                encode_kwargs["prompt"] = prompt
            if prompt_name is not None:
                encode_kwargs["prompt_name"] = prompt_name
            return texts, encode_kwargs

        if prompt is None:
            return texts, {}

        prompts: List[str]
        if isinstance(prompt, str):
            prompts = [prompt] * len(texts)
        else:
            prompts = list(prompt)

        paired = min(len(prompts), len(texts))
        if paired == 0:
            return texts, {}

        prepared = []
        for idx in range(paired):
            prepared.append(f"Instruct: {prompts[idx]}\nQuery: {texts[idx]}")

        if len(texts) > paired:
            prepared.extend(texts[paired:])

        return prepared, {}
    
    async def rerank_documents(
        self,
        request: RerankRequest
    ) -> RerankResponse:
        """Rerank documents using transformers"""
        if not self.is_loaded or not self.model:
            raise RuntimeError(f"Model '{self.model_info.id}' is not loaded")
        
        if self.model_info.type != ModelType.RERANKER:
            raise ValueError(f"Model '{self.model_info.id}' is not a reranker model")
        
        try:
            query = request.query
            documents = request.documents
            
            logger.debug(f"Reranking {len(documents)} documents for query: {query[:100]}...")
            
            if isinstance(self.model, CrossEncoder):
                # Use CrossEncoder for direct scoring
                # Process documents one by one to avoid padding token issues
                scores = []
                for doc in documents:
                    pair = [(query, doc)]
                    score = self.model.predict(pair)[0]
                    scores.append(score)
                
            else:
                # Use SentenceTransformer for similarity-based reranking
                query_embedding = self.model.encode([query], normalize_embeddings=True)
                doc_embeddings = self.model.encode(documents, normalize_embeddings=True)
                
                # Calculate cosine similarity
                scores = np.dot(query_embedding, doc_embeddings.T)[0]
            
            # Create results with original indices
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append(RerankResult(
                    index=i,
                    score=float(score),
                    document=doc
                ))
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply top_k if specified
            if request.top_k:
                results = results[:request.top_k]
            
            return RerankResponse(
                model=request.model,
                results=results
            )
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise
    
    async def generate_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Chat completion not supported by transformers engine"""
        raise NotImplementedError("Chat completion should use vLLM engine")

"""vLLM inference engine for high-performance chat models"""

import asyncio
import uuid
import time
import json
from typing import List, AsyncIterator, Dict, Any, Optional
from loguru import logger

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from app.config import settings
from engines.base import InferenceEngine
from models.types import (
    ModelInfo,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage
)


class VLLMEngine(InferenceEngine):
    """vLLM-based inference engine for chat models"""
    
    def __init__(self, model_info: ModelInfo, use_cpu_offload: bool = False):
        super().__init__(model_info)
        self.engine: Optional[AsyncLLMEngine] = None
        self.sampling_params = None
        self.use_cpu_offload = use_cpu_offload
        logger.info(f"VLLMEngine initialized for {model_info.id}, CPU offload: {use_cpu_offload}")
    
    async def load_model(self) -> None:
        """Load model with vLLM"""
        try:
            logger.info(f"Loading vLLM model from {self.model_info.path}")
            
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_info.path,
                tensor_parallel_size=settings.vllm_tensor_parallel_size,
                dtype=settings.vllm_dtype,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                max_model_len=settings.vllm_max_model_len,
                trust_remote_code=True,
                enforce_eager=True,  # Disable CUDA graphs for more flexibility
                # Enable quantization if model is AWQ
                quantization="awq" if "awq" in self.model_info.id.lower() else None,
                # CPU offloading for large models
                cpu_offload_gb=16 if self.use_cpu_offload else 0,  # Offload 16GB to CPU if enabled
                load_format="auto"
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Set default sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=1024,
                stop=None
            )
            
            self.is_loaded = True
            logger.info(f"vLLM model '{self.model_info.id}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vLLM model '{self.model_info.id}': {e}")
            self.is_loaded = False
            raise
    
    async def unload_model(self) -> None:
        """Unload vLLM model"""
        if self.engine:
            try:
                # vLLM doesn't have explicit unload, we rely on garbage collection
                # and CUDA cache clearing
                self.engine = None
                self.sampling_params = None
                self.clear_cache()
                self.is_loaded = False
                logger.info(f"vLLM model '{self.model_info.id}' unloaded")
            except Exception as e:
                logger.error(f"Error unloading vLLM model: {e}")
                raise
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert OpenAI messages to prompt string"""
        # Simple conversion for now - can be enhanced with chat templates
        prompt_parts = []
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _create_sampling_params(self, request: ChatCompletionRequest) -> SamplingParams:
        """Create sampling parameters from request"""
        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            stop=request.stop,
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            skip_special_tokens=True
        )
    
    async def generate_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate chat completion using vLLM"""
        if not self.is_loaded or not self.engine:
            raise RuntimeError(f"Model '{self.model_info.id}' is not loaded")
        
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(request.messages)
            
            # Create sampling parameters
            sampling_params = self._create_sampling_params(request)
            
            # Generate completion
            request_id = str(uuid.uuid4())
            
            logger.debug(f"Generating completion for prompt: {prompt[:100]}...")
            
            # Use async generation
            results = []
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            ):
                results.append(output)
            
            # Get final result
            final_output = results[-1] if results else None
            
            if not final_output or not final_output.outputs:
                raise RuntimeError("No output generated")
            
            # Extract generated text
            generated_text = final_output.outputs[0].text.strip()
            
            # Calculate token usage (approximate)
            # vLLM provides actual token counts
            prompt_tokens = len(final_output.prompt_token_ids) if hasattr(final_output, 'prompt_token_ids') else 0
            completion_tokens = len(final_output.outputs[0].token_ids) if hasattr(final_output.outputs[0], 'token_ids') else 0
            
            # Create response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason=final_output.outputs[0].finish_reason if hasattr(final_output.outputs[0], 'finish_reason') else "stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    async def generate_chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Generate streaming chat completion"""
        if not self.is_loaded or not self.engine:
            raise RuntimeError(f"Model '{self.model_info.id}' is not loaded")
        
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(request.messages)
            
            # Create sampling parameters
            sampling_params = self._create_sampling_params(request)
            
            # Generate streaming completion
            request_id = str(uuid.uuid4())
            
            logger.debug(f"Starting streaming completion for: {prompt[:100]}...")
            
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            ):
                if output.outputs:
                    # Format as SSE (Server-Sent Events)
                    generated_text = output.outputs[0].text
                    
                    # Create streaming response chunk
                    chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": generated_text},
                            "finish_reason": None
                        }]
                    }
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming completion: {e}")
            raise
"""Chat completions API - OpenAI compatible"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from models.manager import model_manager
from models.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelType
)


router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion - OpenAI compatible endpoint"""
    try:
        logger.info(f"Chat completion request for model: {request.model}")
        
        # Ensure model is loaded (automatic switching)
        model_info = await model_manager.ensure_model_loaded(request.model)
        
        # Verify it's a chat model
        if model_info.type != ModelType.CHAT:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not a chat model (type: {model_info.type})"
            )
        
        # Get engine
        engine = model_manager.get_engine(request.model)
        if not engine:
            raise HTTPException(
                status_code=500,
                detail=f"Engine not found for model '{request.model}'"
            )
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                engine.generate_chat_completion_stream(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # Generate completion
        response = await engine.generate_chat_completion(request)
        
        logger.info(f"Chat completion generated successfully for model: {request.model}")
        return response
        
    except ValueError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating completion: {e}")


@router.post("/completions")
async def create_completion(request: Dict[str, Any]):
    """Legacy completions endpoint for backward compatibility"""
    try:
        # Convert to chat format
        prompt = request.get("prompt", "")
        model = request.get("model", "")
        
        if not model:
            raise HTTPException(status_code=400, detail="Model is required")
        
        # Convert prompt to messages format
        chat_request = ChatCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens"),
            top_p=request.get("top_p", 1.0),
            stop=request.get("stop"),
            stream=request.get("stream", False)
        )
        
        # Use chat completion
        if chat_request.stream:
            return StreamingResponse(
                _convert_chat_stream_to_completion_stream(
                    model_manager.get_engine(model).generate_chat_completion_stream(chat_request),
                    model
                ),
                media_type="text/event-stream"
            )
        else:
            chat_response = await create_chat_completion(chat_request)
            
            # Convert to completion format
            return {
                "id": chat_response.id.replace("chatcmpl-", "cmpl-"),
                "object": "text_completion",
                "created": chat_response.created,
                "model": chat_response.model,
                "choices": [{
                    "text": chat_response.choices[0].message.content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": chat_response.choices[0].finish_reason
                }],
                "usage": chat_response.usage.dict() if chat_response.usage else None
            }
        
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating completion: {e}")


async def _convert_chat_stream_to_completion_stream(chat_stream, model: str):
    """Convert chat completion stream to completion stream format"""
    import json
    
    async for chunk in chat_stream:
        if chunk.startswith("data: "):
            chunk_data = chunk[6:].strip()
            if chunk_data == "[DONE]":
                yield "data: [DONE]\n\n"
                break
            
            try:
                chat_chunk = json.loads(chunk_data)
                # Convert to completion format
                completion_chunk = {
                    "id": chat_chunk["id"].replace("chatcmpl-", "cmpl-"),
                    "object": "text_completion",
                    "created": chat_chunk["created"],
                    "model": model,
                    "choices": [{
                        "text": chat_chunk["choices"][0]["delta"].get("content", ""),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": chat_chunk["choices"][0].get("finish_reason")
                    }]
                }
                yield f"data: {json.dumps(completion_chunk)}\n\n"
            except json.JSONDecodeError:
                continue
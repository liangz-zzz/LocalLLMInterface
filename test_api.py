#!/usr/bin/env python3
"""Test script for Local LLM Interface API"""

import asyncio
import httpx
import json
from typing import Dict, Any


BASE_URL = "http://localhost:15530"


async def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available models: {data['models']['available']}")
            print(f"Loaded models: {data['models']['loaded']}")
            print(f"GPU memory used: {data['memory']['gpu_memory_used_gb']}GB")
        else:
            print(f"Error: {response.text}")


async def test_list_models():
    """Test models list endpoint"""
    print("\n📋 Testing models list...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/models")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['data'])} models:")
            for model in data["data"]:
                print(f"  - {model['id']}: {model['type']} ({model['engine']}, {model['size_gb']}GB)")
            return data["data"]
        else:
            print(f"Error: {response.text}")
            return []


async def test_chat_completion(models: list):
    """Test chat completion endpoint"""
    print("\n💬 Testing chat completion...")
    
    # Find a chat model
    chat_models = [m for m in models if m["type"] == "chat"]
    if not chat_models:
        print("No chat models found, skipping test")
        return
    
    model_name = chat_models[0]["id"]
    print(f"Using model: {model_name}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello! Please respond with a short greeting."}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = await client.post(f"{BASE_URL}/v1/chat/completions", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['choices'][0]['message']['content']}")
            print(f"Usage: {data['usage']}")
        else:
            print(f"Error: {response.text}")


async def test_embeddings(models: list):
    """Test embeddings endpoint"""
    print("\n🔢 Testing embeddings...")
    
    # Find an embedding model
    embedding_models = [m for m in models if m["type"] == "embedding"]
    if not embedding_models:
        print("No embedding models found, skipping test")
        return
    
    model_name = embedding_models[0]["id"]
    print(f"Using model: {model_name}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        request_data = {
            "model": model_name,
            "input": ["Hello world", "Test embedding"]
        }
        
        response = await client.post(f"{BASE_URL}/v1/embeddings", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Generated {len(data['data'])} embeddings")
            print(f"Embedding dimension: {len(data['data'][0]['embedding'])}")
            print(f"Usage: {data['usage']}")
        else:
            print(f"Error: {response.text}")


async def test_rerank(models: list):
    """Test rerank endpoint"""
    print("\n🔄 Testing rerank...")
    
    # Find a reranker model
    rerank_models = [m for m in models if m["type"] == "reranker"]
    if not rerank_models:
        print("No reranker models found, skipping test")
        return
    
    model_name = rerank_models[0]["id"]
    print(f"Using model: {model_name}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        request_data = {
            "model": model_name,
            "query": "artificial intelligence",
            "documents": [
                "AI is a branch of computer science",
                "Machine learning enables AI systems", 
                "Deep learning is a subset of ML"
            ],
            "top_k": 2
        }
        
        response = await client.post(f"{BASE_URL}/v1/rerank", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Reranked {len(data['results'])} documents:")
            for i, result in enumerate(data['results']):
                print(f"  {i+1}. Score: {result['score']:.4f} - {result['document'][:50]}...")
        else:
            print(f"Error: {response.text}")


async def main():
    """Run all tests"""
    print("🚀 Local LLM Interface API Test")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        await test_health()
        
        # Get available models
        models = await test_list_models()
        
        # Test each endpoint type
        await test_chat_completion(models)
        await test_embeddings(models)
        await test_rerank(models)
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
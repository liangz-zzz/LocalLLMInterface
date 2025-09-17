#!/usr/bin/env python3
"""Test script for Local LLM Interface API"""

import asyncio
import httpx
import json
import base64
from typing import Dict, Any
from PIL import Image
from io import BytesIO


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


def load_test_image() -> str:
    """Load real test image and return as base64"""
    import os
    
    # Path to test image
    image_path = os.path.join(os.path.dirname(__file__), "test_data", "6309898_xxl.jpg")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    # Load and convert to base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def test_vision_models():
    """Test vision models endpoint"""
    print("\n👁️  Testing vision models...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/vision/models")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Vision models available: {len(data['data'])}")
            for model in data['data']:
                capabilities = model.get('capabilities', {})
                caps = ', '.join([k for k, v in capabilities.items() if v])
                print(f"  - {model['id']}: {model['type']} ({caps})")
        else:
            print(f"Error: {response.text}")


async def test_vision_encoding(models: list):
    """Test vision encoding endpoint"""
    # Find a vision model
    vision_models = [m for m in models if m.get('type') in ['vision', 'multimodal']]
    if not vision_models:
        print("\n⚠️  No vision models available for testing")
        return
    
    model = vision_models[0]
    print(f"\n🖼️  Testing vision encoding with {model['id']}...")
    
    test_image_b64 = load_test_image()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": model['id'],
            "images": [test_image_b64]
        }
        
        response = await client.post(f"{BASE_URL}/v1/vision/encode", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Encoded {len(data['embeddings'])} images")
            print(f"Feature dimensions: {data['dimensions']}")
            print(f"First few features: {data['embeddings'][0][:5]}")
        else:
            print(f"Error: {response.text}")


async def test_multimodal_features(models: list):
    """Test multimodal features"""
    # Find a multimodal model (CLIP)
    multimodal_models = [m for m in models if m['type'] == 'multimodal']
    if not multimodal_models:
        print("\n⚠️  No multimodal models available for testing")
        return
    
    model = multimodal_models[0]
    print(f"\n🔄 Testing multimodal features with {model['id']}...")
    
    test_image_b64 = load_test_image()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test multimodal encoding
        payload = {
            "model": model['id'],
            "images": [test_image_b64],
            "texts": ["a colorful gradient image", "a test image"]
        }
        
        response = await client.post(f"{BASE_URL}/v1/multimodal/encode", json=payload)
        print(f"Multimodal encode status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('image_embeddings'):
                print(f"Image embeddings: {len(data['image_embeddings'])}")
            if data.get('text_embeddings'):
                print(f"Text embeddings: {len(data['text_embeddings'])}")
            print(f"Feature dimensions: {data['dimensions']}")
        else:
            print(f"Multimodal encode error: {response.text}")
        
        # Test image-text matching
        match_payload = {
            "model": model['id'],
            "images": [test_image_b64],
            "texts": ["a gradient pattern", "a solid color", "random noise"]
        }
        
        response = await client.post(f"{BASE_URL}/v1/multimodal/match", json=match_payload)
        print(f"Image-text match status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Similarity matrix shape: {data['shape']}")
            similarities = data['similarity_matrix'][0]  # First image similarities
            best_match_idx = similarities.index(max(similarities))
            best_match_text = match_payload['texts'][best_match_idx]
            print(f"Best match: '{best_match_text}' (similarity: {similarities[best_match_idx]:.4f})")
        else:
            print(f"Image-text match error: {response.text}")


async def test_similarity_with_features():
    """Test similarity API with feature output"""
    print("\n🔍 Testing similarity computation with features...")
    
    # Load test image
    test_image_b64 = load_test_image()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": "clip-vit-base-patch32",
            "image1": test_image_b64,
            "image2": test_image_b64  # Same image should have similarity ~1.0
        }
        
        response = await client.post(f"{BASE_URL}/v1/vision/similarity", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Model: {data['model']}")
            print(f"Similarity: {data['similarity']:.6f}")
            print(f"Feature dimensions: {data['feature_dimensions']}")
            print(f"Image1 features (first 5): {data['image1_features'][:5]}")
            print(f"Image2 features (first 5): {data['image2_features'][:5]}")
            print(f"Processing time: {data['processing_time']:.3f}s")
            
            # Verify features are identical for same image
            feat1 = data['image1_features']
            feat2 = data['image2_features']
            max_diff = max(abs(f1 - f2) for f1, f2 in zip(feat1, feat2))
            print(f"Max feature difference: {max_diff:.8f}")
            
            if max_diff < 1e-6:
                print("✅ Features are identical for same image")
            else:
                print(f"⚠️  Features differ by {max_diff}")
                
        else:
            print(f"❌ Error: {response.text}")


async def main():
    """Run all tests"""
    print("🚀 Local LLM Interface API Test")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        await test_health()
        
        # Get available models
        models = await test_list_models()
        
        # Get vision models response for testing
        async with httpx.AsyncClient() as client:
            vision_response = await client.get(f"{BASE_URL}/v1/vision/models")
            vision_models = vision_response.json() if vision_response.status_code == 200 else {"data": []}
            
        await test_vision_models()
        await test_vision_encoding(vision_models.get('data', []))
        await test_multimodal_features(vision_models.get('data', []))
        await test_similarity_with_features()
        
        # Test each endpoint type
        await test_embeddings(models)
        await test_rerank(models)
        # await test_chat_completion(models)
        # await test_chat_completion(models)
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
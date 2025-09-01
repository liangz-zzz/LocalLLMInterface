#!/usr/bin/env python3
"""OpenAI client test for Local LLM Interface"""

import time
import requests
from openai import OpenAI

BASE_URL = "http://localhost:15530"

def test_openai_client():
    """Test using OpenAI client similar to user's setup"""
    print("🤖 Testing OpenAI Client Integration")
    print("=" * 50)
    
    # Initialize OpenAI client (same as user's setup)
    client = OpenAI(
        api_key="test-key",  # dummy key
        base_url=f"{BASE_URL}/v1",
        timeout=60.0  # 60 second timeout
    )
    
    # Test cases
    test_cases = [
        {"content": "Hello", "description": "Simple greeting"},
        {"content": "What is 2+2?", "description": "Math question (problematic query)"},
        {"content": "Tell me a short joke", "description": "Creative request"},
        {"content": "What is Python?", "description": "Knowledge question"},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"Query: '{test_case['content']}'")
        
        # Request parameters (same structure as user's)
        request_params = {
            "model": "Qwen2.5-7B-Instruct-AWQ",
            "messages": [
                {
                    "role": "user",
                    "content": test_case['content']
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        start_time = time.time()
        
        try:
            print("📤 Sending request...")
            response = client.chat.completions.create(**request_params)
            end_time = time.time()
            
            # Extract response (same as user's code)
            response_params = response.model_dump(exclude_none=True)
            
            print(f"✅ Success! Response time: {end_time - start_time:.2f}s")
            print(f"📄 Response: {response_params['choices'][0]['message']['content']}")
            print(f"📊 Usage: {response_params['usage']}")
            
        except Exception as e:
            end_time = time.time()
            print(f"❌ Error after {end_time - start_time:.2f}s: {e}")
            print(f"   Error type: {type(e).__name__}")

def test_with_system_message():
    """Test with system message"""
    print(f"\n\n🎭 Testing with System Message")
    print("=" * 50)
    
    client = OpenAI(
        api_key="test-key",
        base_url=f"{BASE_URL}/v1",
        timeout=60.0
    )
    
    request_params = {
        "model": "Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Give concise answers."
            },
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    start_time = time.time()
    
    try:
        print("📤 Sending request with system message...")
        response = client.chat.completions.create(**request_params)
        end_time = time.time()
        
        response_params = response.model_dump(exclude_none=True)
        
        print(f"✅ Success! Response time: {end_time - start_time:.2f}s")
        print(f"📄 Response: {response_params['choices'][0]['message']['content']}")
        print(f"📊 Usage: {response_params['usage']}")
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Error after {end_time - start_time:.2f}s: {e}")

def test_streaming():
    """Test streaming response"""
    print(f"\n\n🌊 Testing Streaming Response")
    print("=" * 50)
    
    client = OpenAI(
        api_key="test-key",
        base_url=f"{BASE_URL}/v1",
        timeout=60.0
    )
    
    request_params = {
        "model": "Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 5"
            }
        ],
        "stream": True,
        "max_tokens": 50
    }
    
    start_time = time.time()
    
    try:
        print("📤 Starting streaming request...")
        response = client.chat.completions.create(**request_params)
        
        print("📥 Streaming response:")
        content_parts = []
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_parts.append(content)
                print(content, end="", flush=True)
        
        end_time = time.time()
        print(f"\n✅ Streaming completed! Total time: {end_time - start_time:.2f}s")
        print(f"📄 Full response: {''.join(content_parts)}")
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Streaming error after {end_time - start_time:.2f}s: {e}")

def test_health_check():
    """Test health endpoint"""
    print(f"\n\n🏥 Testing Health Check")
    print("=" * 50)
    
    import requests
    
    try:
        response = requests.get(f"{BASE_URL}/v1/health", timeout=5)
        print(f"✅ Health check status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Available models: {data['models']['available']}")
            print(f"🔄 Loaded models: {data['models']['loaded']}")
            print(f"💾 GPU memory: {data['memory']['gpu_memory_used_gb']}GB")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    print("🚀 OpenAI Client Test Suite for Local LLM Interface")
    print("=" * 60)
    
    # Test health first
    test_health_check()
    
    # Test basic OpenAI client functionality
    test_openai_client()
    
    # Test with system message
    test_with_system_message()
    
    # Test streaming (if supported)
    test_streaming()
    
    print("\n\n🎯 Test Suite Complete!")
    print("=" * 60)
#!/usr/bin/env python3
"""Test simple query that might cause hanging"""

import asyncio
import httpx
import time
import json

BASE_URL = "http://localhost:15530"

async def test_simple_math():
    """Test the specific query that's causing issues"""
    print("🧮 Testing simple math query that caused hanging...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        request_data = {
            "model": "Qwen2.5-7B-Instruct-AWQ",
            "messages": [
                {
                    "role": "user", 
                    "content": "What is 2+2?"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 50  # Keep it short
        }
        
        print(f"Request: {json.dumps(request_data, indent=2)}")
        print("Sending request...")
        start_time = time.time()
        
        try:
            response = await client.post(f"{BASE_URL}/v1/chat/completions", json=request_data)
            end_time = time.time()
            
            print(f"Response time: {end_time - start_time:.2f}s")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['choices'][0]['message']['content']}")
                print(f"Usage: {data['usage']}")
            else:
                print(f"Error: {response.text}")
                
        except httpx.TimeoutException:
            end_time = time.time()
            print(f"⚠️  Request timed out after {end_time - start_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            print(f"❌ Error after {end_time - start_time:.2f}s: {e}")

async def test_with_system_message():
    """Test with system message for comparison"""
    print("\n🤖 Testing with system message for comparison...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        request_data = {
            "model": "Qwen2.5-7B-Instruct-AWQ", 
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2+2?"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        print(f"Request with system message...")
        start_time = time.time()
        
        try:
            response = await client.post(f"{BASE_URL}/v1/chat/completions", json=request_data)
            end_time = time.time()
            
            print(f"Response time: {end_time - start_time:.2f}s")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['choices'][0]['message']['content']}")
                print(f"Usage: {data['usage']}")
            else:
                print(f"Error: {response.text}")
                
        except httpx.TimeoutException:
            end_time = time.time()
            print(f"⚠️  Request timed out after {end_time - start_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            print(f"❌ Error after {end_time - start_time:.2f}s: {e}")

async def test_different_simple_queries():
    """Test other simple queries to see if it's content-specific"""
    print("\n📝 Testing other simple queries...")
    
    test_queries = [
        "Hello",
        "What is your name?", 
        "Tell me a joke",
        "What is 1+1?",
        "How are you?"
    ]
    
    for query in test_queries:
        print(f"\n➤ Testing: '{query}'")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            request_data = {
                "model": "Qwen2.5-7B-Instruct-AWQ",
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 30
            }
            
            start_time = time.time()
            
            try:
                response = await client.post(f"{BASE_URL}/v1/chat/completions", json=request_data)
                end_time = time.time()
                
                print(f"  ✅ Response time: {end_time - start_time:.2f}s")
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content'][:100]
                    print(f"  📄 Response: {content}...")
                else:
                    print(f"  ❌ Error: {response.text}")
                    
            except httpx.TimeoutException:
                end_time = time.time()
                print(f"  ⚠️  Timed out after {end_time - start_time:.2f}s")
            except Exception as e:
                end_time = time.time()
                print(f"  ❌ Error after {end_time - start_time:.2f}s: {e}")

async def main():
    print("🔍 Testing Simple Query Issues")
    print("=" * 50)
    
    # Check if service is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/v1/health")
            if response.status_code != 200:
                print("❌ Service not healthy, aborting tests")
                return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return
    
    # Run tests
    await test_simple_math()
    await test_with_system_message()
    await test_different_simple_queries()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""Test the specific 'What is 2+2?' query that causes issues"""

import time
from openai import OpenAI

def test_math_query():
    """Test the exact query that user is having trouble with"""
    print("🧮 Testing Math Query: 'What is 2+2?'")
    print("=" * 50)
    
    # Initialize OpenAI client (same as user's setup)
    client = OpenAI(
        api_key="test-key",  # dummy key
        base_url="http://localhost:15530/v1",
        timeout=120.0  # 2 minute timeout
    )
    
    # Create request (same structure as user's)
    request_params = {
        "model": "Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ]
    }
    
    print(f"📤 Sending request...")
    print(f"Request: {request_params}")
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(**request_params)
        end_time = time.time()
        
        # Extract response (same as user's code)
        response_params = response.model_dump(exclude_none=True)
        
        print(f"✅ Success! Response time: {end_time - start_time:.2f}s")
        print(f"📄 Response: {response_params['choices'][0]['message']['content']}")
        print(f"📊 Usage: {response_params['usage']}")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Error after {end_time - start_time:.2f}s: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_math_query()
    if success:
        print("\n🎯 Math query test completed successfully!")
    else:
        print("\n💥 Math query test failed!")
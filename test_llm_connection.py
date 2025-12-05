"""Simple test to debug LLM connection."""

import asyncio
import aiohttp


async def test_connection():
    """Test basic connection to Ollama."""
    base_url = "http://localhost:11434/v1"
    model = "gemma3:27b"
    
    print(f"Testing connection to: {base_url}")
    print(f"Model: {model}")
    print()
    
    # Test 1: Check models endpoint
    print("1. Testing /v1/models endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/models") as response:
                data = await response.json()
                print(f"✓ Models available: {len(data.get('data', []))} models")
                models = [m['id'] for m in data.get('data', [])]
                print(f"  Available: {models}")
                if model in models:
                    print(f"  ✓ Model '{model}' is available")
                else:
                    print(f"  ⚠️  Model '{model}' not found! Available models: {models}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Test 2: Simple completion request
    print("2. Testing /v1/chat/completions endpoint...")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'Hello' and nothing else."}],
        "temperature": 0.7,
        "max_tokens": 10
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"  Sending request...")
            async with session.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                print(f"  Response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"  ✓ Response received: {content}")
                    print(f"  Token usage: {data.get('usage', {})}")
                else:
                    error_text = await response.text()
                    print(f"  ✗ Error response: {error_text}")
    except asyncio.TimeoutError:
        print(f"  ✗ Request timed out after 30 seconds")
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())

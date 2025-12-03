"""Example using self-hosted LLM via HTTP API."""

import asyncio
from src.maker_core import Maker
from src.maker_core.providers import CustomLLMProvider


async def main():
    """
    Demonstrate using MAKER with a self-hosted LLM.
    
    This works with any LLM that has an OpenAI-compatible API, including:
    - vLLM (https://github.com/vllm-project/vllm)
    - Text Generation Inference (https://github.com/huggingface/text-generation-inference)
    - LocalAI (https://localai.io/)
    - Ollama with OpenAI compatibility (https://ollama.ai/)
    - LM Studio (https://lmstudio.ai/)
    """
    
    # Example 1: vLLM Server
    print("=" * 80)
    print("Example 1: vLLM Server")
    print("=" * 80)
    print("""
    To start a vLLM server:
    
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --host 0.0.0.0 \\
        --port 8000
    
    Then use:
    """)
    
    provider_vllm = CustomLLMProvider(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-2-7b-chat-hf",
        timeout=120  # Self-hosted models may need more time
    )
    
    # Example 2: Ollama with OpenAI Compatibility
    print("\n" + "=" * 80)
    print("Example 2: Ollama (OpenAI Compatible Mode)")
    print("=" * 80)
    print("""
    To use Ollama:
    
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull llama2
    3. Ollama automatically provides OpenAI-compatible API
    
    Then use:
    """)
    
    provider_ollama = CustomLLMProvider(
        base_url="http://localhost:11434/v1",
        model="llama2"
    )
    
    # Example 3: Text Generation Inference (TGI)
    print("\n" + "=" * 80)
    print("Example 3: Text Generation Inference (HuggingFace)")
    print("=" * 80)
    print("""
    To start TGI with Docker:
    
    docker run --gpus all --shm-size 1g -p 8080:80 \\
        ghcr.io/huggingface/text-generation-inference:latest \\
        --model-id mistralai/Mistral-7B-Instruct-v0.2
    
    Then use:
    """)
    
    provider_tgi = CustomLLMProvider(
        base_url="http://localhost:8080/v1",
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    # Example 4: LocalAI
    print("\n" + "=" * 80)
    print("Example 4: LocalAI")
    print("=" * 80)
    print("""
    To start LocalAI with Docker:
    
    docker run -p 8080:8080 \\
        -v $PWD/models:/models \\
        localai/localai:latest
    
    Then use:
    """)
    
    provider_localai = CustomLLMProvider(
        base_url="http://localhost:8080/v1",
        model="gpt-3.5-turbo"  # LocalAI model name
    )
    
    # Example 5: LM Studio
    print("\n" + "=" * 80)
    print("Example 5: LM Studio")
    print("=" * 80)
    print("""
    To use LM Studio:
    
    1. Download and open LM Studio
    2. Load a model
    3. Start the local server (usually on port 1234)
    
    Then use:
    """)
    
    provider_lmstudio = CustomLLMProvider(
        base_url="http://localhost:1234/v1",
        model="local-model"
    )
    
    # Choose which provider to use for the demo
    # Replace with your actual running server
    print("\n" + "=" * 80)
    print("Running Demo (using Ollama as example)")
    print("=" * 80)
    print("\nNote: Make sure you have Ollama running with a model loaded!")
    print("Example: ollama run llama2\n")
    
    # Use Ollama provider for the demo
    provider = provider_ollama
    
    # Create Maker instance
    maker = Maker(provider=provider)
    
    # Add event listeners
    maker.on("decomposed", lambda data: 
        print(f"✓ Decomposed into {data['count']} sub-questions"))
    maker.on("votingStart", lambda data: 
        print(f"→ Voting on: {data['question']}"))
    maker.on("votingComplete", lambda data: 
        print(f"✓ Consensus in {data['rounds']} rounds"))
    
    # Ask a simple question (to test connection)
    try:
        question = "What is the capital of France?"
        print(f"Question: {question}\n")
        
        result = await maker.ask(question)
        
        print(f"\n{'='*80}")
        print(f"Answer: {result['answer']}")
        print(f"\nConfidence: {result['confidence'].value}")
        print(f"Decomposition used: {result['decomposition_used']}")
        print(f"Total voting rounds: {result['total_voting_rounds']}")
        print(f"{'='*80}\n")
        
        # Try a more complex question
        print("\n" + "="*80)
        print("Trying a more complex question...")
        print("="*80 + "\n")
        
        complex_question = "What are the main benefits of exercise?"
        print(f"Question: {complex_question}\n")
        
        result2 = await maker.ask(complex_question)
        
        print(f"\n{'='*80}")
        print(f"Answer: {result2['answer']}")
        print(f"\nConfidence: {result2['confidence'].value}")
        print(f"Sub-questions answered: {len(result2['sub_results'])}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your self-hosted LLM server is running!")
        print("For Ollama: Run 'ollama serve' and 'ollama run llama2'")
        print("For vLLM: Start the API server as shown above")
        print("For others: Check the documentation for your specific setup\n")


async def minimal_example():
    """
    Minimal example - just replace the URL and model name.
    """
    print("\n" + "="*80)
    print("MINIMAL EXAMPLE - Copy and modify this:")
    print("="*80 + "\n")
    
    code = """
import asyncio
from maker_core import Maker
from maker_core.providers import CustomLLMProvider

async def main():
    # Replace with your server details
    provider = CustomLLMProvider(
        base_url="http://localhost:8000/v1",  # Your LLM server URL
        model="your-model-name",              # Your model identifier
        api_key=None,                         # Optional API key
        timeout=120                           # Timeout in seconds
    )
    
    maker = Maker(provider=provider)
    
    result = await maker.ask("What is machine learning?")
    print(result['answer'])

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    print(code)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MAKER-CORE: SELF-HOSTED LLM EXAMPLE")
    print("="*80 + "\n")
    
    # Show setup instructions first
    print("This example demonstrates using MAKER with self-hosted LLMs.")
    print("Choose one of the following options:\n")
    
    print("1. Run the full demo (requires running LLM server)")
    print("2. Just show the minimal example code\n")
    
    choice = input("Enter choice (1 or 2, or just press Enter to see code): ").strip()
    
    if choice == "1":
        asyncio.run(main())
    else:
        asyncio.run(minimal_example())

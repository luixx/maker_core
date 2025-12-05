"""Minimal test with MAKER framework."""

import asyncio
from src.maker_core import Maker
from src.maker_core.providers import CustomLLMProvider


async def test_maker():
    """Test MAKER with simple question."""
    print("Setting up MAKER...")
    
    # Configure provider
    provider = CustomLLMProvider(
        base_url="http://localhost:11434/v1",
        model="gemma3:27b",
        timeout=120
    )
    
    print("✓ Provider configured")
    
    # Create MAKER instance
    maker = Maker(provider=provider)
    print("✓ MAKER instance created")
    
    # Ask a simple question
    question = "What is 2+2? Answer with just the number."
    print(f"\nQuestion: {question}")
    print("Waiting for response...\n")
    
    try:
        result = await maker.ask(question)
        
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence'].value}")
        print(f"Decomposition used: {result['decomposition_used']}")
        if result.get('sub_results'):
            print(f"Sub-questions: {len(result['sub_results'])}")
    except Exception as e:
        print("=" * 60)
        print("ERROR!")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_maker())

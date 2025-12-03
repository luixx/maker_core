"""Example using synchronous Maker wrapper."""

import os
from src.maker_core import MakerSync
from src.maker_core.providers import OpenAIProvider


def main():
    """Run MAKER synchronously (no async/await needed)."""
    # Initialize provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    
    # Create synchronous Maker instance
    maker = MakerSync(provider=provider)
    
    # Register event listener
    maker.on("votingComplete", lambda data: print(f"✓ Vote complete: {data['rounds']} rounds"))
    
    # Ask questions synchronously
    questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What are the benefits of exercise?",
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        # No async/await needed!
        result = maker.ask(question)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence'].value}\n")


if __name__ == "__main__":
    main()

"""Basic example using OpenAI provider."""

import asyncio
import os
from src.maker_core import Maker
from src.maker_core.providers import OpenAIProvider


async def main():
    """Run basic MAKER example with OpenAI."""
    # Initialize OpenAI provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    provider = OpenAIProvider(
        api_key=api_key,
        model="gpt-4"
    )
    
    # Create Maker instance
    maker = Maker(provider=provider)
    
    # Register event listeners for progress tracking
    maker.on("decomposed", lambda data: print(f"✓ Decomposed into {data['count']} sub-questions"))
    maker.on("votingStart", lambda data: print(f"→ Voting on: {data['question']}"))
    maker.on("votingComplete", lambda data: print(f"✓ Consensus reached in {data['rounds']} rounds"))
    
    # Ask a question
    question = "What are the main causes of climate change?"
    print(f"\nQuestion: {question}\n")
    
    result = await maker.ask(question)
    
    print(f"\n{'='*60}")
    print(f"Answer: {result['answer']}")
    print(f"\nConfidence: {result['confidence'].value}")
    print(f"Decomposition used: {result['decomposition_used']}")
    print(f"Total voting rounds: {result['total_voting_rounds']}")
    print(f"Red flags: {result['red_flags_encountered']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

"""Example using Anthropic Claude provider."""

import asyncio
import os
from src.maker_core import Maker
from src.maker_core.providers import AnthropicProvider


async def main():
    """Run MAKER example with Anthropic Claude."""
    # Initialize Anthropic provider
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    provider = AnthropicProvider(
        api_key=api_key,
        model="claude-3-5-sonnet-20241022"
    )
    
    # Create Maker instance with custom configuration
    maker = Maker(
        provider=provider,
        config={
            "voting_threshold": 5,
            "max_voting_rounds": 15,
            "max_sub_questions": 4,
            "enable_red_flag_filter": True,
        }
    )
    
    # Event tracking
    def on_progress(data):
        if "round" in data:
            print(f"  Round {data['round']}: {data['vote_counts']}")
    
    maker.on("voteProgress", on_progress)
    
    # Ask a complex question
    question = "What were the key technological innovations during the Industrial Revolution and how did they transform society?"
    print(f"\nQuestion: {question}\n")
    
    result = await maker.ask(question)
    
    print(f"\n{'='*80}")
    print(f"Final Answer:\n{result['answer']}\n")
    print(f"Confidence: {result['confidence'].value}")
    print(f"Sub-questions answered: {len(result['sub_questions'])}")
    
    for i, sub_result in enumerate(result['sub_questions'], 1):
        print(f"\n  {i}. {sub_result['question']}")
        print(f"     Rounds: {sub_result['rounds_taken']}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

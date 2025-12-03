"""Example demonstrating event tracking and progress monitoring."""

import asyncio
import os
from src.maker_core import Maker
from src.maker_core.providers import OpenAIProvider


async def main():
    """Demonstrate comprehensive event tracking."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    maker = Maker(provider=provider)
    
    # Track all events
    def on_decomposition_start(data):
        print(f"\n🔍 Starting decomposition for: {data['question']}")
    
    def on_classification_complete(data):
        classification = data['classification']
        print(f"📊 Complexity: {classification['complexity_score']}/10")
        print(f"   Needs decomposition: {data['needs_decomposition']}")
        print(f"   Reasoning: {classification['reasoning']}")
    
    def on_decomposed(data):
        print(f"\n✂️  Decomposed into {data['count']} sub-questions:")
        for i, sq in enumerate(data['sub_questions'], 1):
            print(f"   {i}. {sq['question']}")
    
    def on_voting_start(data):
        print(f"\n🗳️  Voting on sub-question {data['index']+1}/{data['total']}")
        print(f"   Question: {data['question']}")
    
    def on_vote_progress(data):
        print(f"   Round {data['round']}: {data['vote_counts']}")
    
    def on_voting_complete(data):
        print(f"   ✓ Consensus: {data['answer']} (in {data['rounds']} rounds)")
    
    def on_red_flagged(data):
        print(f"   ⚠️  Red flags: {data['count']}")
    
    def on_synthesis_start(data):
        print(f"\n🔧 Synthesizing {len(data['sub_results'])} sub-answers...")
    
    def on_synthesis_complete(data):
        print("   ✓ Synthesis complete")
    
    def on_complete(data):
        print(f"\n✅ Complete!")
        print(f"   Confidence: {data['confidence'].value}")
        print(f"   Total voting rounds: {data['total_voting_rounds']}")
        print(f"   Red flags encountered: {data['red_flags_encountered']}")
    
    # Register all event listeners
    maker.on("decompositionStart", on_decomposition_start)
    maker.on("classificationComplete", on_classification_complete)
    maker.on("decomposed", on_decomposed)
    maker.on("votingStart", on_voting_start)
    maker.on("voteProgress", on_vote_progress)
    maker.on("votingComplete", on_voting_complete)
    maker.on("redFlagged", on_red_flagged)
    maker.on("synthesisStart", on_synthesis_start)
    maker.on("synthesisComplete", on_synthesis_complete)
    maker.on("complete", on_complete)
    
    # Ask a complex question
    question = "What factors contributed to the Renaissance and how did it change European society?"
    print(f"Question: {question}")
    
    result = await maker.ask(question)
    
    print(f"\n{'='*80}")
    print(f"Final Answer:\n{result['answer']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

"""Debug script to see what's happening with voting."""
import asyncio
from src.maker_core import Maker
from src.maker_core.providers import CustomLLMProvider

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "mistral:latest"

async def debug():
    provider = CustomLLMProvider(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        timeout=300
    )
    
    maker = Maker(provider=provider)
    
    # Add event listeners to debug
    maker.on("classificationStart", lambda d: print(f"[CLASSIFY] Starting..."))
    maker.on("classificationComplete", lambda d: print(f"[CLASSIFY] Result: {d}"))
    maker.on("votingStart", lambda d: print(f"[VOTE] Starting: {d['question'][:50]}..."))
    maker.on("vote", lambda d: print(f"[VOTE] Round {d['round']}: answer='{d['answer'][:50] if d['answer'] else 'None'}...' redFlag={d.get('redFlag')}"))
    maker.on("votingComplete", lambda d: print(f"[VOTE] Complete: consensus={d.get('consensusReached')}, answer='{d['answer'][:50] if d['answer'] else 'None'}...'"))
    
    question = "What is 2 + 2?"
    print(f"Question: {question}\n")
    
    result = await maker.ask(question)
    
    print(f"\n=== RESULT ===")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")

asyncio.run(debug())

# MAKER-Core

A Python implementation of the **MAKER algorithm** (MAximizing DEcomposition and Redundancy) for reliable LLM responses.

Based on the paper: ["Solving a Million-Step LLM Task with Zero Errors"](https://arxiv.org/abs/2511.09030) by Zheng et al. (2024)

Original TypeScript implementation: [github.com/Gatos90/maker](https://github.com/Gatos90/maker)

## Features

- **Maximal Agentic Decomposition (MAD)** - Breaks complex questions into atomic sub-questions
- **First-to-ahead-by-K Voting** - Consensus-based answers using Algorithm 2 from the paper
- **Red-flag Filtering** - Detects unreliable responses (too long, too short, parse failures)
- **Multiple LLM Providers** - OpenAI, Anthropic, Azure OpenAI, and custom/local LLMs
- **Robust Parsing** - Works with both strict JSON APIs and flexible local LLMs

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install aiohttp openai anthropic
```

## Quick Start

### With OpenAI

```python
import asyncio
from maker_core import Maker
from maker_core.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="your-key", model="gpt-4")
    maker = Maker(provider=provider)
    
    result = await maker.ask("What causes climate change?")
    print(result["answer"])
    print(result["confidence"])  # HIGH, MEDIUM, or LOW

asyncio.run(main())
```

### With Local LLMs (Ollama, vLLM, LM Studio)

```python
import asyncio
from maker_core import Maker
from maker_core.providers import CustomLLMProvider

async def main():
    provider = CustomLLMProvider(
        base_url="http://localhost:11434/v1",  # Ollama
        model="mistral:latest",
        timeout=300
    )
    maker = Maker(provider=provider)
    
    result = await maker.ask(
        "What is photosynthesis?",
        options={"context": "Your document text here..."}
    )
    print(result["answer"])

asyncio.run(main())
```

### Synchronous Usage

```python
from maker_core import MakerSync
from maker_core.providers import OpenAIProvider

provider = OpenAIProvider(api_key="your-key", model="gpt-4")
maker = MakerSync(provider=provider)

result = maker.ask("What is the speed of light?")
print(result["answer"])
```

## How It Works

### Algorithm 2: First-to-ahead-by-K Voting

The core voting algorithm from the MAKER paper:

```
while True:
    y = get_vote(x, M)      # Sample answer from LLM
    V[y] = V[y] + 1         # Increment vote count
    if V[y] >= k + max(V[v] for v != y):  # Check consensus
        return y
```

**Key insight**: The winner must be **K votes AHEAD** of the runner-up, not just have K total votes.

Example with K=2:
- If answer A has 3 votes and answer B has 1 vote → A wins (3 >= 2 + 1)
- If answer A has 3 votes and answer B has 2 votes → Continue voting (3 < 2 + 2)

### Red-flag Filtering (Section 3.3)

Responses are filtered out if they:
- Are too long (>750 tokens / ~3000 characters)
- Are too short (<5 characters)
- Failed to parse (for structured responses)

### Question Decomposition

Complex questions are automatically broken down into simpler sub-questions that can be answered independently, then synthesized into a final answer.

## Configuration

```python
from maker_core import Maker

maker = Maker(
    provider=provider,
    config={
        # Voting configuration
        "voting": {
            "k": 2,              # Votes ahead needed for consensus
            "max_rounds": 10,   # Maximum voting rounds
        },
        # Red-flag filtering
        "red_flags": {
            "max_tokens": 750,   # Max answer length in tokens
            "min_length": 5,     # Minimum answer length in chars
        },
        # Decomposition
        "max_sub_questions": 5,  # Max sub-questions for complex queries
    }
)
```

## Result Structure

### MakerResult

```python
{
    "answer": str,              # Final answer
    "confidence": Confidence,   # HIGH, MEDIUM, or LOW
    "is_decomposed": bool,      # Whether question was decomposed
    "sub_results": [...],       # Results for each sub-question
    "voting_stats": {...},      # Aggregated voting statistics
    "total_votes": int,         # Total votes across all sub-questions
    "total_red_flags": int,     # Total red flags encountered
}
```

### VotingResult

```python
{
    "answer": str,                  # Winning answer
    "consensus_reached": bool,      # Whether K-ahead consensus was reached
    "vote_counts": {str: int},      # Votes per normalized answer
    "rounds_taken": int,            # Total voting rounds
    "valid_votes": int,             # Votes not filtered by red-flags
    "red_flags_encountered": int,   # Filtered responses count
    "winning_vote_count": int,      # Votes for winning answer
    "margin": int,                  # Lead over runner-up
}
```

## Event Tracking

Track the MAKER process with event listeners:

```python
maker = Maker(provider=provider)

maker.on("classificationStart", lambda d: print("Classifying..."))
maker.on("classificationComplete", lambda d: print(f"Needs decomposition: {d['needs_decomposition']}"))
maker.on("decomposed", lambda d: print(f"Split into {d['count']} sub-questions"))
maker.on("votingStart", lambda d: print(f"Voting on: {d['question'][:50]}..."))
maker.on("vote", lambda d: print(f"Vote {d['round']}: {d['answer'][:30]}..."))
maker.on("votingComplete", lambda d: print(f"Consensus: {d['consensusReached']}"))
maker.on("synthesisStart", lambda d: print("Synthesizing..."))
maker.on("synthesisComplete", lambda d: print("Done!"))
maker.on("error", lambda d: print(f"Error: {d['error']}"))

result = await maker.ask("Your question here")
```

### Available Events

| Event | Data | Description |
|-------|------|-------------|
| `classificationStart` | `{question}` | Starting question classification |
| `classificationComplete` | `{needs_decomposition, complexity_score}` | Classification finished |
| `decomposed` | `{count, sub_questions}` | Question decomposed |
| `votingStart` | `{question}` | Starting voting on a question |
| `vote` | `{round, answer, redFlag}` | Individual vote cast |
| `votingComplete` | `{answer, rounds, consensusReached}` | Voting finished |
| `synthesisStart` | `{count}` | Starting answer synthesis |
| `synthesisComplete` | `{answer}` | Synthesis finished |
| `error` | `{error, phase}` | Error occurred |

## Providers

### OpenAI
```python
from maker_core.providers import OpenAIProvider
provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
```

### Anthropic
```python
from maker_core.providers import AnthropicProvider
provider = AnthropicProvider(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
```

### Azure OpenAI
```python
from maker_core.providers import AzureOpenAIProvider
provider = AzureOpenAIProvider(
    api_key="...",
    endpoint="https://your-resource.openai.azure.com",
    deployment="gpt-4",
    api_version="2024-02-15-preview"
)
```

### Custom/Local LLMs
```python
from maker_core.providers import CustomLLMProvider
provider = CustomLLMProvider(
    base_url="http://localhost:11434/v1",  # Ollama, vLLM, LM Studio, etc.
    model="mistral:latest",
    timeout=300  # Increase for large models
)
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/maker_core --cov-report=term-missing
```

## Examples

See the `examples/` directory for complete examples:

- `basic_openai.py` - Basic usage with OpenAI
- `anthropic_example.py` - Using Anthropic Claude
- `azure_openai_example.py` - Azure OpenAI integration
- `custom_llm_example.py` - Local LLM with Ollama
- `document_context_example.py` - Using document context
- `document_context_localhost.py` - Document context with local LLMs
- `event_tracking.py` - Tracking MAKER events
- `sync_usage.py` - Synchronous API usage

## Architecture

```
maker_core/
├── __init__.py           # Package exports
├── maker.py              # Main Maker and MakerSync classes
├── decomposer.py         # Question decomposition (MAD)
├── voting_engine.py      # First-to-ahead-by-K voting
├── synthesizer.py        # Answer synthesis
├── red_flag_filter.py    # Response filtering
├── types.py              # TypedDict definitions
└── providers/
    ├── base.py           # Abstract LLMProvider
    ├── openai.py         # OpenAI provider
    ├── anthropic.py      # Anthropic provider
    ├── azure_openai.py   # Azure OpenAI provider
    └── custom_llm.py     # Custom/local LLM provider
```

## License

MIT License - See LICENSE file for details.

## References

- Paper: [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030)
- Original Implementation: [github.com/Gatos90/maker](https://github.com/Gatos90/maker)

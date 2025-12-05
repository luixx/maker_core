# maker-core 🎯

A Python implementation of the **MAKER** (MAximizing DEcomposition and Redundancy) algorithm for achieving highly reliable responses from Large Language Models (LLMs).

Based on the research paper: ["Solving a Million-Step LLM Task with Zero Errors"](https://arxiv.org/abs/2511.09030) by Zheng et al. (2024).

## 🌟 Features

- **Maximal Agentic Decomposition (MAD)**: Automatically breaks complex questions into atomic sub-questions
- **Consensus Voting**: Uses first-to-ahead-by-K algorithm to achieve reliable answers through redundancy
- **Red-Flag Filtering**: Detects and filters unreliable LLM responses
- **Multi-Provider Support**: Works with OpenAI, Anthropic Claude, and Azure OpenAI
- **Event-Driven Architecture**: Track progress and monitor the pipeline in real-time
- **Async & Sync Support**: Use async/await or synchronous interfaces
- **Type-Safe**: Fully typed with mypy-compatible type hints
- **Well-Tested**: Comprehensive test suite with pytest

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/maker-core.git
cd maker-core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🚀 Quick Start

### Basic Usage with OpenAI

```python
import asyncio
from maker_core import Maker
from maker_core.providers import OpenAIProvider

async def main():
    # Initialize provider
    provider = OpenAIProvider(
        api_key="your-openai-api-key",
        model="gpt-4"
    )
    
    # Create Maker instance
    maker = Maker(provider=provider)
    
    # Ask a question
    result = await maker.ask("What are the main causes of climate change?")
    
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")

asyncio.run(main())
```

### Synchronous Usage

```python
from maker_core import MakerSync
from maker_core.providers import OpenAIProvider

provider = OpenAIProvider(api_key="your-key", model="gpt-4")
maker = MakerSync(provider=provider)

result = maker.ask("What is photosynthesis?")
print(result['answer'])
```

## 🎛️ Configuration

Customize MAKER's behavior with configuration options:

```python
from maker_core import Maker, MakerConfig

config = MakerConfig(
    voting_threshold=5,          # Votes ahead needed for consensus
    max_voting_rounds=15,        # Maximum voting attempts
    max_sub_questions=4,         # Max sub-questions per decomposition
    enable_red_flag_filter=True  # Enable response filtering
)

maker = Maker(provider=provider, config=config)
```

## 🔌 Supported Providers

### OpenAI

```python
from maker_core.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    model="gpt-4",  # or "gpt-4-turbo", "gpt-3.5-turbo"
)
```

### Anthropic Claude

```python
from maker_core.providers import AnthropicProvider

provider = AnthropicProvider(
    api_key="your-api-key",
    model="claude-3-5-sonnet-20241022"
)
```

### Azure OpenAI

```python
from maker_core.providers import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    api_key="your-api-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-10-21",
    deployment_name="your-deployment"
)
```

## 📊 Event Tracking

Monitor the MAKER pipeline with event listeners:

```python
maker = Maker(provider=provider)

# Track decomposition
maker.on("decomposed", lambda data: 
    print(f"Decomposed into {data['count']} sub-questions"))

# Track voting progress
maker.on("voteProgress", lambda data:
    print(f"Round {data['round']}: {data['vote_counts']}"))

# Track completion
maker.on("complete", lambda data:
    print(f"Confidence: {data['confidence']}"))

result = await maker.ask("Your question here")
```

### Available Events

- `decompositionStart`: Decomposition begins
- `classificationComplete`: Question classified
- `decomposed`: Sub-questions generated
- `votingStart`: Voting begins for a sub-question
- `voteProgress`: Each voting round update
- `votingComplete`: Consensus reached
- `redFlagged`: Red flags detected
- `synthesisStart`: Synthesis begins
- `synthesisComplete`: Synthesis complete
- `complete`: Full pipeline complete

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/maker_core --cov-report=html

# Run specific test file
pytest tests/test_maker.py

# Run with verbose output
pytest -v
```

## 📚 Examples

Check the `examples/` directory for detailed usage examples:

- `basic_openai.py` - Basic usage with OpenAI
- `anthropic_example.py` - Using Anthropic Claude
- `azure_openai_example.py` - Azure OpenAI integration
- `event_tracking.py` - Comprehensive event monitoring
- `sync_usage.py` - Synchronous wrapper usage

## 🏗️ Architecture

### Pipeline Flow

```
User Question
    ↓
[Classification] → Determine complexity
    ↓
[Decomposition] → Break into sub-questions (if needed)
    ↓
[Voting] → Consensus for each sub-question
    ↓
[Synthesis] → Combine into final answer
    ↓
Final Result
```

### Key Components

- **Decomposer**: Implements Maximal Agentic Decomposition (MAD)
- **VotingEngine**: First-to-ahead-by-K consensus algorithm
- **Synthesizer**: Combines sub-answers coherently
- **RedFlagFilter**: Detects unreliable responses
- **Providers**: Abstract LLM provider interface with multiple implementations

## 🔬 How It Works

### 1. Classification

MAKER first classifies the question complexity (1-10 scale). Questions with complexity ≥ 6 undergo decomposition.

### 2. Decomposition

Complex questions are broken into 2-5 atomic sub-questions that can be answered independently.

### 3. Consensus Voting

For each sub-question, MAKER uses the **first-to-ahead-by-K** algorithm:
- Samples multiple answers with varying temperature
- Continues until one answer is K votes ahead
- Filters out "red-flagged" responses (too long/short, parse failures)

### 4. Synthesis

Sub-answers are combined by the LLM into a coherent final response.

## 📈 Performance

Based on the original paper, MAKER achieves:
- **Near-zero error rates** on complex multi-step tasks
- **High reliability** through consensus voting
- **Efficient decomposition** of complex problems
- **Scalable** to million-step tasks

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original MAKER paper: [Zheng et al. (2024)](https://arxiv.org/abs/2511.09030)
- JavaScript implementation: [@sittingduck/maker-core](https://www.npmjs.com/package/@sittingduck/maker-core)

## 📮 Contact

For questions, issues, or suggestions, please [open an issue](https://github.com/yourusername/maker-core/issues) on GitHub.

---

**Note**: This is a research implementation. For production use, ensure proper API rate limiting, error handling, and cost management.

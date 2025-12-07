"""
maker-core: A Python implementation of the MAKER algorithm for reliable LLM responses.

Based on the paper "Solving a Million-Step LLM Task with Zero Errors"
by Zheng et al. (2024): https://arxiv.org/abs/2511.09030

This package implements:
- Maximal Agentic Decomposition (MAD) for breaking down complex questions
- First-to-ahead-by-K voting (Algorithm 2) for consensus-based answers
- Red-flag filtering (Section 3.3) for detecting unreliable responses

Quick Start:
    >>> from maker_core import Maker
    >>> from maker_core.providers import OpenAIProvider
    >>>
    >>> provider = OpenAIProvider(api_key="your-key", model="gpt-4")
    >>> maker = Maker(provider=provider)
    >>>
    >>> result = await maker.ask("What causes climate change?")
    >>> print(result["answer"])
    >>> print(result["confidence"])  # HIGH, MEDIUM, or LOW

For synchronous usage:
    >>> from maker_core import MakerSync
    >>> maker = MakerSync(provider=provider)
    >>> result = maker.ask("What is photosynthesis?")
"""

from .maker import Maker, MakerSync, EventEmitter
from .decomposer import Decomposer
from .voting_engine import VotingEngine, VoteData
from .synthesizer import Synthesizer
from .red_flag_filter import RedFlagFilter
from .types import (
    Confidence,
    RedFlagReason,
    Message,
    TokenUsage,
    CompletionRequest,
    CompletionResponse,
    Classification,
    SubQuestion,
    DecompositionResult,
    VotingResult,
    SubQuestionResult,
    VotingStats,
    MakerResult,
    MakerConfig,
    VotingConfig,
    RedFlagConfig,
    AskOptions,
)
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    AzureOpenAIProvider,
    CustomLLMProvider,
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "Maker",
    "MakerSync",
    "EventEmitter",
    # Core components
    "Decomposer",
    "VotingEngine",
    "VoteData",
    "Synthesizer",
    "RedFlagFilter",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "CustomLLMProvider",
    # Types - Enums
    "Confidence",
    "RedFlagReason",
    # Types - LLM Communication
    "Message",
    "TokenUsage",
    "CompletionRequest",
    "CompletionResponse",
    # Types - Pipeline
    "Classification",
    "SubQuestion",
    "DecompositionResult",
    "VotingResult",
    "SubQuestionResult",
    "VotingStats",
    "MakerResult",
    # Types - Configuration
    "MakerConfig",
    "VotingConfig",
    "RedFlagConfig",
    "AskOptions",
]

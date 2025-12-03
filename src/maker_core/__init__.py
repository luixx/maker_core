"""
maker-core: A Python implementation of the MAKER algorithm for reliable LLM responses.

Based on the paper "Solving a Million-Step LLM Task with Zero Errors"
by Zheng et al. (2024).
"""

from .maker import Maker, MakerSync, EventEmitter
from .decomposer import Decomposer
from .voting_engine import VotingEngine
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
    MakerResult,
    MakerConfig,
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
    # Components
    "Decomposer",
    "VotingEngine",
    "Synthesizer",
    "RedFlagFilter",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "CustomLLMProvider",
    # Types
    "Confidence",
    "RedFlagReason",
    "Message",
    "TokenUsage",
    "CompletionRequest",
    "CompletionResponse",
    "Classification",
    "SubQuestion",
    "DecompositionResult",
    "VotingResult",
    "SubQuestionResult",
    "MakerResult",
    "MakerConfig",
    "AskOptions",
]

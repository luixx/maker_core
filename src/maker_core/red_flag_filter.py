"""Red-flag filter for detecting unreliable LLM responses."""

import tiktoken
from typing import Optional
from .types import RedFlagResult, RedFlagReason, RedFlagConfig


class RedFlagFilter:
    """
    Filters out unreliable LLM responses based on the MAKER paper (Section 3.3).

    The filter uses two red-flag criteria:
    1. Response too long - Exceeds max_tokens (default 750). Long responses
       correlate with confusion and errors.
    2. Response too short - Below min_chars (default 5). Very short responses
       are often incomplete or invalid.

    Attributes:
        max_tokens: Maximum allowed token count before flagging (default: 750).
        min_chars: Minimum character count before flagging (default: 5).
        encoding: Tiktoken encoding for token counting.
    """

    def __init__(self, config: Optional[RedFlagConfig] = None) -> None:
        """
        Initialize the red-flag filter.

        Args:
            config: Optional configuration with max_tokens and min_chars.
                   Defaults to max_tokens=750, min_chars=5.
        """
        config = config or {}
        self.max_tokens = config.get("max_tokens", 750)
        self.min_chars = config.get("min_chars", 5)

        # Initialize tiktoken encoding for token counting
        # Using cl100k_base which is used by GPT-4 and GPT-3.5-turbo
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to a simple approximation if tiktoken fails
            self.encoding = None

    def check(self, answer: str, parse_succeeded: bool = True) -> RedFlagResult:
        """
        Check if a response should be red-flagged.

        According to the MAKER paper, red-flagged responses are discarded
        and the system re-samples.

        Args:
            answer: The LLM response text to check.
            parse_succeeded: Whether the response was successfully parsed (default: True).
                           Set to False if JSON parsing or other parsing failed.

        Returns:
            RedFlagResult containing is_flagged boolean and reason.

        Examples:
            >>> filter = RedFlagFilter()
            >>> result = filter.check("Valid answer text")
            >>> print(result["is_flagged"])  # False
            >>> result = filter.check("x" * 10000)
            >>> print(result["reason"])  # RedFlagReason.TOO_LONG
        """
        # Check for parse failure
        if not parse_succeeded:
            return RedFlagResult(is_flagged=True, reason=RedFlagReason.PARSE_FAILURE)

        # Check minimum character length
        if len(answer) < self.min_chars:
            return RedFlagResult(is_flagged=True, reason=RedFlagReason.TOO_SHORT)

        # Check maximum token length
        token_count = self._count_tokens(answer)
        if token_count > self.max_tokens:
            return RedFlagResult(is_flagged=True, reason=RedFlagReason.TOO_LONG)

        return RedFlagResult(is_flagged=False, reason=RedFlagReason.NONE)

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Uses tiktoken for accurate token counting when available,
        falls back to word-based estimation otherwise.

        Args:
            text: Text to count tokens in.

        Returns:
            Approximate number of tokens.
        """
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass

        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4

    def get_config(self) -> RedFlagConfig:
        """
        Get the current configuration.

        Returns:
            RedFlagConfig with current max_tokens and min_chars settings.
        """
        return RedFlagConfig(max_tokens=self.max_tokens, min_chars=self.min_chars)

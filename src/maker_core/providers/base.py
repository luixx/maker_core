"""Base provider interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..types import CompletionRequest, CompletionResponse


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement
    to work with the MAKER framework.
    """

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion from the LLM.

        Args:
            request: The completion request containing messages, temperature, and other parameters.

        Returns:
            CompletionResponse: The completion response with content, raw text, and usage stats.

        Raises:
            Exception: If the completion request fails.
        """
        pass

    def _extract_json_content(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON content from markdown code blocks if present.

        Args:
            content: The raw content that may contain JSON in markdown.

        Returns:
            Dict containing the parsed JSON, or the original content if not JSON.
        """
        import json
        import re

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse directly as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Return as-is if not valid JSON
            return {"content": content}

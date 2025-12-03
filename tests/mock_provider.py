"""Mock LLM provider for testing."""

import json
from typing import Dict
from src.maker_core.providers.base import LLMProvider
from src.maker_core.types import CompletionRequest, CompletionResponse


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses based on the prompt content.
    """

    def __init__(self, responses: Dict[str, str] = None) -> None:
        """
        Initialize the mock provider.

        Args:
            responses: Dictionary mapping prompt keywords to responses.
        """
        self.responses = responses or {}
        self.call_count = 0
        self.last_request = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Return a mock completion response.

        Args:
            request: Completion request.

        Returns:
            Mock completion response.
        """
        self.call_count += 1
        self.last_request = request

        # Extract the user message
        user_message = ""
        for msg in request["messages"]:
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # Find matching response
        content = "Mock response"
        for keyword, response in self.responses.items():
            if keyword.lower() in user_message.lower():
                content = response
                break

        return CompletionResponse(
            content=content,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

    def reset(self) -> None:
        """Reset call count and last request."""
        self.call_count = 0
        self.last_request = None

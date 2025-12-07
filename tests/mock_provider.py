"""Mock LLM provider for testing."""

import json
from typing import Dict, Any
from src.maker_core.providers.base import LLMProvider
from src.maker_core.types import CompletionRequest, CompletionResponse


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Returns predefined responses based on the prompt content.
    
    The mock provider automatically detects what kind of response is needed:
    - If the prompt asks for JSON (classification/decomposition), returns JSON
    - For voting prompts, wraps the answer in JSON format
    
    This allows proper mocking for both:
    - Classification/decomposition (expects raw JSON like {"needs_decomposition": true})
    - Voting (expects {"answer": "...", "confidence": "high"})
    """

    def __init__(self, responses: Dict[str, str] = None, json_mode: bool = True) -> None:
        """
        Initialize the mock provider.

        Args:
            responses: Dictionary mapping prompt keywords to responses.
            json_mode: If True, wrap non-JSON responses in JSON format for voting engine.
        """
        self.responses = responses or {}
        self.json_mode = json_mode
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
        answer = "Mock response"
        for keyword, response in self.responses.items():
            if keyword.lower() in user_message.lower():
                answer = response
                break

        # Check if response_format requests JSON
        response_format = request.get("response_format", {})
        is_json_request = response_format.get("type") == "json_object"
        
        # Also detect voting-style prompts that expect JSON
        is_voting_prompt = "answer" in user_message.lower() and "json" in user_message.lower()

        # Determine content format
        if (is_json_request or is_voting_prompt) and self.json_mode:
            # Check if the answer is already JSON-like (classification/decomposition responses)
            stripped = answer.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                # Already JSON - return as-is (for classification, decomposition)
                content = stripped
            else:
                # Plain text answer - wrap in voting format
                content = json.dumps({
                    "answer": answer,
                    "confidence": "high",
                    "reasoning": "Mock reasoning"
                })
        else:
            content = answer

        return CompletionResponse(
            content=content,
            raw=content,
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

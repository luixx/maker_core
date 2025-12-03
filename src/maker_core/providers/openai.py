"""OpenAI provider implementation."""

import json
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from .base import LLMProvider
from ..types import CompletionRequest, CompletionResponse, TokenUsage


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation.

    This provider interfaces with OpenAI's API to generate completions
    using GPT models.

    Attributes:
        api_key: OpenAI API key for authentication.
        model: Model identifier (e.g., 'gpt-4o-mini', 'gpt-4').
        client: AsyncOpenAI client instance.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model identifier to use for completions.
        """
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using OpenAI's API.

        Args:
            request: Completion request with messages and parameters.

        Returns:
            CompletionResponse with generated content and usage statistics.

        Raises:
            Exception: If the API call fails.
        """
        # Prepare the API call parameters
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": request["messages"],
            "temperature": request["temperature"],
        }

        # Add optional parameters
        if "max_tokens" in request:
            params["max_tokens"] = request["max_tokens"]

        if "response_format" in request:
            params["response_format"] = request["response_format"]

        # Make the API call
        response = await self.client.chat.completions.create(**params)

        # Extract content
        content = response.choices[0].message.content or ""

        # Parse JSON if response format was requested
        parsed_content = content
        if "response_format" in request and request["response_format"].get("type") == "json_object":
            try:
                parsed_data = json.loads(content)
                parsed_content = json.dumps(parsed_data)
            except json.JSONDecodeError:
                parsed_content = content

        # Extract usage statistics
        usage: TokenUsage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return CompletionResponse(content=parsed_content, raw=content, usage=usage)

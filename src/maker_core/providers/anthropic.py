"""Anthropic (Claude) provider implementation."""

import json
from typing import Any, Dict
from anthropic import AsyncAnthropic
from .base import LLMProvider
from ..types import CompletionRequest, CompletionResponse, TokenUsage, Message


class AnthropicProvider(LLMProvider):
    """
    Anthropic (Claude) LLM provider implementation.

    This provider interfaces with Anthropic's API to generate completions
    using Claude models.

    Attributes:
        api_key: Anthropic API key for authentication.
        model: Model identifier (e.g., 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022').
        client: AsyncAnthropic client instance.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model identifier to use for completions.
        """
        self.api_key = api_key
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using Anthropic's API.

        Args:
            request: Completion request with messages and parameters.

        Returns:
            CompletionResponse with generated content and usage statistics.

        Raises:
            Exception: If the API call fails.
        """
        # Anthropic requires separating system messages from user/assistant messages
        messages = request["messages"]
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append({"role": msg["role"], "content": msg["content"]})

        # Prepare the API call parameters
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": conversation_messages,
            "temperature": request["temperature"],
            "max_tokens": request.get("max_tokens", 4096),  # Anthropic requires max_tokens
        }

        if system_message:
            params["system"] = system_message

        # Make the API call
        response = await self.client.messages.create(**params)

        # Extract content
        content = ""
        if response.content:
            # Anthropic returns content as a list of content blocks
            content = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

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
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage
                else 0
            ),
        }

        return CompletionResponse(content=parsed_content, raw=content, usage=usage)

"""Azure OpenAI provider implementation."""

import json
from typing import Any, Dict, Optional
from openai import AsyncAzureOpenAI
from .base import LLMProvider
from ..types import CompletionRequest, CompletionResponse, TokenUsage


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI LLM provider implementation.

    This provider interfaces with Azure OpenAI Service to generate completions
    using GPT models hosted on Azure.

    Attributes:
        api_key: Azure OpenAI API key for authentication.
        model: Model identifier (e.g., 'gpt-4', 'gpt-35-turbo').
        endpoint: Azure OpenAI endpoint URL.
        api_version: API version to use.
        deployment: Optional deployment name (defaults to model name).
        client: AsyncAzureOpenAI client instance.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        endpoint: str,
        api_version: str = "2024-02-15-preview",
        deployment: Optional[str] = None,
    ) -> None:
        """
        Initialize the Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key.
            model: Model identifier to use for completions.
            endpoint: Azure OpenAI endpoint URL.
            api_version: API version to use (default: '2024-02-15-preview').
            deployment: Optional deployment name (defaults to model name).
        """
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.api_version = api_version
        self.deployment = deployment or model
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion using Azure OpenAI's API.

        Args:
            request: Completion request with messages and parameters.

        Returns:
            CompletionResponse with generated content and usage statistics.

        Raises:
            Exception: If the API call fails.
        """
        # Prepare the API call parameters
        params: Dict[str, Any] = {
            "model": self.deployment,  # Azure uses deployment name
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

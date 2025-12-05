"""Custom LLM provider for self-hosted models via HTTP API."""

import json
from typing import Optional
import aiohttp
from .base import LLMProvider
from ..types import CompletionRequest, CompletionResponse


class CustomLLMProvider(LLMProvider):
    """
    Provider for self-hosted LLM models accessed via HTTP API.
    
    Works with any LLM that accepts OpenAI-compatible API format,
    including models hosted with:
    - vLLM
    - Text Generation Inference (TGI)
    - LocalAI
    - Ollama (with OpenAI compatibility mode)
    - LM Studio
    
    Attributes:
        base_url: The base URL of your self-hosted LLM API.
        model: Model identifier (optional, depends on your API).
        api_key: API key if required (optional).
        timeout: Request timeout in seconds.
        
    Examples:
        >>> # vLLM server
        >>> provider = CustomLLMProvider(
        ...     base_url="http://localhost:8000/v1",
        ...     model="meta-llama/Llama-2-7b-chat-hf"
        ... )
        >>> 
        >>> # Ollama with OpenAI compatibility
        >>> provider = CustomLLMProvider(
        ...     base_url="http://localhost:11434/v1",
        ...     model="llama2"
        ... )
    """

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        """
        Initialize the custom LLM provider.

        Args:
            base_url: Base URL of the LLM API (e.g., "http://localhost:8000/v1").
            model: Model identifier (default: "default").
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds (default: 60).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Send completion request to self-hosted LLM.

        Args:
            request: Completion request with messages and parameters.

        Returns:
            CompletionResponse with generated content and token usage.

        Raises:
            Exception: If the API request fails.

        Examples:
            >>> request = CompletionRequest(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     temperature=0.7
            ... )
            >>> response = await provider.complete(request)
            >>> print(response["content"])
        """
        # Build OpenAI-compatible request payload
        payload = {
            "model": self.model,
            "messages": request["messages"],
            "temperature": request.get("temperature", 0.7),
            "stream": False,  # Explicitly disable streaming
        }

        # Add optional parameters
        if "max_tokens" in request:
            payload["max_tokens"] = request["max_tokens"]

        # Handle response_format for JSON mode
        if "response_format" in request:
            response_format = request["response_format"]
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                # Some APIs support this directly
                payload["response_format"] = response_format
                # Others might need a prompt instruction instead
                # Adjust based on your specific API

        # Set up headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Make async HTTP request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Extract content from OpenAI-compatible response
                    content = data["choices"][0]["message"]["content"]

                    # Extract token usage if available
                    usage = data.get("usage", {})
                    token_usage = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                    # Handle JSON response format if needed
                    if "response_format" in request:
                        content = self._extract_json_content(content)

                    return CompletionResponse(
                        content=content,
                        usage=token_usage,
                    )

            except aiohttp.ClientError as e:
                raise Exception(f"Custom LLM API request failed: {str(e)}")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise Exception(f"Failed to parse custom LLM response: {str(e)}")

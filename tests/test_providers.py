"""Tests for LLM provider implementations."""

import pytest
from src.maker_core.providers.base import LLMProvider
from src.maker_core.types import CompletionRequest, Message


class TestProviderInterface:
    """Test that providers implement the required interface."""
    
    def test_provider_is_abstract(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()
    
    
    def test_extract_json_content(self):
        """Test JSON extraction from markdown blocks."""
        from src.maker_core.providers.base import LLMProvider
        
        content_with_markdown = """```json
{
    "answer": "Paris",
    "confidence": 0.95
}
```"""
        
        result = LLMProvider._extract_json_content(None, content_with_markdown)
        assert '"answer"' in result
        assert '"Paris"' in result
        assert '```' not in result
        
        plain_json = '{"answer": "Paris"}'
        result = LLMProvider._extract_json_content(None, plain_json)
        assert result == plain_json


def test_openai_provider_exists():
    """Test that OpenAI provider can be imported."""
    from src.maker_core.providers import OpenAIProvider
    assert OpenAIProvider is not None


def test_anthropic_provider_exists():
    """Test that Anthropic provider can be imported."""
    from src.maker_core.providers import AnthropicProvider
    assert AnthropicProvider is not None


def test_azure_openai_provider_exists():
    """Test that Azure OpenAI provider can be imported."""
    from src.maker_core.providers import AzureOpenAIProvider
    assert AzureOpenAIProvider is not None

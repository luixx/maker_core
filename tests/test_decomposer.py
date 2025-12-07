"""Tests for the Decomposer class."""

import pytest
from src.maker_core.decomposer import Decomposer
from tests.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_decomposer_simple_question():
    """Test decomposer with a simple question that doesn't need decomposition."""
    classification_response = """
    {
        "needs_decomposition": false,
        "reasoning": "This is a simple factual question",
        "complexity_score": 2
    }
    """
    
    # Use "capital" as keyword since it's in the question
    provider = MockProvider({"capital": classification_response.strip()})
    decomposer = Decomposer(provider)
    
    result = await decomposer.decompose("What is the capital of France?")
    
    assert result["classification"]["needs_decomposition"] is False
    assert len(result["sub_questions"]) == 1
    assert result["sub_questions"][0]["question"] == "What is the capital of France?"


@pytest.mark.asyncio
async def test_decomposer_complex_question():
    """Test decomposer with a complex question requiring decomposition."""
    classification_response = """
    {
        "needs_decomposition": true,
        "reasoning": "This requires multiple steps",
        "complexity_score": 8
    }
    """
    
    decomposition_response = """
    {
        "sub_questions": [
            {
                "question": "What were the economic factors?",
                "reasoning": "Economic context is important"
            },
            {
                "question": "What were the political factors?",
                "reasoning": "Political context is important"
            },
            {
                "question": "What were the social factors?",
                "reasoning": "Social context is important"
            }
        ]
    }
    """
    
    # The MockProvider matches keywords in the user message.
    # For classification, the user message is "Question: What factors..."
    # For decomposition, the user message is "Original Question: What factors..."
    # Both contain "factors" and "Roman Empire", so we can use those
    provider = MockProvider({
        "Roman Empire": classification_response.strip(),
    })
    # Override the default response for decomposition since both calls match the same keyword
    # We need a smarter way to distinguish the calls
    
    # Alternative: Use a stateful mock
    call_count = [0]
    original_complete = provider.complete
    
    async def stateful_complete(request):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: classification
            provider.responses = {"Roman Empire": classification_response.strip()}
        else:
            # Second call: decomposition
            provider.responses = {"Roman Empire": decomposition_response.strip()}
        return await original_complete.__func__(provider, request)
    
    provider.complete = stateful_complete
    decomposer = Decomposer(provider, max_sub_questions=5)
    
    result = await decomposer.decompose(
        "What factors led to the fall of the Roman Empire?"
    )
    
    assert result["classification"]["needs_decomposition"] is True
    assert len(result["sub_questions"]) == 3
    assert "economic" in result["sub_questions"][0]["question"].lower()


@pytest.mark.asyncio
async def test_decomposer_respects_max_sub_questions():
    """Test that decomposer respects max_sub_questions limit."""
    classification_response = """
    {
        "needs_decomposition": true,
        "reasoning": "Complex question",
        "complexity_score": 9
    }
    """
    
    decomposition_response = """
    {
        "sub_questions": [
            {"question": "Q1?", "reasoning": "R1"},
            {"question": "Q2?", "reasoning": "R2"},
            {"question": "Q3?", "reasoning": "R3"},
            {"question": "Q4?", "reasoning": "R4"},
            {"question": "Q5?", "reasoning": "R5"},
            {"question": "Q6?", "reasoning": "R6"}
        ]
    }
    """
    
    # Stateful mock to return different responses
    call_count = [0]
    provider = MockProvider({"Complex": classification_response.strip()})
    original_complete = provider.complete
    
    async def stateful_complete(request):
        call_count[0] += 1
        if call_count[0] == 1:
            provider.responses = {"Complex": classification_response.strip()}
        else:
            provider.responses = {"Complex": decomposition_response.strip()}
        return await original_complete.__func__(provider, request)
    
    provider.complete = stateful_complete
    decomposer = Decomposer(provider, max_sub_questions=3)
    
    result = await decomposer.decompose("Complex question?")
    
    assert len(result["sub_questions"]) <= 3

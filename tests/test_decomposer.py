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
    
    provider = MockProvider({"complexity": classification_response.strip()})
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
    
    provider = MockProvider({
        "complexity": classification_response.strip(),
        "original question": decomposition_response.strip(),
    })
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
    
    provider = MockProvider({
        "complexity": classification_response.strip(),
        "original": decomposition_response.strip(),
    })
    decomposer = Decomposer(provider, max_sub_questions=3)
    
    result = await decomposer.decompose("Complex question?")
    
    assert len(result["sub_questions"]) <= 3

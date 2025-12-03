"""Tests for the Synthesizer class."""

import pytest
from src.maker_core.synthesizer import Synthesizer
from src.maker_core.types import SubQuestionResult
from tests.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_synthesizer_single_answer():
    """Test synthesizer with single sub-answer."""
    provider = MockProvider()
    synthesizer = Synthesizer(provider)
    
    sub_results = [
        SubQuestionResult(
            question="What is photosynthesis?",
            answer="Photosynthesis is the process by which plants convert light into energy.",
            vote_counts={"answer": 3},
            rounds_taken=3,
        )
    ]
    
    result = await synthesizer.synthesize(
        "What is photosynthesis?",
        sub_results
    )
    
    # Single answer should be returned directly
    assert "photosynthesis" in result["answer"].lower()
    assert "light" in result["answer"].lower()


@pytest.mark.asyncio
async def test_synthesizer_multiple_answers():
    """Test synthesizer combining multiple sub-answers."""
    synthesis_response = """
The Roman Empire fell due to a combination of factors. 
Economically, the empire faced inflation and taxation issues. 
Militarily, it struggled with barbarian invasions and internal conflicts. 
Politically, corruption and weak leadership undermined stability. 
These interconnected factors led to the empire's gradual decline.
"""
    
    provider = MockProvider({"original question": synthesis_response.strip()})
    synthesizer = Synthesizer(provider)
    
    sub_results = [
        SubQuestionResult(
            question="What were the economic factors?",
            answer="The Roman Empire faced severe inflation and taxation problems.",
            vote_counts={"answer": 3},
            rounds_taken=3,
        ),
        SubQuestionResult(
            question="What were the military factors?",
            answer="Barbarian invasions and internal military conflicts weakened Rome.",
            vote_counts={"answer": 3},
            rounds_taken=3,
        ),
        SubQuestionResult(
            question="What were the political factors?",
            answer="Corruption and weak political leadership destabilized the empire.",
            vote_counts={"answer": 3},
            rounds_taken=3,
        ),
    ]
    
    result = await synthesizer.synthesize(
        "What factors led to the fall of the Roman Empire?",
        sub_results
    )
    
    answer = result["answer"].lower()
    assert len(answer) > 50
    assert "economic" in answer or "inflation" in answer
    assert "military" in answer or "barbarian" in answer or "invasion" in answer


@pytest.mark.asyncio
async def test_synthesizer_fallback():
    """Test synthesizer fallback when LLM fails."""
    class FailingProvider(MockProvider):
        async def complete(self, request):
            raise Exception("API failure")
    
    provider = FailingProvider()
    synthesizer = Synthesizer(provider)
    
    sub_results = [
        SubQuestionResult(
            question="Q1?",
            answer="Answer 1",
            vote_counts={"A1": 3},
            rounds_taken=3,
        ),
        SubQuestionResult(
            question="Q2?",
            answer="Answer 2",
            vote_counts={"A2": 3},
            rounds_taken=3,
        ),
    ]
    
    result = await synthesizer.synthesize("Original question?", sub_results)
    
    assert "answer 1" in result["answer"].lower()
    assert "answer 2" in result["answer"].lower()

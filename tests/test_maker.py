"""Tests for the main Maker class."""

import pytest
from src.maker_core.maker import Maker, MakerSync
from src.maker_core.types import Confidence
from tests.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_maker_simple_question():
    """Test Maker with a simple question."""
    responses = {
        "complexity": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 2}',
        "capital": "Paris",
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    result = await maker.ask("What is the capital of France?")
    
    assert "answer" in result
    assert result["confidence"] in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]
    assert result["decomposition_used"] is False


@pytest.mark.asyncio
async def test_maker_complex_question():
    """Test Maker with a complex question requiring decomposition."""
    responses = {
        "complexity": '{"needs_decomposition": true, "reasoning": "Complex", "complexity_score": 8}',
        "original question": '''
        {
            "sub_questions": [
                {"question": "What is the economic impact?", "reasoning": "Economic"},
                {"question": "What is the social impact?", "reasoning": "Social"}
            ]
        }
        ''',
        "economic": "The economic impact includes inflation and trade issues.",
        "social": "The social impact includes population decline and unrest.",
        "factors led": "Multiple interconnected factors led to the decline.",
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    result = await maker.ask("What factors led to the fall of Rome?")
    
    assert "answer" in result
    assert result["decomposition_used"] is True
    assert len(result["sub_results"]) == 2


@pytest.mark.asyncio
async def test_maker_events():
    """Test Maker event emissions."""
    responses = {
        "complexity": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 2}',
        "": "Test answer",
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    events_received = []
    
    def event_handler(data):
        events_received.append(data)
    
    maker.on("complete", event_handler)
    
    result = await maker.ask("Test question?")
    
    assert len(events_received) > 0
    assert result["answer"] == "Test answer"


def test_maker_sync():
    """Test synchronous Maker wrapper."""
    responses = {
        "complexity": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 1}',
        "": "Sync answer",
    }
    provider = MockProvider(responses)
    maker = MakerSync(provider)
    
    result = maker.ask("Sync test question?")
    
    assert "answer" in result
    assert result["answer"] == "Sync answer"


@pytest.mark.asyncio
async def test_maker_confidence_calculation():
    """Test confidence calculation based on voting performance."""
    responses = {
        "complexity": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 2}',
        "": "Quick answer",
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    result = await maker.ask("Test?")
    
    assert result["confidence"] in [Confidence.HIGH, Confidence.MEDIUM]

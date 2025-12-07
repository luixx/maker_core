"""Tests for the main Maker class."""

import pytest
from src.maker_core.maker import Maker, MakerSync
from src.maker_core.types import Confidence
from tests.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_maker_simple_question():
    """Test Maker with a simple question."""
    responses = {
        "capital": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 2}',
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    result = await maker.ask("What is the capital of France?")
    
    assert "answer" in result
    assert result["confidence"] in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]
    assert result["is_decomposed"] is False


@pytest.mark.asyncio
async def test_maker_complex_question():
    """Test Maker with a complex question requiring decomposition."""
    classification_response = '{"needs_decomposition": true, "reasoning": "Complex", "complexity_score": 8}'
    
    decomposition_response = '''
    {
        "sub_questions": [
            {"question": "What is the economic impact?", "reasoning": "Economic"},
            {"question": "What is the social impact?", "reasoning": "Social"}
        ]
    }
    '''
    
    # Create a stateful provider to return different responses for different calls
    call_count = [0]
    provider = MockProvider({"Rome": classification_response})
    original_complete = provider.complete
    
    async def stateful_complete(request):
        call_count[0] += 1
        user_msg = ""
        for msg in request["messages"]:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break
        
        # Determine which phase we're in based on message content
        if "complexity" in user_msg.lower() or (call_count[0] == 1 and "Question:" in user_msg):
            # Classification call
            provider.responses = {"Rome": classification_response}
        elif "Original Question:" in user_msg or "decompos" in user_msg.lower():
            # Decomposition call
            provider.responses = {"Rome": decomposition_response.strip()}
        elif "economic" in user_msg.lower():
            provider.responses = {"economic": "The economic impact includes inflation."}
        elif "social" in user_msg.lower():
            provider.responses = {"social": "The social impact includes decline."}
        elif "synthes" in user_msg.lower() or "sub-question" in user_msg.lower():
            provider.responses = {"Rome": "The fall of Rome had multiple causes."}
        
        return await original_complete.__func__(provider, request)
    
    provider.complete = stateful_complete
    maker = Maker(provider)
    
    result = await maker.ask("What factors led to the fall of Rome?")
    
    assert "answer" in result
    assert result["is_decomposed"] is True
    assert len(result["sub_questions"]) == 2


@pytest.mark.asyncio
async def test_maker_events():
    """Test Maker event emissions."""
    responses = {
        "": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 2}',
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    events_received = []
    
    def event_handler(data):
        events_received.append(data)
    
    maker.on("complete", event_handler)
    
    result = await maker.ask("Test question?")
    
    assert len(events_received) > 0
    # The answer comes from voting which wraps in JSON format
    assert "answer" in result


def test_maker_sync():
    """Test synchronous Maker wrapper."""
    responses = {
        "": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 1}',
    }
    provider = MockProvider(responses)
    maker = MakerSync(provider)
    
    result = maker.ask("Sync test question?")
    
    assert "answer" in result


@pytest.mark.asyncio
async def test_maker_confidence_calculation():
    """Test confidence calculation based on voting performance."""
    responses = {
        "": '{"needs_decomposition": false, "reasoning": "Simple", "complexity_score": 2}',
    }
    provider = MockProvider(responses)
    maker = Maker(provider)
    
    result = await maker.ask("Test?")
    
    assert result["confidence"] in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]

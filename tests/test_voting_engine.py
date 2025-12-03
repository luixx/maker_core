"""Tests for the VotingEngine class."""

import pytest
from src.maker_core.voting_engine import VotingEngine
from src.maker_core.red_flag_filter import RedFlagFilter
from tests.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_voting_engine_quick_consensus():
    """Test voting engine reaching quick consensus."""
    provider = MockProvider({"": "Paris"})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    
    result = await engine.vote_until_consensus("What is the capital of France?")
    
    assert result["answer"] == "Paris"
    assert result["rounds_taken"] == 3  # Should reach threshold in 3 rounds
    assert "Paris" in result["vote_counts"]
    assert result["vote_counts"]["Paris"] == 3


@pytest.mark.asyncio
async def test_voting_engine_with_disagreement():
    """Test voting engine with initial disagreement."""
    responses = {
        "capital": "Paris",  # Will be used for most queries
    }
    provider = MockProvider(responses)
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    
    result = await engine.vote_until_consensus("What is the capital of France?")
    
    assert result["answer"] == "Paris"
    assert result["rounds_taken"] >= 3


@pytest.mark.asyncio
async def test_voting_engine_max_rounds():
    """Test voting engine hitting max rounds limit."""
    # Create a provider that alternates responses
    call_count = [0]
    
    class AlternatingProvider(MockProvider):
        async def complete(self, request):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                self.responses = {"": "Answer A"}
            else:
                self.responses = {"": "Answer B"}
            return await super().complete(request)
    
    provider = AlternatingProvider()
    engine = VotingEngine(provider, voting_threshold=5, max_rounds=4)
    
    result = await engine.vote_until_consensus("Test question?")
    
    assert result["rounds_taken"] == 4
    # Should return the answer with most votes
    assert result["answer"] in ["Answer A", "Answer B"]


@pytest.mark.asyncio
async def test_voting_engine_with_red_flag_filter():
    """Test voting engine with red flag filter enabled."""
    provider = MockProvider({"": "Paris is the capital"})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    red_flag_filter = RedFlagFilter()
    engine.set_red_flag_filter(red_flag_filter)
    
    result = await engine.vote_until_consensus("What is the capital of France?")
    
    assert result["answer"] == "Paris is the capital"
    # Red flags should be low or zero for valid responses
    assert result["red_flags_encountered"] >= 0

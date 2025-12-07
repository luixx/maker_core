"""Tests for the VotingEngine class."""

import pytest
from src.maker_core.voting_engine import VotingEngine
from src.maker_core.red_flag_filter import RedFlagFilter
from tests.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_voting_engine_quick_consensus():
    """Test voting engine reaching quick consensus with single answer."""
    provider = MockProvider({"": "Paris"})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    
    result = await engine.vote_until_consensus("What is the capital of France?")
    
    assert result["answer"] == "Paris"
    assert result["consensus_reached"] == True
    # With K=3 and only one answer, needs 3 total votes
    assert result["rounds_taken"] == 3
    # Check normalized key is in vote counts (lowercase)
    assert "paris" in result["vote_counts"]
    assert result["vote_counts"]["paris"] == 3


@pytest.mark.asyncio
async def test_voting_engine_consensus_with_k_ahead():
    """Test first-to-ahead-by-K algorithm with multiple answers."""
    call_count = [0]
    
    class MixedProvider(MockProvider):
        """Provider that returns different answers."""
        async def complete(self, request):
            call_count[0] += 1
            # First vote: Answer A
            # Second vote: Answer B
            # Then mostly Answer A to get K ahead
            if call_count[0] == 2:
                self.responses = {"": "Answer B"}
            else:
                self.responses = {"": "Answer A"}
            return await super().complete(request)
    
    provider = MixedProvider()
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=20)
    
    result = await engine.vote_until_consensus("Test question?")
    
    # With K=3, need Answer A to be 3 ahead of Answer B
    # If A:4, B:1, then 4-1=3 >= K, so consensus
    assert result["answer"] == "Answer A"
    assert result["consensus_reached"] == True


@pytest.mark.asyncio
async def test_voting_engine_max_rounds():
    """Test voting engine hitting max rounds limit without consensus."""
    call_count = [0]
    
    class AlternatingProvider(MockProvider):
        """Provider that strictly alternates between two answers."""
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
    assert result["consensus_reached"] == False
    # Should return the answer with most votes (or tie)
    assert result["answer"] in ["Answer A", "Answer B"]


@pytest.mark.asyncio
async def test_voting_engine_with_red_flag_filter():
    """Test voting engine with red flag filter enabled."""
    provider = MockProvider({"": "Paris is the capital"})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    red_flag_filter = RedFlagFilter()
    engine.set_red_flag_filter(red_flag_filter)
    
    result = await engine.vote_until_consensus("What is the capital of France?")
    
    assert "capital" in result["answer"].lower() or "paris" in result["answer"].lower()
    assert result["consensus_reached"] == True
    # Red flags should be low or zero for valid responses
    assert result["red_flags_encountered"] >= 0


@pytest.mark.asyncio
async def test_voting_engine_red_flags_too_long():
    """Test that overly long responses are red-flagged."""
    # Create a response that exceeds 750 tokens (roughly 3000 chars)
    long_answer = "This is a very long answer. " * 200
    provider = MockProvider({"": long_answer})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=5)
    red_flag_filter = RedFlagFilter()
    engine.set_red_flag_filter(red_flag_filter)
    
    result = await engine.vote_until_consensus("Test question?")
    
    # All responses should be red-flagged
    assert result["red_flags_encountered"] > 0
    assert result["consensus_reached"] == False


@pytest.mark.asyncio
async def test_voting_engine_normalization():
    """Test that answers are normalized for comparison."""
    call_count = [0]
    
    class VariedCaseProvider(MockProvider):
        """Provider that returns same answer with different casing."""
        async def complete(self, request):
            call_count[0] += 1
            variations = ["Paris", "PARIS", "paris", "Paris.", "Paris!"]
            self.responses = {"": variations[call_count[0] % len(variations)]}
            return await super().complete(request)
    
    provider = VariedCaseProvider()
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    
    result = await engine.vote_until_consensus("What is the capital of France?")
    
    # All variations should normalize to "paris" and count as same answer
    assert result["consensus_reached"] == True
    assert "paris" in result["vote_counts"]
    assert result["vote_counts"]["paris"] >= 3


@pytest.mark.asyncio
async def test_voting_engine_tracks_valid_votes():
    """Test that valid_votes is correctly tracked separately from rounds."""
    provider = MockProvider({"": "Valid answer"})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    
    result = await engine.vote_until_consensus("Test question?")
    
    assert result["valid_votes"] == result["rounds_taken"]
    assert result["red_flags_encountered"] == 0


@pytest.mark.asyncio
async def test_voting_engine_event_emission():
    """Test that voting progress events are emitted."""
    provider = MockProvider({"": "Answer"})
    engine = VotingEngine(provider, voting_threshold=3, max_rounds=10)
    
    events = []
    def on_progress(data):
        events.append(data)
    
    engine.on("voteProgress", on_progress)
    
    result = await engine.vote_until_consensus("Test question?")
    
    assert len(events) >= 3  # At least 3 events for K=3
    assert "round" in events[0]
    assert "vote_counts" in events[0]

"""Voting engine implementing first-to-ahead-by-K consensus algorithm."""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Any
from .providers.base import LLMProvider
from .red_flag_filter import RedFlagFilter
from .types import VotingResult, CompletionRequest, Message


class VotingEngine:
    """
    Implements the first-to-ahead-by-K voting algorithm from the MAKER paper.

    The voting engine continuously samples LLM responses until one answer
    is K votes ahead of all others, ensuring consensus and reliability.

    Attributes:
        provider: LLM provider for generating answers.
        voting_threshold: Number of votes the winner must be ahead by (K).
        max_rounds: Safety limit to prevent infinite voting.
        red_flag_filter: Optional filter for unreliable responses.
    """

    def __init__(
        self,
        provider: LLMProvider,
        voting_threshold: int = 3,
        max_rounds: int = 10,
    ) -> None:
        """
        Initialize the voting engine.

        Args:
            provider: LLM provider to use for voting.
            voting_threshold: Votes ahead needed for consensus (K).
            max_rounds: Maximum voting rounds before stopping.
        """
        self.provider = provider
        self.voting_threshold = voting_threshold
        self.max_rounds = max_rounds
        self.red_flag_filter: Optional[RedFlagFilter] = None
        self._listeners: Dict[str, List[Callable]] = {}

    def set_red_flag_filter(self, filter: RedFlagFilter) -> None:
        """
        Set the red flag filter for detecting unreliable responses.

        Args:
            filter: RedFlagFilter instance.
        """
        self.red_flag_filter = filter

    def on(self, event: str, listener: Callable) -> None:
        """Register an event listener."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event."""
        if event in self._listeners:
            for listener in self._listeners[event]:
                try:
                    listener(data)
                except Exception:
                    pass

    def remove_listener(self, event: str, listener: Callable) -> None:
        """Remove an event listener."""
        if event in self._listeners and listener in self._listeners[event]:
            self._listeners[event].remove(listener)

    async def vote_until_consensus(
        self, question: str, context: str = ""
    ) -> VotingResult:
        """
        Vote on a question until consensus is reached.

        Uses the first-to-ahead-by-K algorithm: continues voting until
        one answer is K votes ahead of all others.

        Args:
            question: Question to answer.
            context: Optional context information.

        Returns:
            VotingResult with consensus answer, vote counts, and statistics.
        """
        vote_counts: Dict[str, int] = defaultdict(int)
        red_flags_count = 0
        rounds = 0

        while rounds < self.max_rounds:
            rounds += 1

            # Sample an answer
            temperature = 0.0 if rounds == 1 else 0.1
            answer = await self._sample_answer(question, context, temperature)

            # Check for red flags
            if self.red_flag_filter and answer:
                check_result = self.red_flag_filter.check(answer)
                if check_result["is_flagged"]:
                    red_flags_count += 1
                    continue  # Skip this answer

            # Count the vote
            if answer:
                vote_counts[answer] += 1

            # Emit progress event
            self.emit(
                "voteProgress",
                {"round": rounds, "vote_counts": dict(vote_counts)},
            )

            # Check for consensus
            if self._check_consensus(vote_counts):
                break

        # Determine winner
        if vote_counts:
            winner = max(vote_counts.items(), key=lambda x: x[1])
            answer = winner[0]
        else:
            answer = "Unable to reach consensus"

        return VotingResult(
            answer=answer,
            vote_counts=dict(vote_counts),
            rounds_taken=rounds,
            red_flags_encountered=red_flags_count,
        )

    async def _sample_answer(
        self, question: str, context: str, temperature: float
    ) -> str:
        """
        Sample an answer from the LLM.

        Args:
            question: Question to answer.
            context: Context information.
            temperature: Sampling temperature.

        Returns:
            Answer string.
        """
        prompt = question
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"

        messages: List[Message] = [
            {"role": "user", "content": prompt}
        ]

        request = CompletionRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=500,
        )

        try:
            response = await self.provider.complete(request)
            return response["content"].strip()
        except Exception:
            return ""

    def _check_consensus(self, vote_counts: Dict[str, int]) -> bool:
        """
        Check if consensus has been reached.

        Consensus is reached when the leading answer is K votes ahead
        of the second-place answer.

        Args:
            vote_counts: Current vote counts.

        Returns:
            True if consensus reached, False otherwise.
        """
        if len(vote_counts) < 1:
            return False

        # Get sorted vote counts
        sorted_counts = sorted(vote_counts.values(), reverse=True)

        # Check if leader is K ahead
        if len(sorted_counts) == 1:
            return sorted_counts[0] >= self.voting_threshold
        else:
            lead = sorted_counts[0] - sorted_counts[1]
            return lead >= self.voting_threshold

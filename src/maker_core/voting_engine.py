"""Voting engine implementing first-to-ahead-by-K consensus (Algorithm 2)."""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

from .types import (
    VotingResult,
    Message,
    CompletionRequest,
)
from .red_flag_filter import RedFlagFilter


@dataclass
class VoteData:
    """Data associated with a vote for a particular answer."""
    count: int = 0
    original_answer: str = ""
    confidence: str = "medium"


class VotingEngine:
    """
    Voting engine implementing the do_voting algorithm from the MAKER paper.

    From Algorithm 2:
    ```
    while True:
        y = get_vote(x, M)
        V[y] = V[y] + 1
        if V[y] >= k + max(V[v] for v != y):
            return y
    ```

    The algorithm continuously samples until one answer is K votes ahead
    of all other answers. Red-flagged votes are discarded (get_vote).
    """

    def __init__(
        self,
        provider: Any,
        k: int = None,
        voting_threshold: int = None,  # Alias for k (backwards compatibility)
        max_rounds: int = 10,
        red_flag_filter: Optional[RedFlagFilter] = None,
    ):
        """
        Initialize voting engine.

        Args:
            provider: LLM provider for generating answers.
            k: The K in first-to-ahead-by-K voting (default 2).
            voting_threshold: Alias for k (for backwards compatibility).
            max_rounds: Maximum voting rounds before giving up.
            red_flag_filter: Optional filter for detecting bad answers.
        """
        self.provider = provider
        # Support both 'k' and 'voting_threshold' parameter names
        self.k = k if k is not None else (voting_threshold if voting_threshold is not None else 2)
        self.max_rounds = max_rounds
        self.red_flag_filter = red_flag_filter
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable) -> None:
        """Register an event listener."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def emit(self, event: str, data: dict) -> None:
        """Emit an event to all listeners."""
        for callback in self._listeners.get(event, []):
            try:
                callback(data)
            except Exception:
                pass

    def _check_consensus(self, vote_counts: Dict[str, VoteData]) -> bool:
        """
        Check if we have reached consensus (first-to-ahead-by-K).

        From Algorithm 2: V[y] >= k + max(V[v] for v != y)

        This means the winner must be K votes AHEAD of the runner-up.

        Args:
            vote_counts: Dictionary of normalized answers to their vote data.

        Returns:
            True if consensus is reached, False otherwise.
        """
        if not vote_counts:
            return False

        counts = [v.count for v in vote_counts.values()]
        if not counts:
            return False

        max_count = max(counts)

        if len(counts) == 1:
            # Only one unique answer - need at least K total votes
            return max_count >= self.k

        # Find runner-up count (second highest)
        sorted_counts = sorted(counts, reverse=True)
        runner_up_count = sorted_counts[1]

        # Winner must be K votes AHEAD of runner-up
        return max_count >= self.k + runner_up_count

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.

        Makes comparison case-insensitive and removes trivial differences.
        """
        if not answer:
            return ""
        normalized = answer.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[.,!?;:]+$', '', normalized)
        return normalized

    async def vote_until_consensus(
        self, question: str, context: str = ""
    ) -> VotingResult:
        """
        Vote on a question until consensus is reached (Algorithm 2 from MAKER paper).

        Implements do_voting from the paper:
        ```
        while True:
            y = get_vote(x, M)
            V[y] = V[y] + 1
            if V[y] >= k + max(V[v] for v != y):
                return y
        ```

        Continuously samples until one answer is K votes ahead of all others.
        Red-flagged votes are discarded and not counted.

        Args:
            question: Question to answer.
            context: Optional context information.

        Returns:
            VotingResult with consensus answer, vote counts, and statistics.
        """
        vote_counts: Dict[str, VoteData] = {}
        all_votes: List[dict] = []
        valid_vote_count = 0
        vote_index = 0

        while True:
            # Get temperature: first vote at 0, rest at 0.1 (per paper)
            temperature = 0.0 if vote_index == 0 else 0.1

            # Sample an answer
            answer, parse_succeeded, confidence = await self._sample_answer_with_metadata(
                question, context, temperature
            )
            
            # Emit vote event for debugging
            self.emit("vote", {
                "round": vote_index + 1,
                "answer": answer,
                "temperature": temperature,
                "parse_succeeded": parse_succeeded,
            })
            
            vote_data = {
                "vote_index": vote_index,
                "answer": answer,
                "temperature": temperature,
                "parse_succeeded": parse_succeeded,
                "red_flagged": False,
                "red_flag_reason": None,
            }
            all_votes.append(vote_data)
            vote_index += 1

            # Apply red-flag filter (Algorithm 3: get_vote)
            if self.red_flag_filter and answer:
                check_result = self.red_flag_filter.check(answer, parse_succeeded)
                if check_result["is_flagged"]:
                    vote_data["red_flagged"] = True
                    vote_data["red_flag_reason"] = check_result["reason"]
                    self.emit("voteProgress", {
                        "round": vote_index,
                        "vote_counts": {k: v.count for k, v in vote_counts.items()},
                        "red_flagged": True,
                    })
                    
                    # Check safety limit
                    if len(all_votes) >= self.max_rounds:
                        break
                    continue  # Discard and resample

            # Count valid vote
            valid_vote_count += 1
            normalized = self._normalize_answer(answer)

            if not normalized:
                # Empty answer after normalization - skip
                self.emit("voteProgress", {
                    "round": vote_index,
                    "vote_counts": {k: v.count for k, v in vote_counts.items()},
                })
                if len(all_votes) >= self.max_rounds:
                    break
                continue

            # Update vote counts
            if normalized in vote_counts:
                vote_counts[normalized].count += 1
                # Keep higher confidence
                if self._compare_confidence(confidence, vote_counts[normalized].confidence) > 0:
                    vote_counts[normalized].confidence = confidence
            else:
                vote_counts[normalized] = VoteData(
                    count=1,
                    original_answer=answer,
                    confidence=confidence,
                )

            # Emit progress event
            self.emit("voteProgress", {
                "round": vote_index,
                "vote_counts": {k: v.count for k, v in vote_counts.items()},
            })

            # Check first-to-ahead-by-K condition
            if self._check_consensus(vote_counts):
                # Find winner
                max_count = max(v.count for v in vote_counts.values())
                for key, data in vote_counts.items():
                    if data.count == max_count:
                        counts = sorted([v.count for v in vote_counts.values()], reverse=True)
                        runner_up = counts[1] if len(counts) > 1 else 0
                        return VotingResult(
                            answer=data.original_answer,
                            consensus_reached=True,
                            vote_counts={k: v.count for k, v in vote_counts.items()},
                            rounds_taken=len(all_votes),
                            valid_votes=valid_vote_count,
                            red_flags_encountered=len(all_votes) - valid_vote_count,
                            winning_vote_count=max_count,
                            margin=max_count - runner_up,
                        )

            # Safety limit to prevent infinite loops
            if len(all_votes) >= self.max_rounds:
                break

        # Return best guess even without consensus
        if vote_counts:
            sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1].count, reverse=True)
            winner_key, winner_data = sorted_votes[0]
            runner_up_count = sorted_votes[1][1].count if len(sorted_votes) > 1 else 0
            
            return VotingResult(
                answer=winner_data.original_answer,
                consensus_reached=False,
                vote_counts={k: v.count for k, v in vote_counts.items()},
                rounds_taken=len(all_votes),
                valid_votes=valid_vote_count,
                red_flags_encountered=len(all_votes) - valid_vote_count,
                winning_vote_count=winner_data.count,
                margin=winner_data.count - runner_up_count,
            )
        
        return VotingResult(
            answer="Unable to reach consensus",
            consensus_reached=False,
            vote_counts={},
            rounds_taken=len(all_votes),
            valid_votes=0,
            red_flags_encountered=len(all_votes),
            winning_vote_count=0,
            margin=0,
        )

    async def _sample_answer_with_metadata(
        self, question: str, context: str, temperature: float
    ) -> Tuple[str, bool, str]:
        """
        Sample an answer from the LLM with parse success and confidence metadata.

        This method is robust to both JSON and plain text responses from LLMs.

        Args:
            question: Question to answer.
            context: Context information.
            temperature: Sampling temperature.

        Returns:
            Tuple of (answer, parse_succeeded, confidence).
        """
        # Build a simple, clear prompt that works with most LLMs
        prompt = f"""Answer the following question concisely.

If possible, respond in JSON format:
{{"answer": "your answer", "confidence": "high/medium/low"}}

Otherwise, just give a direct answer.

"""
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Question: {question}"

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
            content = response["content"].strip()
            
            # Try to parse as JSON first
            answer, parsed, confidence = self._try_parse_json_answer(content)
            if parsed:
                return answer, True, confidence
            
            # Try to extract JSON from mixed content
            answer, parsed, confidence = self._try_extract_json(content)
            if parsed:
                return answer, True, confidence
            
            # Fall back to plain text - use the whole response as the answer
            # This is still valid, just mark parse_succeeded as False
            clean_answer = self._clean_plain_text_answer(content)
            if clean_answer:
                return clean_answer, False, "medium"
            
            return content, False, "low"
            
        except Exception as e:
            return "", False, "low"

    def _try_parse_json_answer(self, content: str) -> Tuple[str, bool, str]:
        """Try to parse content as a JSON response."""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = str(parsed.get("answer", ""))
                confidence = parsed.get("confidence", "medium")
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                return answer, True, confidence
        except json.JSONDecodeError:
            pass
        return "", False, "low"
    
    def _try_extract_json(self, content: str) -> Tuple[str, bool, str]:
        """Try to extract JSON from content that may have other text."""
        # Look for JSON object in the content
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*[^{}]+\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = str(parsed.get("answer", ""))
                    confidence = parsed.get("confidence", "medium")
                    if confidence not in ("high", "medium", "low"):
                        confidence = "medium"
                    return answer, True, confidence
            except json.JSONDecodeError:
                pass
        return "", False, "low"
    
    def _clean_plain_text_answer(self, content: str) -> str:
        """Clean a plain text answer, removing common prefixes/wrappers."""
        answer = content.strip()
        
        # Remove common answer prefixes
        prefixes = [
            "Answer:", "The answer is:", "The answer is", 
            "Response:", "Here's the answer:",
            "Based on the context,", "Based on the document,",
        ]
        lower_answer = answer.lower()
        for prefix in prefixes:
            if lower_answer.startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break
        
        # If the response is very long, it might be an explanation
        # Try to get just the first sentence for simple questions
        if len(answer) > 500:
            first_sentence = re.split(r'[.!?]\s', answer, maxsplit=1)[0]
            if len(first_sentence) > 10:
                answer = first_sentence
        
        return answer

    def _compare_confidence(self, a: str, b: str) -> int:
        """
        Compare confidence levels.
        Returns positive if a > b, negative if a < b, 0 if equal.
        """
        order = {"high": 3, "medium": 2, "low": 1}
        return order.get(a, 0) - order.get(b, 0)

    def set_red_flag_filter(self, red_flag_filter: Optional[RedFlagFilter]) -> None:
        """Set the red flag filter after initialization."""
        self.red_flag_filter = red_flag_filter

    def remove_listener(self, event: str, listener: Callable) -> None:
        """Remove an event listener."""
        if event in self._listeners:
            try:
                self._listeners[event].remove(listener)
            except ValueError:
                pass  # Listener not found, ignore

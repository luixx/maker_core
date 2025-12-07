"""Main Maker class implementing the full MAKER pipeline with event support."""

import asyncio
import time
from typing import Callable, Dict, List, Optional, Any
from .providers.base import LLMProvider
from .decomposer import Decomposer
from .voting_engine import VotingEngine
from .synthesizer import Synthesizer
from .red_flag_filter import RedFlagFilter
from .types import (
    MakerConfig,
    AskOptions,
    MakerResult,
    SubQuestionResult,
    VotingStats,
    Confidence,
)


class EventEmitter:
    """
    Simple event emitter for progress tracking.

    Allows registering listeners for various events during the MAKER pipeline
    execution, enabling progress monitoring and debugging.
    """

    def __init__(self) -> None:
        """Initialize the event emitter with empty listener registry."""
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event: str, listener: Callable) -> None:
        """
        Register an event listener.

        Args:
            event: Event name to listen for.
            listener: Callback function to invoke when event is emitted.

        Examples:
            >>> emitter = EventEmitter()
            >>> emitter.on("progress", lambda data: print(data))
        """
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def emit(self, event: str, data: Any = None) -> None:
        """
        Emit an event with optional data.

        Calls all registered listeners for the event with the provided data.

        Args:
            event: Event name to emit.
            data: Optional data to pass to listeners.
        """
        if event in self._listeners:
            for listener in self._listeners[event]:
                try:
                    listener(data)
                except Exception:
                    # Silently ignore listener errors to prevent disruption
                    pass

    def remove_listener(self, event: str, listener: Callable) -> None:
        """
        Remove a specific event listener.

        Args:
            event: Event name.
            listener: Listener function to remove.
        """
        if event in self._listeners and listener in self._listeners[event]:
            self._listeners[event].remove(listener)


class Maker(EventEmitter):
    """
    Main MAKER orchestration class implementing the complete pipeline.

    Coordinates decomposition, voting, and synthesis to answer complex
    questions with high reliability. Extends EventEmitter to provide
    progress tracking.

    The pipeline:
    1. Classification: Determine if decomposition is needed
    2. Decomposition: Break complex questions into sub-questions (if needed)
    3. Voting: Use consensus voting for each sub-question
    4. Synthesis: Combine sub-answers into final coherent answer

    Events emitted:
    - classificationComplete: After question classification
    - decomposed: After decomposition into sub-questions
    - votingStart: Before voting starts for a sub-question
    - voteProgress: Each vote during continuous voting
    - votingComplete: After voting ends for a sub-question
    - redFlagged: When a vote is red-flagged
    - synthesisStart: Before synthesis
    - synthesisComplete: After synthesis
    - complete: Processing complete

    Attributes:
        config: Configuration for the MAKER pipeline.
        provider: Primary LLM provider for decomposition and synthesis.
        decomposer: Question decomposer instance.
        voting_engine: Voting engine for consensus.
        synthesizer: Answer synthesizer instance.
        red_flag_filter: Filter for detecting unreliable responses.

    Examples:
        >>> from maker_core import Maker
        >>> from maker_core.providers import OpenAIProvider
        >>> 
        >>> provider = OpenAIProvider(api_key="your-key", model="gpt-4")
        >>> maker = Maker(provider=provider)
        >>> 
        >>> # Register progress listener
        >>> maker.on("votingProgress", lambda data: print(f"Vote {data['round']}"))
        >>> 
        >>> # Ask a question
        >>> result = await maker.ask("What factors led to the fall of Rome?")
        >>> print(result["answer"])
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[MakerConfig] = None,
    ) -> None:
        """
        Initialize the Maker instance.

        Args:
            provider: Primary LLM provider for decomposition and synthesis.
            config: Optional configuration (uses defaults if not provided).
        """
        super().__init__()
        self.provider = provider
        self.config = config or {}

        # Get configuration values with defaults
        # Support both nested config (voting.k) and flat config (voting_threshold)
        voting_config = self.config.get("voting", {})
        k = voting_config.get("k", self.config.get("voting_threshold", 3))
        max_votes = voting_config.get("max_votes", self.config.get("max_voting_rounds", 100))
        
        decomposition_config = self.config.get("decomposition", {})
        max_sub_questions = decomposition_config.get(
            "max_sub_questions", 
            self.config.get("max_sub_questions", 5)
        )
        
        red_flag_config = self.config.get("red_flags", {})
        enable_red_flag = self.config.get("enable_red_flag_filter", True)

        # Initialize components
        self.decomposer = Decomposer(
            provider=provider,
            max_sub_questions=max_sub_questions,
        )
        self.voting_engine = VotingEngine(
            provider=provider,
            voting_threshold=k,
            max_rounds=max_votes,
        )
        self.synthesizer = Synthesizer(provider=provider)
        self.red_flag_filter = RedFlagFilter(config=red_flag_config if red_flag_config else None)

        # Enable red flag filter if configured
        if enable_red_flag:
            self.voting_engine.set_red_flag_filter(self.red_flag_filter)
        
        # Store K for stats
        self._k = k

    async def ask(
        self,
        question: str,
        options: Optional[AskOptions] = None,
    ) -> MakerResult:
        """
        Ask a question and get a reliable answer using the MAKER pipeline.

        This is the main entry point for using MAKER. It orchestrates the full
        pipeline: classification, decomposition, voting, and synthesis.

        Args:
            question: The question to answer.
            options: Optional configuration overrides.

        Returns:
            MakerResult with answer, confidence, and metadata.

        Raises:
            Exception: If the pipeline encounters an unrecoverable error.

        Examples:
            >>> result = await maker.ask(
            ...     "What are the main causes of climate change?",
            ...     options={"k": 5}
            ... )
            >>> print(f"Answer: {result['answer']}")
            >>> print(f"Confidence: {result['confidence']}")
        """
        start_time = time.time()
        opts = options or {}
        context = opts.get("context", "")

        # Step 1: Decompose the question
        self.emit("decompositionStart", {"question": question})
        decomposition_result = await self.decomposer.decompose(question, context)

        classification = decomposition_result["classification"]
        self.emit("classificationComplete", classification)
        
        self.emit(
            "decomposed",
            decomposition_result["sub_questions"],
        )

        # Step 2: Answer each sub-question using voting (Algorithm 2: do_voting)
        sub_results: List[SubQuestionResult] = []

        for i, sub_q in enumerate(decomposition_result["sub_questions"]):
            self.emit("votingStart", {
                "subQuestionIndex": i,
                "question": sub_q["question"],
            })

            # Set up voting progress callback
            def voting_progress_handler(data: Dict) -> None:
                self.emit("voteProgress", {
                    "subQuestionIndex": i,
                    "voteIndex": data.get("round", 0),
                    "voteCounts": data.get("vote_counts", {}),
                    "redFlagged": data.get("red_flagged", False),
                })
                
                # Emit red-flagged event if applicable
                if data.get("red_flagged", False):
                    self.emit("redFlagged", {
                        "answer": "",  # Answer discarded
                        "reason": "red_flagged",
                    })

            self.voting_engine.on("voteProgress", voting_progress_handler)

            # Run voting for this sub-question
            voting_result = await self.voting_engine.vote_until_consensus(
                sub_q["question"], context
            )

            self.voting_engine.remove_listener(
                "voteProgress", voting_progress_handler
            )

            sub_result = SubQuestionResult(
                question=sub_q["question"],
                answer=voting_result["answer"],
                consensus_reached=voting_result["consensus_reached"],
                vote_counts=voting_result["vote_counts"],
                rounds_taken=voting_result["rounds_taken"],
                valid_votes=voting_result["valid_votes"],
                red_flags_encountered=voting_result["red_flags_encountered"],
            )
            sub_results.append(sub_result)

            self.emit("votingComplete", {
                "subQuestionIndex": i,
                "consensusReached": voting_result["consensus_reached"],
                "answer": voting_result["answer"],
            })

        # Step 3: Synthesize the final answer
        self.emit("synthesisStart", {"subAnswers": sub_results})

        synthesis_result = await self.synthesizer.synthesize(
            original_question=question,
            sub_results=sub_results,
            context=context,
        )

        final_answer = synthesis_result["answer"]
        self.emit("synthesisComplete", {"answer": final_answer})

        # Calculate overall statistics
        voting_stats = self._aggregate_voting_stats(sub_results)
        overall_confidence = self._calculate_overall_confidence(sub_results)
        
        execution_time_ms = (time.time() - start_time) * 1000

        # Build final result
        result = MakerResult(
            answer=final_answer,
            confidence=overall_confidence,
            consensus_reached=all(r["consensus_reached"] for r in sub_results),
            is_decomposed=len(decomposition_result["sub_questions"]) > 1,
            sub_questions=sub_results,
            voting_stats=voting_stats,
            execution_time_ms=execution_time_ms,
        )

        self.emit("complete", result)

        return result

    def _aggregate_voting_stats(
        self, sub_results: List[SubQuestionResult]
    ) -> VotingStats:
        """
        Aggregate voting statistics from all sub-questions.
        
        Args:
            sub_results: Results from all sub-questions.
            
        Returns:
            Aggregated VotingStats.
        """
        if not sub_results:
            return VotingStats(
                total_votes=0,
                valid_votes=0,
                total_red_flagged=0,
                winning_vote_count=0,
                margin=0,
                k=self._k,
            )
        
        if len(sub_results) == 1:
            r = sub_results[0]
            counts = list(r["vote_counts"].values())
            sorted_counts = sorted(counts, reverse=True)
            winning = sorted_counts[0] if sorted_counts else 0
            runner_up = sorted_counts[1] if len(sorted_counts) > 1 else 0
            
            return VotingStats(
                total_votes=r["rounds_taken"],
                valid_votes=r["valid_votes"],
                total_red_flagged=r["red_flags_encountered"],
                winning_vote_count=winning,
                margin=winning - runner_up,
                k=self._k,
            )
        
        # Aggregate from multiple sub-questions
        total_votes = sum(r["rounds_taken"] for r in sub_results)
        valid_votes = sum(r["valid_votes"] for r in sub_results)
        total_red_flagged = sum(r["red_flags_encountered"] for r in sub_results)
        
        # For winning_vote_count and margin, take the minimum (weakest link)
        winning_counts = []
        margins = []
        for r in sub_results:
            counts = list(r["vote_counts"].values())
            sorted_counts = sorted(counts, reverse=True)
            winning = sorted_counts[0] if sorted_counts else 0
            runner_up = sorted_counts[1] if len(sorted_counts) > 1 else 0
            winning_counts.append(winning)
            margins.append(winning - runner_up)
        
        return VotingStats(
            total_votes=total_votes,
            valid_votes=valid_votes,
            total_red_flagged=total_red_flagged,
            winning_vote_count=min(winning_counts) if winning_counts else 0,
            margin=min(margins) if margins else 0,
            k=self._k,
        )

    def _calculate_overall_confidence(
        self,
        sub_results: List[SubQuestionResult],
    ) -> Confidence:
        """
        Calculate overall confidence from sub-results.
        
        Per MAKER paper, confidence is determined by:
        - Whether all sub-questions reached consensus
        - The margin of victory in voting
        - The number of red-flagged responses

        Args:
            sub_results: Results from all sub-questions.

        Returns:
            Confidence level (HIGH, MEDIUM, or LOW).
        """
        if not sub_results:
            return Confidence.LOW

        # Check if any sub-result didn't reach consensus
        all_consensus = all(r["consensus_reached"] for r in sub_results)
        
        if not all_consensus:
            return Confidence.LOW
        
        # Calculate red flag rate
        total_votes = sum(r["rounds_taken"] for r in sub_results)
        total_red_flags = sum(r["red_flags_encountered"] for r in sub_results)
        red_flag_rate = total_red_flags / total_votes if total_votes > 0 else 0
        
        # Calculate average votes per question
        avg_votes = total_votes / len(sub_results)
        
        # High confidence: Quick consensus (avg <= 5 votes), low red flag rate (< 10%)
        if avg_votes <= 5 and red_flag_rate < 0.1:
            return Confidence.HIGH
        
        # Low confidence: Many rounds (> 10) or high red flag rate (> 30%)
        if avg_votes > 10 or red_flag_rate > 0.3:
            return Confidence.LOW
        
        return Confidence.MEDIUM

    def get_config(self) -> MakerConfig:
        """Get the current configuration."""
        return dict(self.config)

    def get_provider(self) -> LLMProvider:
        """Get the underlying provider."""
        return self.provider


class MakerSync:
    """
    Synchronous wrapper for Maker.

    Provides a synchronous interface to the async Maker class for use
    in non-async contexts. Runs async operations in a new event loop.

    Attributes:
        maker: The underlying async Maker instance.

    Examples:
        >>> from maker_core import MakerSync
        >>> from maker_core.providers import OpenAIProvider
        >>> 
        >>> provider = OpenAIProvider(api_key="your-key", model="gpt-4")
        >>> maker = MakerSync(provider=provider)
        >>> 
        >>> result = maker.ask("What is the capital of France?")
        >>> print(result["answer"])
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[MakerConfig] = None,
    ) -> None:
        """
        Initialize the synchronous Maker wrapper.

        Args:
            provider: LLM provider instance.
            config: Optional configuration.
        """
        self.maker = Maker(provider=provider, config=config)

    def ask(
        self,
        question: str,
        options: Optional[AskOptions] = None,
    ) -> MakerResult:
        """
        Ask a question synchronously.

        Args:
            question: The question to answer.
            options: Optional configuration overrides.

        Returns:
            MakerResult with answer and metadata.

        Examples:
            >>> result = maker.ask("What is photosynthesis?")
            >>> print(result["answer"])
        """
        return asyncio.run(self.maker.ask(question, options))

    def on(self, event: str, listener: Callable) -> None:
        """
        Register an event listener.

        Args:
            event: Event name.
            listener: Callback function.
        """
        self.maker.on(event, listener)

    def remove_listener(self, event: str, listener: Callable) -> None:
        """
        Remove an event listener.

        Args:
            event: Event name.
            listener: Listener to remove.
        """
        self.maker.remove_listener(event, listener)

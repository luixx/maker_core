"""Main Maker class implementing the full MAKER pipeline with event support."""

import asyncio
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
        self.config = config or MakerConfig(
            voting_threshold=3,
            max_voting_rounds=10,
            max_sub_questions=5,
            enable_red_flag_filter=True,
        )

        # Initialize components
        self.decomposer = Decomposer(
            provider=provider,
            max_sub_questions=self.config["max_sub_questions"],
        )
        self.voting_engine = VotingEngine(
            provider=provider,
            voting_threshold=self.config["voting_threshold"],
            max_rounds=self.config["max_voting_rounds"],
        )
        self.synthesizer = Synthesizer(provider=provider)
        self.red_flag_filter = RedFlagFilter()

        # Enable red flag filter if configured
        if self.config["enable_red_flag_filter"]:
            self.voting_engine.set_red_flag_filter(self.red_flag_filter)

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
            ...     options={"voting_threshold": 5}
            ... )
            >>> print(f"Answer: {result['answer']}")
            >>> print(f"Confidence: {result['confidence']}")
        """
        opts = options or {}
        context = opts.get("context", "")

        # Step 1: Decompose the question
        self.emit("decompositionStart", {"question": question})
        decomposition_result = await self.decomposer.decompose(question, context)

        self.emit(
            "classificationComplete",
            {
                "classification": decomposition_result["classification"],
                "needs_decomposition": decomposition_result["classification"][
                    "needs_decomposition"
                ],
            },
        )

        self.emit(
            "decomposed",
            {
                "sub_questions": decomposition_result["sub_questions"],
                "count": len(decomposition_result["sub_questions"]),
            },
        )

        # Step 2: Answer each sub-question using voting
        sub_results: List[SubQuestionResult] = []
        total_votes = 0
        total_red_flags = 0

        for i, sub_q in enumerate(decomposition_result["sub_questions"]):
            self.emit(
                "votingStart",
                {
                    "question": sub_q["question"],
                    "index": i,
                    "total": len(decomposition_result["sub_questions"]),
                },
            )

            # Set up voting progress callback
            def voting_progress_handler(data: Dict) -> None:
                self.emit("voteProgress", data)

            self.voting_engine.on("voteProgress", voting_progress_handler)

            # Run voting for this sub-question
            voting_result = await self.voting_engine.vote_until_consensus(
                sub_q["question"], context
            )

            self.voting_engine.remove_listener(
                "voteProgress", voting_progress_handler
            )

            self.emit(
                "votingComplete",
                {
                    "question": sub_q["question"],
                    "answer": voting_result["answer"],
                    "rounds": voting_result["rounds_taken"],
                    "vote_counts": voting_result["vote_counts"],
                },
            )

            # Track statistics
            total_votes += voting_result["rounds_taken"]
            total_red_flags += voting_result["red_flags_encountered"]

            # Emit red flag events if any occurred
            if voting_result["red_flags_encountered"] > 0:
                self.emit(
                    "redFlagged",
                    {
                        "question": sub_q["question"],
                        "count": voting_result["red_flags_encountered"],
                    },
                )

            sub_results.append(
                SubQuestionResult(
                    question=sub_q["question"],
                    answer=voting_result["answer"],
                    vote_counts=voting_result["vote_counts"],
                    rounds_taken=voting_result["rounds_taken"],
                )
            )

        # Step 3: Synthesize the final answer
        self.emit("synthesisStart", {"sub_results": sub_results})

        synthesis_result = await self.synthesizer.synthesize(
            original_question=question,
            sub_results=sub_results,
            context=context,
        )

        final_answer = synthesis_result["answer"]

        self.emit("synthesisComplete", {"answer": final_answer})

        # Calculate confidence based on voting performance
        confidence = self._calculate_confidence(
            total_votes=total_votes,
            total_red_flags=total_red_flags,
            num_sub_questions=len(sub_results),
        )

        # Build final result
        result = MakerResult(
            answer=final_answer,
            confidence=confidence,
            sub_results=sub_results,
            total_voting_rounds=total_votes,
            red_flags_encountered=total_red_flags,
            decomposition_used=decomposition_result["classification"][
                "needs_decomposition"
            ],
        )

        self.emit("complete", result)

        return result

    def _calculate_confidence(
        self,
        total_votes: int,
        total_red_flags: int,
        num_sub_questions: int,
    ) -> Confidence:
        """
        Calculate confidence level based on voting performance.

        Args:
            total_votes: Total voting rounds used.
            total_red_flags: Total red flags encountered.
            num_sub_questions: Number of sub-questions answered.

        Returns:
            Confidence level (HIGH, MEDIUM, or LOW).
        """
        if num_sub_questions == 0:
            return Confidence.MEDIUM

        avg_votes_per_question = total_votes / num_sub_questions
        red_flag_rate = (
            total_red_flags / total_votes if total_votes > 0 else 0
        )

        # High confidence: Quick consensus, few red flags
        if avg_votes_per_question <= 4 and red_flag_rate < 0.1:
            return Confidence.HIGH

        # Low confidence: Many rounds or high red flag rate
        if avg_votes_per_question > 7 or red_flag_rate > 0.3:
            return Confidence.LOW

        # Medium confidence: Everything else
        return Confidence.MEDIUM


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

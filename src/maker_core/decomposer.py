"""Question decomposer implementing Maximal Agentic Decomposition (MAD)."""

import json
from typing import List, Optional
from .providers.base import LLMProvider
from .types import (
    Classification,
    SubQuestion,
    DecompositionResult,
    CompletionRequest,
    Message,
)


class Decomposer:
    """
    Implements Maximal Agentic Decomposition (MAD) from the MAKER paper.

    The decomposer breaks complex questions into atomic sub-questions that
    can be answered independently. It first classifies whether decomposition
    is needed, then generates sub-questions if necessary.

    Attributes:
        provider: LLM provider for generating classifications and decompositions.
        max_sub_questions: Maximum number of sub-questions to generate (default: 5).
    """

    def __init__(self, provider: LLMProvider, max_sub_questions: int = 5) -> None:
        """
        Initialize the decomposer.

        Args:
            provider: LLM provider to use for decomposition.
            max_sub_questions: Maximum number of sub-questions (default: 5).
        """
        self.provider = provider
        self.max_sub_questions = max_sub_questions

    async def decompose(
        self, question: str, context: str = ""
    ) -> DecompositionResult:
        """
        Decompose a question into sub-questions using MAD.

        First classifies the question complexity, then generates sub-questions
        if decomposition is needed.

        Args:
            question: The question to decompose.
            context: Optional context information.

        Returns:
            DecompositionResult with classification and sub-questions.

        Examples:
            >>> decomposer = Decomposer(provider)
            >>> result = await decomposer.decompose(
            ...     "What factors led to the fall of the Roman Empire?"
            ... )
            >>> print(len(result["sub_questions"]))  # 3-5 sub-questions
        """
        # First, classify if decomposition is needed
        classification = await self._classify_question(question, context)

        if not classification["needs_decomposition"]:
            # Simple question, no decomposition needed
            return DecompositionResult(
                sub_questions=[
                    SubQuestion(question=question, reasoning="Question is simple enough")
                ],
                classification=classification,
            )

        # Complex question, generate sub-questions
        sub_questions = await self._generate_sub_questions(question, context)

        return DecompositionResult(
            sub_questions=sub_questions, classification=classification
        )

    async def _classify_question(
        self, question: str, context: str
    ) -> Classification:
        """
        Classify whether a question needs decomposition.

        Uses the LLM to determine question complexity on a scale of 1-10.
        Questions with complexity >= 6 are candidates for decomposition.

        Args:
            question: Question to classify.
            context: Context information.

        Returns:
            Classification with needs_decomposition flag, reasoning, and complexity score.
        """
        system_prompt = """You are an expert at analyzing question complexity.
Classify the question's complexity on a scale of 1-10:
- 1-3: Simple factual questions (e.g., "What is the capital of France?")
- 4-5: Moderate questions requiring some reasoning
- 6-7: Complex questions requiring multiple steps
- 8-10: Very complex questions requiring decomposition

Respond in JSON format with:
{
  "needs_decomposition": boolean (true if complexity >= 6),
  "reasoning": "explanation of the classification",
  "complexity_score": number (1-10)
}"""

        user_prompt = f"Question: {question}"
        if context:
            user_prompt += f"\n\nContext: {context}"

        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request = CompletionRequest(
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            response = await self.provider.complete(request)
            result = json.loads(response["content"])

            return Classification(
                needs_decomposition=result.get("needs_decomposition", False),
                reasoning=result.get("reasoning", ""),
                complexity_score=result.get("complexity_score", 1),
            )
        except Exception:
            # If classification fails, assume decomposition is needed (safer)
            return Classification(
                needs_decomposition=True,
                reasoning="Classification failed, defaulting to decomposition",
                complexity_score=6,
            )

    async def _generate_sub_questions(
        self, question: str, context: str
    ) -> List[SubQuestion]:
        """
        Generate atomic sub-questions for a complex question.

        Uses the LLM to break down the question into 2-5 independent
        sub-questions that can be answered separately.

        Args:
            question: Complex question to decompose.
            context: Context information.

        Returns:
            List of SubQuestion dictionaries.
        """
        system_prompt = f"""You are an expert at decomposing complex questions into atomic sub-questions.

Break down the given question into {self.max_sub_questions} or fewer independent sub-questions that:
1. Are atomic and focused on a single aspect
2. Can be answered independently
3. Together provide a complete answer to the original question
4. Are ordered logically (foundational questions first)

Respond in JSON format with:
{{
  "sub_questions": [
    {{
      "question": "sub-question text",
      "reasoning": "why this sub-question is important"
    }},
    ...
  ]
}}"""

        user_prompt = f"Original Question: {question}"
        if context:
            user_prompt += f"\n\nContext: {context}"

        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request = CompletionRequest(
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        try:
            response = await self.provider.complete(request)
            result = json.loads(response["content"])

            sub_questions = result.get("sub_questions", [])

            # Ensure we don't exceed max_sub_questions
            sub_questions = sub_questions[: self.max_sub_questions]

            # Validate and format sub-questions
            formatted_questions: List[SubQuestion] = []
            for sq in sub_questions:
                if isinstance(sq, dict) and "question" in sq:
                    formatted_questions.append(
                        SubQuestion(
                            question=sq["question"],
                            reasoning=sq.get("reasoning", ""),
                        )
                    )

            # If no valid sub-questions, return original question
            if not formatted_questions:
                formatted_questions = [
                    SubQuestion(question=question, reasoning="Decomposition failed")
                ]

            return formatted_questions

        except Exception:
            # If decomposition fails, return original question as single sub-question
            return [SubQuestion(question=question, reasoning="Decomposition failed")]

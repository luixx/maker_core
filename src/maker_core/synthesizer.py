"""Synthesizer for combining sub-answers into coherent final answer."""

import json
from typing import List
from .providers.base import LLMProvider
from .types import SubQuestionResult, CompletionRequest, Message


class Synthesizer:
    """
    Synthesizes sub-answers into a coherent final answer.

    Takes the answers from multiple sub-questions and combines them
    into a well-structured, comprehensive response to the original question.

    Attributes:
        provider: LLM provider for generating the synthesis.
    """

    def __init__(self, provider: LLMProvider) -> None:
        """
        Initialize the synthesizer.

        Args:
            provider: LLM provider to use for synthesis.
        """
        self.provider = provider

    async def synthesize(
        self,
        original_question: str,
        sub_results: List[SubQuestionResult],
        context: str = "",
    ) -> dict:
        """
        Synthesize sub-answers into a final coherent answer.

        Combines multiple sub-question answers into a well-structured
        response that comprehensively addresses the original question.

        Args:
            original_question: The original question asked.
            sub_results: List of SubQuestionResult with questions and answers.
            context: Optional context information.

        Returns:
            Dictionary with "answer" key containing the synthesized answer.

        Examples:
            >>> synthesizer = Synthesizer(provider)
            >>> result = await synthesizer.synthesize(
            ...     "What factors led to the fall of the Roman Empire?",
            ...     sub_results
            ... )
            >>> print(result["answer"])  # Coherent comprehensive answer
        """
        # If only one sub-question, return its answer directly
        if len(sub_results) == 1:
            return {"answer": sub_results[0]["answer"]}

        # Build the synthesis prompt
        system_prompt = """You are an expert at synthesizing information from multiple sources into coherent, comprehensive answers.

Given an original question and answers to several sub-questions, create a well-structured, flowing response that:
1. Directly addresses the original question
2. Integrates all sub-answers naturally
3. Maintains logical flow and coherence
4. Is concise but comprehensive
5. Uses proper transitions between ideas

Your response should read as a unified answer, not a list of separate answers."""

        # Format the sub-results for the prompt
        sub_answers_text = "\n\n".join(
            [
                f"Sub-question {i+1}: {result['question']}\nAnswer: {result['answer']}"
                for i, result in enumerate(sub_results)
            ]
        )

        user_prompt = f"""Original Question: {original_question}

Sub-questions and their answers:
{sub_answers_text}"""

        if context:
            user_prompt += f"\n\nAdditional Context: {context}"

        user_prompt += "\n\nProvide a synthesized answer that directly addresses the original question using the information from all sub-answers."

        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        request = CompletionRequest(
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )

        try:
            response = await self.provider.complete(request)
            answer = response["content"].strip()

            # Ensure the answer is not empty
            if not answer:
                answer = self._fallback_synthesis(sub_results)

            return {"answer": answer}

        except Exception:
            # If synthesis fails, provide a fallback
            return {"answer": self._fallback_synthesis(sub_results)}

    def _fallback_synthesis(self, sub_results: List[SubQuestionResult]) -> str:
        """
        Provide a fallback synthesis if the LLM synthesis fails.

        Simply concatenates the sub-answers with basic formatting.

        Args:
            sub_results: List of sub-question results.

        Returns:
            Basic concatenated answer string.
        """
        if not sub_results:
            return "Unable to generate answer."

        if len(sub_results) == 1:
            return sub_results[0]["answer"]

        # Simple concatenation with numbering
        parts = [
            f"{i+1}. {result['answer']}" for i, result in enumerate(sub_results)
        ]
        return " ".join(parts)

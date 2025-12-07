"""Question decomposer implementing Maximal Agentic Decomposition (MAD)."""

import json
import re
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

    def _try_parse_json(self, content: str) -> Optional[dict]:
        """Try to parse JSON from content, with fallback to extraction."""
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object from content
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON with nested objects (for sub_questions)
        json_match = re.search(r'\{.*"sub_questions"\s*:\s*\[.*\].*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
                
        return None

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
        # Simplified prompt that works better with local LLMs
        prompt = f"""Analyze this question's complexity on a scale of 1-10:
- 1-3: Simple factual questions (e.g., "What is 2+2?", "What is the capital of France?")
- 4-5: Moderate questions requiring some reasoning
- 6-7: Complex questions requiring multiple steps
- 8-10: Very complex questions requiring decomposition into sub-questions

Question: {question}
"""
        if context:
            prompt += f"\nContext provided: {context[:200]}..."

        prompt += """

Respond in JSON format:
{"needs_decomposition": true/false, "reasoning": "brief explanation", "complexity_score": 1-10}"""

        messages: List[Message] = [
            {"role": "user", "content": prompt},
        ]

        request = CompletionRequest(
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )

        try:
            response = await self.provider.complete(request)
            content = response["content"].strip()
            
            result = self._try_parse_json(content)
            if result:
                return Classification(
                    needs_decomposition=result.get("needs_decomposition", False),
                    reasoning=result.get("reasoning", ""),
                    complexity_score=result.get("complexity_score", 1),
                )
            
            # Try to extract from plain text response
            needs_decomposition = self._infer_decomposition_from_text(content, question)
            return Classification(
                needs_decomposition=needs_decomposition,
                reasoning=f"Inferred from response: {content[:100]}",
                complexity_score=6 if needs_decomposition else 3,
            )
            
        except Exception as e:
            # If classification fails, use simple heuristic
            needs_decomposition = self._simple_complexity_heuristic(question)
            return Classification(
                needs_decomposition=needs_decomposition,
                reasoning=f"Classification failed ({e}), using heuristic",
                complexity_score=6 if needs_decomposition else 3,
            )

    def _infer_decomposition_from_text(self, response: str, question: str) -> bool:
        """Infer if decomposition is needed from a non-JSON response."""
        lower = response.lower()
        
        # Look for explicit indicators
        if "needs_decomposition" in lower and "true" in lower:
            return True
        if "needs_decomposition" in lower and "false" in lower:
            return False
        if "complex" in lower or "decompos" in lower or "break down" in lower:
            return True
        if "simple" in lower or "straightforward" in lower or "basic" in lower:
            return False
        
        # Look for complexity score mentions
        score_match = re.search(r'(?:score|complexity)[:\s]*(\d+)', lower)
        if score_match:
            score = int(score_match.group(1))
            return score >= 6
        
        # Fall back to question heuristic
        return self._simple_complexity_heuristic(question)

    def _simple_complexity_heuristic(self, question: str) -> bool:
        """Simple heuristic to determine if a question needs decomposition."""
        # Count complexity indicators
        complexity_indicators = [
            " and ",
            " also ",
            " additionally ",
            " furthermore ",
            " compare ",
            " contrast ",
            " analyze ",
            " explain ",
            " describe ",
            " multiple ",
            " several ",
            " various ",
            "?.*?",  # Multiple question marks
        ]
        
        lower_q = question.lower()
        indicator_count = sum(1 for ind in complexity_indicators if ind in lower_q)
        
        # Also check question length
        word_count = len(question.split())
        
        # Complex if: many indicators OR very long question
        return indicator_count >= 2 or word_count > 25

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
        prompt = f"""Break down this complex question into {self.max_sub_questions} or fewer simple sub-questions.

Each sub-question should:
1. Focus on a single aspect
2. Be answerable independently
3. Together cover the full original question

Original Question: {question}
"""
        if context:
            prompt += f"\nContext: {context[:300]}..."

        prompt += """

Respond in JSON format:
{"sub_questions": [{"question": "sub-question 1", "reasoning": "why important"}, ...]}"""

        messages: List[Message] = [
            {"role": "user", "content": prompt},
        ]

        request = CompletionRequest(
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )

        try:
            response = await self.provider.complete(request)
            content = response["content"].strip()
            
            result = self._try_parse_json(content)
            if result and "sub_questions" in result:
                sub_questions = result.get("sub_questions", [])
                # Ensure we don't exceed max_sub_questions
                sub_questions = sub_questions[: self.max_sub_questions]
                
                return [
                    SubQuestion(
                        question=sq.get("question", ""),
                        reasoning=sq.get("reasoning", ""),
                    )
                    for sq in sub_questions
                    if sq.get("question")
                ]
            
            # Try to extract sub-questions from plain text
            sub_questions = self._extract_sub_questions_from_text(content)
            if sub_questions:
                return sub_questions[:self.max_sub_questions]
            
        except Exception:
            pass

        # Fallback: return original question as the only sub-question
        return [
            SubQuestion(
                question=question,
                reasoning="Could not decompose, using original question",
            )
        ]

    def _extract_sub_questions_from_text(self, content: str) -> List[SubQuestion]:
        """Extract sub-questions from a plain text response."""
        sub_questions = []
        
        # Look for numbered questions (1. Question? 2. Question?)
        numbered = re.findall(r'\d+[.)]\s*([^?\n]+\?)', content)
        for q in numbered:
            sub_questions.append(SubQuestion(
                question=q.strip(),
                reasoning="Extracted from response",
            ))
        
        if sub_questions:
            return sub_questions
        
        # Look for bullet points
        bullets = re.findall(r'[-•*]\s*([^?\n]+\?)', content)
        for q in bullets:
            sub_questions.append(SubQuestion(
                question=q.strip(),
                reasoning="Extracted from response",
            ))
        
        if sub_questions:
            return sub_questions
        
        # Look for any sentences ending in ?
        questions = re.findall(r'([A-Z][^.!?]*\?)', content)
        for q in questions:
            if len(q) > 10:  # Filter out very short matches
                sub_questions.append(SubQuestion(
                    question=q.strip(),
                    reasoning="Extracted from response",
                ))
        
        return sub_questions

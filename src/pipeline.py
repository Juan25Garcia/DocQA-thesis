from __future__ import annotations

from collections.abc import Callable

from src.models import AnswerResult, RetrievedEvidence


class DocumentQAPipeline:
    def __init__(
        self,
        retriever,
        answer_function: Callable[
            [str, list[RetrievedEvidence]],
            str,
        ],
        generator_name: str = "baseline_answer",
    ):
        self.retriever = retriever
        self.answer_function = answer_function
        self.generator_name = generator_name

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> AnswerResult:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        evidence = self.retriever.retrieve(
            normalized_question,
            top_k,
        )

        if not evidence:
            return AnswerResult(
                question=normalized_question,
                answer="No relevant evidence was found.",
                evidence=[],
                generator_name=self.generator_name,
                metadata={
                    "retriever": self.retriever.name,
                    "top_k": top_k,
                    "retrieval_empty": True,
                },
            )

        answer = self.answer_function(
            normalized_question,
            evidence,
        )

        return AnswerResult(
            question=normalized_question,
            answer=answer,
            evidence=evidence,
            generator_name=self.generator_name,
            metadata={
                "retriever": self.retriever.name,
                "top_k": top_k,
            },
        )

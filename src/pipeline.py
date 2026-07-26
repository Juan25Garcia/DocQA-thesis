from __future__ import annotations

from typing import Protocol, Sequence

from src.models import (
    AnswerResult,
    RetrievedEvidence,
)


class RetrieverProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        ...


class GeneratorProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> AnswerResult:
        ...


class DocumentQAPipeline:
    def __init__(
        self,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        normalized_question = self._validate_question(
            question
        )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        return self.retriever.retrieve(
            query=normalized_question,
            top_k=top_k,
        )

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> AnswerResult:
        normalized_question = self._validate_question(
            question
        )

        evidence = self.retrieve(
            question=normalized_question,
            top_k=top_k,
        )

        if not evidence:
            return AnswerResult(
                question=normalized_question,
                answer=(
                    "No relevant evidence was found "
                    "in the indexed documents."
                ),
                evidence=[],
                generator_name=self.generator.name,
                metadata={
                    "retriever": self.retriever.name,
                    "top_k": top_k,
                    "retrieval_empty": True,
                },
            )

        result = self.generator.generate(
            question=normalized_question,
            evidence=evidence,
        )

        result.metadata.update(
            {
                "retriever": self.retriever.name,
                "top_k": top_k,
            }
        )

        return result

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty."
            )

        return normalized_question
from __future__ import annotations

from collections.abc import Sequence

from src.answer import QwenVLAnswerer
from src.models import (
    AnswerResult,
    RetrievedEvidence,
)


class QwenAnswerGenerator:
    """
    Adapter between RetrievedEvidence and QwenVLAnswerer.

    The Qwen model is loaded lazily so that retrieval-only
    executions do not load the vision-language model.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._answerer: QwenVLAnswerer | None = None

    @property
    def name(self) -> str:
        return "qwen2_vl_answer_generator"

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> AnswerResult:
        image_paths = self._get_image_paths(
            evidence
        )

        if not image_paths:
            return AnswerResult(
                question=question,
                answer=(
                    "No document page images were available "
                    "to generate an answer."
                ),
                evidence=list(evidence),
                generator_name=self.name,
                metadata={
                    "number_of_images": 0,
                },
            )

        answerer = self._get_answerer()

        answer = answerer.answer(
            question=question,
            image_paths=image_paths,
        )

        return AnswerResult(
            question=question,
            answer=answer,
            evidence=list(evidence),
            generator_name=self.name,
            metadata={
                "number_of_images": len(image_paths),
                "model_name": answerer.model_name,
            },
        )

    def _get_answerer(self) -> QwenVLAnswerer:
        if self._answerer is None:
            if self.model_name is None:
                self._answerer = QwenVLAnswerer()
            else:
                self._answerer = QwenVLAnswerer(
                    model_name=self.model_name
                )

        return self._answerer

    @staticmethod
    def _get_image_paths(
        evidence: Sequence[RetrievedEvidence],
    ) -> list[str]:
        image_paths: list[str] = []

        for item in evidence:
            image_path = item.node.image_path

            if image_path is None:
                continue

            image_paths.append(str(image_path))

        return image_paths

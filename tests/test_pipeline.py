from pathlib import Path
from typing import Sequence

from src.models import (
    AnswerResult,
    DocumentNode,
    NodeType,
    RetrievedEvidence,
)
from src.pipeline import DocumentQAPipeline


class FakeRetriever:
    @property
    def name(self) -> str:
        return "fake_retriever"

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        node = DocumentNode(
            node_id="invoice:page:2",
            document_id="invoice",
            node_type=NodeType.PAGE,
            page_number=2,
            text="Invoice total: 1,250.00",
            image_path=Path(
                "data/pages/invoice/page_002.png"
            ),
        )

        return [
            RetrievedEvidence(
                node=node,
                score=0.95,
                rank=1,
                retriever_name=self.name,
            )
        ]


class FakeGenerator:
    @property
    def name(self) -> str:
        return "fake_generator"

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer="The invoice total is 1,250.00.",
            evidence=list(evidence),
            generator_name=self.name,
        )


def test_pipeline_returns_answer_and_evidence() -> None:
    pipeline = DocumentQAPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    result = pipeline.ask(
        question="What is the invoice total?",
        top_k=3,
    )

    assert result.answer == "The invoice total is 1,250.00."
    assert len(result.evidence) == 1
    assert result.evidence[0].node.page_number == 2
    assert result.evidence[0].score == 0.95
    assert result.metadata["top_k"] == 3
    assert result.metadata["retriever"] == "fake_retriever"


def test_pipeline_rejects_empty_question() -> None:
    pipeline = DocumentQAPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    try:
        pipeline.ask("   ")
    except ValueError as error:
        assert str(error) == "Question cannot be empty."
    else:
        raise AssertionError(
            "An empty question should raise ValueError."
        )
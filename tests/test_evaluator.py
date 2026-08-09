from __future__ import annotations

import pytest

from src.evaluation import (
    RetrievalEvaluator,
    RetrievalExample,
)
from src.models import (
    DocumentNode,
    NodeType,
    RetrievedEvidence,
)


class FakeRetriever:
    @property
    def name(self) -> str:
        return "fake_retriever"

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        node_ids = [
            "doc:table:wrong",
            "doc:table:correct",
            "doc:paragraph:0",
        ]

        evidence: list[RetrievedEvidence] = []

        for rank, node_id in enumerate(
            node_ids[:top_k],
            start=1,
        ):
            node = DocumentNode(
                node_id=node_id,
                document_id="doc",
                node_type=NodeType.TABLE,
            )

            evidence.append(
                RetrievedEvidence(
                    node=node,
                    score=1.0 / rank,
                    rank=rank,
                    retriever_name=self.name,
                )
            )

        return evidence


def test_evaluator_computes_metrics() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="What is the total amount?",
            relevant_node_ids={
                "doc:table:correct"
            },
        )
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=3,
    )

    assert result.retriever_name == (
        "fake_retriever"
    )
    assert result.example_count == 1
    assert result.top_k == 3

    assert result.hit_rate_at_k == pytest.approx(
        1.0
    )
    assert result.recall_at_k == pytest.approx(
        1.0
    )
    assert result.mrr == pytest.approx(
        0.5
    )

    assert result.average_latency_ms >= 0.0


def test_evaluator_multiple_examples() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="Question one",
            relevant_node_ids={
                "doc:table:correct"
            },
        ),
        RetrievalExample(
            question="Question two",
            relevant_node_ids={
                "missing"
            },
        ),
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=3,
    )

    assert result.example_count == 2

    assert result.hit_rate_at_k == pytest.approx(
        0.5
    )
    assert result.recall_at_k == pytest.approx(
        0.5
    )
    assert result.mrr == pytest.approx(
        0.25
    )


def test_evaluator_stores_per_example_results() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="What is the correct table?",
            relevant_node_ids={
                "doc:table:correct"
            },
            metadata={
                "id": "test_001",
            },
        )
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=3,
    )

    assert len(result.example_results) == 1

    example_result = result.example_results[0]

    assert example_result.is_hit is True
    assert example_result.hit == pytest.approx(
        1.0
    )
    assert example_result.recall == pytest.approx(
        1.0
    )
    assert (
        example_result.first_relevant_rank
        == 2
    )
    assert (
        example_result.reciprocal_rank
        == pytest.approx(0.5)
    )

    assert example_result.retrieved_node_ids == [
        "doc:table:wrong",
        "doc:table:correct",
        "doc:paragraph:0",
    ]

    assert (
        example_result.retrieved_scores[0]
        == pytest.approx(1.0)
    )
    assert (
        example_result.retrieved_scores[1]
        == pytest.approx(0.5)
    )
    assert (
        example_result.retrieved_scores[2]
        == pytest.approx(1.0 / 3.0)
    )

    assert (
        example_result.metadata["id"]
        == "test_001"
    )

    assert example_result.latency_ms >= 0.0


def test_evaluator_records_miss() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="Missing evidence",
            relevant_node_ids={
                "doc:missing"
            },
            metadata={
                "id": "test_002",
            },
        )
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=3,
    )

    assert len(result.example_results) == 1

    example_result = result.example_results[0]

    assert example_result.is_hit is False
    assert example_result.hit == pytest.approx(
        0.0
    )
    assert example_result.recall == pytest.approx(
        0.0
    )
    assert example_result.first_relevant_rank is None
    assert example_result.reciprocal_rank == pytest.approx(
        0.0
    )


def test_evaluator_respects_top_k() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="What is the correct table?",
            relevant_node_ids={
                "doc:table:correct"
            },
        )
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=1,
    )

    example_result = result.example_results[0]

    assert example_result.retrieved_node_ids == [
        "doc:table:wrong"
    ]

    assert example_result.is_hit is False
    assert example_result.first_relevant_rank is None
    assert result.hit_rate_at_k == pytest.approx(
        0.0
    )
    assert result.mrr == pytest.approx(
        0.0
    )


def test_evaluator_computes_partial_recall() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="Find relevant evidence",
            relevant_node_ids={
                "doc:table:correct",
                "doc:paragraph:missing",
            },
        )
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=3,
    )

    example_result = result.example_results[0]

    assert example_result.is_hit is True
    assert example_result.hit == pytest.approx(
        1.0
    )

    assert example_result.recall == pytest.approx(
        0.5
    )

    assert result.recall_at_k == pytest.approx(
        0.5
    )


def test_evaluation_result_to_dict() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="What is the correct table?",
            relevant_node_ids={
                "doc:table:correct"
            },
            metadata={
                "id": "test_003",
                "document_id": "doc",
            },
        )
    ]

    result = evaluator.evaluate(
        examples=examples,
        top_k=3,
    )

    payload = result.to_dict()

    assert payload["retriever_name"] == (
        "fake_retriever"
    )
    assert payload["example_count"] == 1
    assert payload["top_k"] == 3

    example_results = payload["example_results"]

    assert isinstance(example_results, list)
    assert len(example_results) == 1

    example_payload = example_results[0]

    assert (
        example_payload["question"]
        == "What is the correct table?"
    )

    assert example_payload[
        "relevant_node_ids"
    ] == [
        "doc:table:correct"
    ]

    assert example_payload[
        "first_relevant_rank"
    ] == 2

    assert example_payload["metadata"] == {
        "id": "test_003",
        "document_id": "doc",
    }


def test_evaluator_rejects_empty_examples() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    with pytest.raises(
        ValueError,
        match="At least one evaluation example",
    ):
        evaluator.evaluate(
            examples=[],
            top_k=3,
        )


def test_evaluator_rejects_invalid_top_k() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="Question",
            relevant_node_ids={"node"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        evaluator.evaluate(
            examples=examples,
            top_k=0,
        )


def test_evaluator_rejects_empty_question() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="   ",
            relevant_node_ids={
                "doc:table:correct"
            },
        )
    ]

    with pytest.raises(
        ValueError,
        match="questions cannot be empty",
    ):
        evaluator.evaluate(
            examples=examples,
            top_k=3,
        )


def test_evaluator_rejects_empty_relevant_nodes() -> None:
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever()
    )

    examples = [
        RetrievalExample(
            question="Question",
            relevant_node_ids=set(),
        )
    ]

    with pytest.raises(
        ValueError,
        match="at least one relevant node",
    ):
        evaluator.evaluate(
            examples=examples,
            top_k=3,
        )

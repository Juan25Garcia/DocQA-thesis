from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol

from src.models import RetrievedEvidence


EvaluationLevel = Literal["node", "page"]


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


@dataclass(frozen=True)
class RetrievalExample:
    question: str

    relevant_node_ids: set[str] = field(
        default_factory=set
    )

    relevant_pages: set[tuple[str, int]] = field(
        default_factory=set
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RetrievalExampleResult:
    question: str

    relevant_node_ids: set[str]
    relevant_pages: set[tuple[str, int]]

    retrieved_node_ids: list[str]
    retrieved_pages: list[tuple[str, int | None]]
    retrieved_scores: list[float]

    hit: float
    recall: float
    reciprocal_rank: float

    first_relevant_rank: int | None

    latency_ms: float

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def is_hit(self) -> bool:
        return self.hit > 0.0


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    retriever_name: str
    evaluation_level: EvaluationLevel
    example_count: int
    top_k: int

    hit_rate_at_k: float
    recall_at_k: float
    mrr: float
    average_latency_ms: float

    example_results: list[
        RetrievalExampleResult
    ] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "retriever_name": self.retriever_name,
            "evaluation_level": self.evaluation_level,
            "example_count": self.example_count,
            "top_k": self.top_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "average_latency_ms": (
                self.average_latency_ms
            ),
            "example_results": [
                {
                    "question": item.question,
                    "relevant_node_ids": sorted(
                        item.relevant_node_ids
                    ),
                    "relevant_pages": [
                        {
                            "document_id": document_id,
                            "page_number": page_number,
                        }
                        for (
                            document_id,
                            page_number,
                        ) in sorted(
                            item.relevant_pages
                        )
                    ],
                    "retrieved_node_ids": (
                        item.retrieved_node_ids
                    ),
                    "retrieved_pages": [
                        {
                            "document_id": document_id,
                            "page_number": page_number,
                        }
                        for (
                            document_id,
                            page_number,
                        ) in item.retrieved_pages
                    ],
                    "retrieved_scores": (
                        item.retrieved_scores
                    ),
                    "hit": item.hit,
                    "recall": item.recall,
                    "reciprocal_rank": (
                        item.reciprocal_rank
                    ),
                    "first_relevant_rank": (
                        item.first_relevant_rank
                    ),
                    "latency_ms": item.latency_ms,
                    "metadata": item.metadata,
                }
                for item in self.example_results
            ],
        }


class RetrievalEvaluator:
    """Evaluate node-level or page-level retrieval."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        evaluation_level: EvaluationLevel = "node",
    ) -> None:
        self.retriever = retriever
        self.evaluation_level = evaluation_level

    def evaluate(
        self,
        examples: list[RetrievalExample],
        top_k: int = 3,
    ) -> RetrievalEvaluationResult:
        if not examples:
            raise ValueError(
                "At least one evaluation example is required."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        example_results: list[
            RetrievalExampleResult
        ] = []

        for example in examples:
            self._validate_example(example)

            start = perf_counter()

            evidence = self.retriever.retrieve(
                query=example.question.strip(),
                top_k=top_k,
            )

            latency_ms = (
                perf_counter() - start
            ) * 1000.0

            relevance_flags = [
                self._is_relevant(
                    item=item,
                    example=example,
                )
                for item in evidence
            ]

            first_relevant_rank = (
                self._first_relevant_rank(
                    relevance_flags
                )
            )

            hit = float(
                first_relevant_rank is not None
            )

            recall = self._compute_recall(
                evidence=evidence,
                example=example,
            )

            reciprocal_rank = (
                0.0
                if first_relevant_rank is None
                else 1.0 / first_relevant_rank
            )

            example_results.append(
                RetrievalExampleResult(
                    question=example.question.strip(),
                    relevant_node_ids=set(
                        example.relevant_node_ids
                    ),
                    relevant_pages=set(
                        example.relevant_pages
                    ),
                    retrieved_node_ids=[
                        item.node.node_id
                        for item in evidence
                    ],
                    retrieved_pages=[
                        (
                            item.node.document_id,
                            item.node.page_number,
                        )
                        for item in evidence
                    ],
                    retrieved_scores=[
                        item.score
                        for item in evidence
                    ],
                    hit=hit,
                    recall=recall,
                    reciprocal_rank=reciprocal_rank,
                    first_relevant_rank=(
                        first_relevant_rank
                    ),
                    latency_ms=latency_ms,
                    metadata=dict(
                        example.metadata
                    ),
                )
            )

        example_count = len(example_results)

        return RetrievalEvaluationResult(
            retriever_name=self.retriever.name,
            evaluation_level=self.evaluation_level,
            example_count=example_count,
            top_k=top_k,
            hit_rate_at_k=(
                sum(
                    item.hit
                    for item in example_results
                )
                / example_count
            ),
            recall_at_k=(
                sum(
                    item.recall
                    for item in example_results
                )
                / example_count
            ),
            mrr=(
                sum(
                    item.reciprocal_rank
                    for item in example_results
                )
                / example_count
            ),
            average_latency_ms=(
                sum(
                    item.latency_ms
                    for item in example_results
                )
                / example_count
            ),
            example_results=example_results,
        )

    def _validate_example(
        self,
        example: RetrievalExample,
    ) -> None:
        if not example.question.strip():
            raise ValueError(
                "Evaluation questions cannot be empty."
            )

        if (
            self.evaluation_level == "node"
            and not example.relevant_node_ids
        ):
            raise ValueError(
                "Node-level evaluation requires at least "
                "one relevant node."
            )

        if (
            self.evaluation_level == "page"
            and not example.relevant_pages
        ):
            raise ValueError(
                "Page-level evaluation requires at least "
                "one relevant page."
            )

    def _is_relevant(
        self,
        item: RetrievedEvidence,
        example: RetrievalExample,
    ) -> bool:
        if self.evaluation_level == "node":
            return (
                item.node.node_id
                in example.relevant_node_ids
            )

        if item.node.page_number is None:
            return False

        return (
            item.node.document_id,
            item.node.page_number,
        ) in example.relevant_pages

    def _compute_recall(
        self,
        evidence: list[RetrievedEvidence],
        example: RetrievalExample,
    ) -> float:
        if self.evaluation_level == "node":
            retrieved = {
                item.node.node_id
                for item in evidence
            }

            relevant = example.relevant_node_ids

        else:
            retrieved = {
                (
                    item.node.document_id,
                    item.node.page_number,
                )
                for item in evidence
                if item.node.page_number is not None
            }

            relevant = example.relevant_pages

        return len(
            retrieved & relevant
        ) / len(relevant)

    @staticmethod
    def _first_relevant_rank(
        relevance_flags: list[bool],
    ) -> int | None:
        for rank, relevant in enumerate(
            relevance_flags,
            start=1,
        ):
            if relevant:
                return rank

        return None

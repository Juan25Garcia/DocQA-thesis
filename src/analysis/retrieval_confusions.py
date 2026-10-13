
"""Analyze retrieval errors against known relevant document blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.analysis import StructuralRelationExtractor
from src.evaluation import RetrievalExample
from src.models import RetrievedEvidence


@dataclass(frozen=True)
class ConfusionRecord:
    question: str
    positive_node_id: str
    candidate_node_id: str
    positive_score: float
    candidate_score: float
    candidate_rank: int
    margin: float
    confusion: bool
    tied: bool
    relations: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        relations = result.pop("relations")
        result.update(relations)
        return result


class RetrievalConfusionAnalyzer:
    """Analyze incorrect blocks relative to the best relevant block."""

    def __init__(
        self,
        retriever: Any,
        relation_extractor: StructuralRelationExtractor,
    ) -> None:
        self.retriever = retriever
        self.relation_extractor = relation_extractor

    def analyze(
        self,
        example: RetrievalExample,
    ) -> list[ConfusionRecord]:
        if not example.question.strip():
            raise ValueError("Question cannot be empty.")

        if not example.relevant_node_ids:
            raise ValueError(
                "Node-level ground truth is required."
            )

        # Retrieve all indexed blocks to ensure that relevant
        # blocks outside the usual top_k are not omitted.
        indexed_nodes = self.retriever.indexed_nodes
        indexed_ids = {node.node_id for node in indexed_nodes}

        missing = example.relevant_node_ids - indexed_ids
        if missing:
            raise ValueError(
                "Relevant nodes are not indexed: "
                + ", ".join(sorted(missing))
            )

        if not indexed_nodes:
            raise ValueError("The retriever index is empty.")

        # A minimum-score filter would hide candidates and
        # invalidate the full-ranking comparison.
        if self.retriever.minimum_score is not None:
            raise ValueError(
                "Full-ranking analysis requires minimum_score=None."
            )

        evidence = self.retriever.retrieve(
            query=example.question,
            top_k=len(indexed_nodes),
        )

        self._validate_ranking(evidence, indexed_ids)

        positives = [
            item for item in evidence
            if item.node.node_id in example.relevant_node_ids
        ]

        if not positives:
            raise RuntimeError(
                "No relevant block was returned by the retriever."
            )

        # Best-scoring relevant block is the reference.
        positive = max(
            positives,
            key=lambda item: item.score,
        )

        records: list[ConfusionRecord] = []

        for candidate in evidence:
            if candidate.node.node_id in example.relevant_node_ids:
                continue

            margin = positive.score - candidate.score

            relations = self.relation_extractor.compare(
                positive_node_id=positive.node.node_id,
                candidate_node_id=candidate.node.node_id,
            )

            records.append(
                ConfusionRecord(
                    question=example.question,
                    positive_node_id=positive.node.node_id,
                    candidate_node_id=candidate.node.node_id,
                    positive_score=positive.score,
                    candidate_score=candidate.score,
                    candidate_rank=candidate.rank,
                    margin=margin,
                    confusion=margin < 0,
                    tied=margin == 0,
                    relations=relations,
                )
            )

        return records

    @staticmethod
    def _validate_ranking(
        evidence: list[RetrievedEvidence],
        indexed_ids: set[str],
    ) -> None:
        returned_ids = [
            item.node.node_id for item in evidence
        ]

        if (
            len(returned_ids) != len(indexed_ids)
            or set(returned_ids) != indexed_ids
        ):
            raise RuntimeError(
                "Retriever did not return a complete, unique ranking."
            )

        scores = [item.score for item in evidence]
        if any(
            scores[i] < scores[i + 1]
            for i in range(len(scores) - 1)
        ):
            raise RuntimeError(
                "Retriever results are not sorted by descending score."
            )

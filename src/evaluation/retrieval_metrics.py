from __future__ import annotations

from statistics import mean

from src.models import RetrievedEvidence


def hit_at_k(
    evidence: list[RetrievedEvidence],
    relevant_node_ids: set[str],
    k: int,
) -> float:
    retrieved_ids = {
        item.node.node_id
        for item in evidence[:k]
    }

    return float(
        bool(retrieved_ids.intersection(relevant_node_ids))
    )


def reciprocal_rank(
    evidence: list[RetrievedEvidence],
    relevant_node_ids: set[str],
) -> float:
    for rank, item in enumerate(evidence, start=1):
        if item.node.node_id in relevant_node_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    predictions: list[
        tuple[list[RetrievedEvidence], set[str]]
    ],
) -> float:
    values = [
        reciprocal_rank(evidence, relevant_ids)
        for evidence, relevant_ids in predictions
    ]

    return mean(values) if values else 0.0

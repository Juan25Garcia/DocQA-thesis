from __future__ import annotations

from collections.abc import Sequence

from src.models import RetrievedEvidence


def hit_rate_at_k(
    evidence: Sequence[RetrievedEvidence],
    relevant_node_ids: set[str],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant node appears in the top-k results.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_node_ids:
        raise ValueError(
            "relevant_node_ids cannot be empty."
        )

    top_results = evidence[:k]

    return float(
        any(
            item.node.node_id in relevant_node_ids
            for item in top_results
        )
    )


def reciprocal_rank(
    evidence: Sequence[RetrievedEvidence],
    relevant_node_ids: set[str],
) -> float:
    """
    Return the reciprocal rank of the first relevant result.

    Example:
        Relevant result at rank 1 -> 1.0
        Relevant result at rank 2 -> 0.5
        No relevant result -> 0.0
    """
    if not relevant_node_ids:
        raise ValueError(
            "relevant_node_ids cannot be empty."
        )

    for position, item in enumerate(
        evidence,
        start=1,
    ):
        if item.node.node_id in relevant_node_ids:
            return 1.0 / position

    return 0.0


def recall_at_k(
    evidence: Sequence[RetrievedEvidence],
    relevant_node_ids: set[str],
    k: int,
) -> float:
    """
    Return the fraction of relevant nodes retrieved in the top-k.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_node_ids:
        raise ValueError(
            "relevant_node_ids cannot be empty."
        )

    retrieved_ids = {
        item.node.node_id
        for item in evidence[:k]
    }

    relevant_retrieved = (
        retrieved_ids & relevant_node_ids
    )

    return (
        len(relevant_retrieved)
        / len(relevant_node_ids)
    )

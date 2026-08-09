import pytest

from src.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.models import (
    DocumentNode,
    NodeType,
    RetrievedEvidence,
)


def make_evidence(
    node_id: str,
    rank: int,
) -> RetrievedEvidence:
    node = DocumentNode(
        node_id=node_id,
        document_id="doc",
        node_type=NodeType.TABLE,
    )

    return RetrievedEvidence(
        node=node,
        score=1.0 / rank,
        rank=rank,
        retriever_name="test",
    )


def test_hit_rate_at_k_hit() -> None:
    evidence = [
        make_evidence("a", 1),
        make_evidence("b", 2),
    ]

    assert hit_rate_at_k(
        evidence,
        {"b"},
        2,
    ) == 1.0


def test_hit_rate_at_k_miss() -> None:
    evidence = [
        make_evidence("a", 1),
        make_evidence("b", 2),
    ]

    assert hit_rate_at_k(
        evidence,
        {"c"},
        2,
    ) == 0.0


def test_reciprocal_rank() -> None:
    evidence = [
        make_evidence("a", 1),
        make_evidence("b", 2),
        make_evidence("c", 3),
    ]

    assert reciprocal_rank(
        evidence,
        {"b"},
    ) == pytest.approx(0.5)


def test_recall_at_k() -> None:
    evidence = [
        make_evidence("a", 1),
        make_evidence("b", 2),
        make_evidence("c", 3),
    ]

    assert recall_at_k(
        evidence,
        {"a", "c"},
        2,
    ) == pytest.approx(0.5)


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        hit_rate_at_k(
            [],
            {"a"},
            0,
        )

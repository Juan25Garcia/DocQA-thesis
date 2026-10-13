
"""Tests for the Retrieval Confusion Analyzer."""

import pytest

from src.analysis import (
    RetrievalConfusionAnalyzer,
    StructuralRelationExtractor,
)
from src.evaluation import RetrievalExample
from src.indexing import HierarchicalDocumentIndex
from src.models import DocumentNode, NodeType, RetrievedEvidence


class FakeRetriever:
    name = "fake_block_retriever"
    minimum_score = None

    def __init__(self, nodes, scores):
        self.indexed_nodes = nodes
        self.scores = scores

    def retrieve(self, query, top_k=3):
        ranked = sorted(
            self.indexed_nodes,
            key=lambda node: self.scores[node.node_id],
            reverse=True,
        )

        return [
            RetrievedEvidence(
                node=node,
                score=self.scores[node.node_id],
                rank=rank,
                retriever_name=self.name,
            )
            for rank, node in enumerate(
                ranked[:top_k], start=1
            )
        ]


@pytest.fixture
def setup_analyzer():
    nodes = [
        DocumentNode(
            node_id="doc:root",
            document_id="doc",
            node_type=NodeType.DOCUMENT,
            children_ids=["doc:page"],
        ),
        DocumentNode(
            node_id="doc:page",
            document_id="doc",
            node_type=NodeType.PAGE,
            page_number=1,
            parent_id="doc:root",
            children_ids=["doc:total", "doc:subtotal", "doc:note"],
        ),
        DocumentNode(
            node_id="doc:total",
            document_id="doc",
            node_type=NodeType.TABLE,
            page_number=1,
            parent_id="doc:page",
        ),
        DocumentNode(
            node_id="doc:subtotal",
            document_id="doc",
            node_type=NodeType.TABLE,
            page_number=1,
            parent_id="doc:page",
        ),
        DocumentNode(
            node_id="doc:note",
            document_id="doc",
            node_type=NodeType.PARAGRAPH,
            page_number=1,
            parent_id="doc:page",
        ),
    ]

    index = HierarchicalDocumentIndex(nodes)

    retriever = FakeRetriever(
        nodes=nodes[2:],
        scores={
            "doc:total": 0.81,
            "doc:subtotal": 0.84,
            "doc:note": 0.40,
        },
    )

    analyzer = RetrievalConfusionAnalyzer(
        retriever=retriever,
        relation_extractor=StructuralRelationExtractor(index),
    )

    return analyzer, retriever


def test_detects_confusion(setup_analyzer):
    analyzer, _ = setup_analyzer

    records = analyzer.analyze(
        RetrievalExample(
            question="What is the total amount?",
            relevant_node_ids={"doc:total"},
        )
    )

    assert len(records) == 2

    subtotal = next(
        record for record in records
        if record.candidate_node_id == "doc:subtotal"
    )

    assert subtotal.confusion is True
    assert subtotal.margin == pytest.approx(-0.03)
    assert subtotal.candidate_rank == 1
    assert subtotal.relations["same_page"] is True
    assert subtotal.relations["same_type"] is True
    assert subtotal.relations["hierarchical_distance"] == 2


def test_easy_negative_is_not_confusion(setup_analyzer):
    analyzer, _ = setup_analyzer

    records = analyzer.analyze(
        RetrievalExample(
            question="What is the total amount?",
            relevant_node_ids={"doc:total"},
        )
    )

    note = next(
        record for record in records
        if record.candidate_node_id == "doc:note"
    )

    assert note.confusion is False
    assert note.margin == pytest.approx(0.41)


def test_positive_outside_top_one_is_found(setup_analyzer):
    analyzer, _ = setup_analyzer

    records = analyzer.analyze(
        RetrievalExample(
            question="What is the total amount?",
            relevant_node_ids={"doc:note"},
        )
    )

    assert len(records) == 2
    assert all(record.confusion for record in records)


def test_multiple_positives_are_excluded(setup_analyzer):
    analyzer, _ = setup_analyzer

    records = analyzer.analyze(
        RetrievalExample(
            question="What is the total amount?",
            relevant_node_ids={"doc:total", "doc:subtotal"},
        )
    )

    assert len(records) == 1
    assert records[0].candidate_node_id == "doc:note"
    assert records[0].positive_node_id == "doc:subtotal"


def test_missing_positive_raises(setup_analyzer):
    analyzer, _ = setup_analyzer

    with pytest.raises(ValueError, match="not indexed"):
        analyzer.analyze(
            RetrievalExample(
                question="Question",
                relevant_node_ids={"missing"},
            )
        )


def test_empty_ground_truth_raises(setup_analyzer):
    analyzer, _ = setup_analyzer

    with pytest.raises(ValueError, match="ground truth"):
        analyzer.analyze(
            RetrievalExample(question="Question")
        )


def test_tied_scores(setup_analyzer):
    analyzer, retriever = setup_analyzer
    retriever.scores["doc:subtotal"] = 0.81

    records = analyzer.analyze(
        RetrievalExample(
            question="Question",
            relevant_node_ids={"doc:total"},
        )
    )

    subtotal = next(
        record for record in records
        if record.candidate_node_id == "doc:subtotal"
    )

    assert subtotal.tied is True
    assert subtotal.confusion is False

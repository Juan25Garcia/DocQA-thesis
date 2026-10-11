
"""Tests for the structural relation extractor (Issue #11)."""

import pytest

from src.analysis import StructuralRelationExtractor
from src.indexing import HierarchicalDocumentIndex
from src.models import DocumentNode, NodeType


@pytest.fixture
def extractor() -> StructuralRelationExtractor:
    nodes = [
        DocumentNode(
            node_id="a:doc",
            document_id="a",
            node_type=NodeType.DOCUMENT,
            children_ids=["a:p1", "a:p2"],
        ),
        DocumentNode(
            node_id="a:p1",
            document_id="a",
            node_type=NodeType.PAGE,
            page_number=1,
            parent_id="a:doc",
            children_ids=["a:t", "a:x"],
        ),
        DocumentNode(
            node_id="a:p2",
            document_id="a",
            node_type=NodeType.PAGE,
            page_number=2,
            parent_id="a:doc",
            children_ids=["a:y"],
        ),
        DocumentNode(
            node_id="a:t",
            document_id="a",
            node_type=NodeType.TABLE,
            page_number=1,
            parent_id="a:p1",
        ),
        DocumentNode(
            node_id="a:x",
            document_id="a",
            node_type=NodeType.PARAGRAPH,
            page_number=1,
            parent_id="a:p1",
        ),
        DocumentNode(
            node_id="a:y",
            document_id="a",
            node_type=NodeType.TABLE,
            page_number=2,
            parent_id="a:p2",
        ),
        DocumentNode(
            node_id="b:doc",
            document_id="b",
            node_type=NodeType.DOCUMENT,
            children_ids=["b:p1"],
        ),
        DocumentNode(
            node_id="b:p1",
            document_id="b",
            node_type=NodeType.PAGE,
            page_number=1,
            parent_id="b:doc",
            children_ids=["b:t"],
        ),
        DocumentNode(
            node_id="b:t",
            document_id="b",
            node_type=NodeType.TABLE,
            page_number=1,
            parent_id="b:p1",
        ),
    ]

    return StructuralRelationExtractor(
        HierarchicalDocumentIndex(nodes)
    )


def test_siblings(extractor):
    relations = extractor.compare("a:t", "a:x")

    assert relations == {
        "same_document": True,
        "same_page": True,
        "same_type": False,
        "same_parent": True,
        "sibling": True,
        "hierarchical_distance": 2,
    }


def test_different_pages_same_document(extractor):
    relations = extractor.compare("a:t", "a:y")

    assert relations["same_document"] is True
    assert relations["same_page"] is False
    assert relations["same_type"] is True
    assert relations["same_parent"] is False
    assert relations["hierarchical_distance"] == 4


def test_different_documents(extractor):
    relations = extractor.compare("a:t", "b:t")

    assert relations["same_document"] is False
    assert relations["same_page"] is False
    assert relations["hierarchical_distance"] is None


def test_same_node_and_ancestor(extractor):
    same = extractor.compare("a:t", "a:t")

    assert same["hierarchical_distance"] == 0
    assert same["sibling"] is False

    ancestor = extractor.compare("a:t", "a:p1")

    assert ancestor["hierarchical_distance"] == 1


def test_missing_page_number(extractor):
    relations = extractor.compare("a:doc", "a:t")

    assert relations["same_page"] is None


def test_unknown_node(extractor):
    with pytest.raises(KeyError):
        extractor.compare("missing", "a:t")

import pytest

from src.indexing import (
    HierarchicalDocumentIndex,
    HierarchicalIndexError,
)
from src.models import DocumentNode, NodeType


def build_sample_nodes() -> list[DocumentNode]:
    document = DocumentNode(
        node_id="invoice:document",
        document_id="invoice",
        node_type=NodeType.DOCUMENT,
        children_ids=["invoice:page:1"],
    )

    page = DocumentNode(
        node_id="invoice:page:1",
        document_id="invoice",
        node_type=NodeType.PAGE,
        page_number=1,
        parent_id="invoice:document",
        children_ids=[
            "invoice:section:0",
            "invoice:table:0",
            "invoice:paragraph:0",
        ],
    )

    section = DocumentNode(
        node_id="invoice:section:0",
        document_id="invoice",
        node_type=NodeType.SECTION,
        page_number=1,
        text="INVOICE",
        parent_id="invoice:page:1",
    )

    table = DocumentNode(
        node_id="invoice:table:0",
        document_id="invoice",
        node_type=NodeType.TABLE,
        page_number=1,
        text="Total Amount | $1,245.50",
        parent_id="invoice:page:1",
    )

    paragraph = DocumentNode(
        node_id="invoice:paragraph:0",
        document_id="invoice",
        node_type=NodeType.PARAGRAPH,
        page_number=1,
        text="Payment notes",
        parent_id="invoice:page:1",
    )

    return [
        document,
        page,
        section,
        table,
        paragraph,
    ]


def test_build_indexes_nodes() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    assert len(index) == 5
    assert index.node_count == 5
    assert index.document_ids == ["invoice"]
    assert "invoice:table:0" in index


def test_get_node() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    node = index.get_node("invoice:table:0")

    assert node.node_type == NodeType.TABLE
    assert node.text == "Total Amount | $1,245.50"


def test_get_root() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    root = index.get_root("invoice")

    assert root.node_type == NodeType.DOCUMENT
    assert root.parent_id is None


def test_get_children_preserves_reading_order() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    children = index.get_children("invoice:page:1")

    assert [
        node.node_id
        for node in children
    ] == [
        "invoice:section:0",
        "invoice:table:0",
        "invoice:paragraph:0",
    ]


def test_get_parent() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    parent = index.get_parent(
        "invoice:table:0"
    )

    assert parent is not None
    assert parent.node_id == "invoice:page:1"


def test_get_ancestors() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    ancestors = index.get_ancestors(
        "invoice:table:0"
    )

    assert [
        node.node_id
        for node in ancestors
    ] == [
        "invoice:page:1",
        "invoice:document",
    ]


def test_get_descendants() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    descendants = index.get_descendants(
        "invoice:document"
    )

    assert [
        node.node_id
        for node in descendants
    ] == [
        "invoice:page:1",
        "invoice:section:0",
        "invoice:table:0",
        "invoice:paragraph:0",
    ]


def test_get_descendants_with_max_depth() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    descendants = index.get_descendants(
        "invoice:document",
        max_depth=1,
    )

    assert [
        node.node_id
        for node in descendants
    ] == [
        "invoice:page:1",
    ]


def test_get_nodes_by_type() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    tables = index.get_nodes_by_type(
        NodeType.TABLE
    )

    assert len(tables) == 1
    assert tables[0].node_id == "invoice:table:0"


def test_get_page_nodes() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    nodes = index.get_page_nodes(
        document_id="invoice",
        page_number=1,
    )

    assert {
        node.node_id
        for node in nodes
    } == {
        "invoice:page:1",
        "invoice:section:0",
        "invoice:table:0",
        "invoice:paragraph:0",
    }


def test_get_content_nodes_excludes_containers() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    content_nodes = index.get_content_nodes()

    assert {
        node.node_type
        for node in content_nodes
    } == {
        NodeType.SECTION,
        NodeType.TABLE,
        NodeType.PARAGRAPH,
    }


def test_duplicate_node_id_raises() -> None:
    nodes = build_sample_nodes()
    nodes.append(nodes[-1])

    with pytest.raises(
        HierarchicalIndexError,
        match="Duplicate node ID",
    ):
        HierarchicalDocumentIndex(nodes)


def test_missing_parent_raises() -> None:
    nodes = build_sample_nodes()

    nodes[-1].parent_id = "missing-parent"

    with pytest.raises(
        HierarchicalIndexError,
        match="missing parent",
    ):
        HierarchicalDocumentIndex(nodes)


def test_inconsistent_parent_child_relationship_raises() -> None:
    nodes = build_sample_nodes()

    page = nodes[1]
    page.children_ids.remove(
        "invoice:paragraph:0"
    )

    with pytest.raises(
        HierarchicalIndexError,
        match="Parent-child relationship mismatch",
    ):
        HierarchicalDocumentIndex(nodes)


def test_missing_node_raises_key_error() -> None:
    index = HierarchicalDocumentIndex(
        build_sample_nodes()
    )

    with pytest.raises(
        KeyError,
        match="Node not found",
    ):
        index.get_node("missing")
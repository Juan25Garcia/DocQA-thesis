from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models import DocumentNode, NodeType
from src.pipeline import DocumentQAPipeline


def build_hierarchical_nodes() -> list[DocumentNode]:
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

    return [
        document,
        page,
        section,
        table,
    ]


def test_ingest_builds_hierarchical_index(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "invoice.pdf"
    document_path.write_bytes(b"%PDF-1.4 mock")

    parser = MagicMock()
    parser.parse.return_value = (
        build_hierarchical_nodes()
    )

    retriever = MagicMock()
    retriever.name = "mock-retriever"

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        document_parser=parser,
    )

    nodes = pipeline.ingest(document_path)

    assert len(nodes) == 4
    assert pipeline.document_index is not None
    assert pipeline.document_index.node_count == 4
    assert pipeline.document_index.document_ids == [
        "invoice"
    ]


def test_pipeline_can_query_structured_index(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "invoice.pdf"
    document_path.write_bytes(b"%PDF-1.4 mock")

    parser = MagicMock()
    parser.parse.return_value = (
        build_hierarchical_nodes()
    )

    retriever = MagicMock()
    retriever.name = "mock-retriever"

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        document_parser=parser,
    )

    pipeline.ingest(document_path)

    tables = pipeline.get_structured_nodes_by_type(
        NodeType.TABLE
    )

    assert len(tables) == 1
    assert tables[0].text == (
        "Total Amount | $1,245.50"
    )

    children = pipeline.get_structured_children(
        "invoice:page:1"
    )

    assert [
        child.node_id
        for child in children
    ] == [
        "invoice:section:0",
        "invoice:table:0",
    ]

    ancestors = pipeline.get_structured_ancestors(
        "invoice:table:0"
    )

    assert [
        ancestor.node_id
        for ancestor in ancestors
    ] == [
        "invoice:page:1",
        "invoice:document",
    ]


def test_structured_index_requires_ingestion() -> None:
    retriever = MagicMock()
    retriever.name = "mock-retriever"

    pipeline = DocumentQAPipeline(
        retriever=retriever,
    )

    with pytest.raises(
        RuntimeError,
        match="Call ingest",
    ):
        pipeline.get_structured_content_nodes()


def test_clear_document_nodes_clears_index(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "invoice.pdf"
    document_path.write_bytes(b"%PDF-1.4 mock")

    parser = MagicMock()
    parser.parse.return_value = (
        build_hierarchical_nodes()
    )

    retriever = MagicMock()
    retriever.name = "mock-retriever"

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        document_parser=parser,
    )

    pipeline.ingest(document_path)
    pipeline.clear_document_nodes()

    assert pipeline.document_nodes == []
    assert pipeline.document_index is None
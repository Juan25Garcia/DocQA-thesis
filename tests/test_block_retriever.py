from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.indexing import HierarchicalDocumentIndex
from src.models import (
    DocumentNode,
    NodeType,
)
from src.retrievers.block_retriever import (
    SemanticBlockRetriever,
)


class FakeEncoder:
    """
    Deterministic encoder for unit tests.

    Text related to totals maps to [1, 0].
    Text related to payment notes maps to [0, 1].
    """

    def encode(
        self,
        sentences: str | list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            return self._encode_text(sentences)

        return np.asarray(
            [
                self._encode_text(text)
                for text in sentences
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _encode_text(
        text: str,
    ) -> np.ndarray:
        normalized = text.lower()

        if (
            "total" in normalized
            or "$1,245.50" in normalized
        ):
            return np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            )

        if (
            "payment" in normalized
            or "notes" in normalized
        ):
            return np.asarray(
                [0.0, 1.0],
                dtype=np.float32,
            )

        return np.asarray(
            [0.5, 0.5],
            dtype=np.float32,
        )


def build_nodes() -> list[DocumentNode]:
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
        text="Invoice line items",
        parent_id="invoice:page:1",
    )

    table = DocumentNode(
        node_id="invoice:table:0",
        document_id="invoice",
        node_type=NodeType.TABLE,
        page_number=1,
        text=(
            "Description | Quantity | Amount\n"
            "Calibration service | 1 | $265.50\n"
            "Total Amount | $1,245.50"
        ),
        parent_id="invoice:page:1",
    )

    paragraph = DocumentNode(
        node_id="invoice:paragraph:0",
        document_id="invoice",
        node_type=NodeType.PARAGRAPH,
        page_number=1,
        text="Please reference the invoice when submitting payment.",
        parent_id="invoice:page:1",
    )

    return [
        document,
        page,
        section,
        table,
        paragraph,
    ]


def build_retriever() -> SemanticBlockRetriever:
    index = HierarchicalDocumentIndex(
        build_nodes()
    )

    return SemanticBlockRetriever(
        document_index=index,
        model=FakeEncoder(),
        model_name="fake-model",
    )


def test_build_indexes_content_blocks() -> None:
    retriever = build_retriever()

    assert retriever.indexed_node_count == 3

    assert {
        node.node_type
        for node in retriever.indexed_nodes
    } == {
        NodeType.SECTION,
        NodeType.TABLE,
        NodeType.PARAGRAPH,
    }


def test_retrieve_returns_relevant_table_first() -> None:
    retriever = build_retriever()

    results = retriever.retrieve(
        query="What is the total amount?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].node.node_type == NodeType.TABLE
    assert "$1,245.50" in results[0].node.text
    assert results[0].rank == 1
    assert results[0].retriever_name == (
        "semantic_block_retriever"
    )


def test_retrieve_returns_payment_paragraph() -> None:
    retriever = build_retriever()

    results = retriever.retrieve(
        query="How should payment be submitted?",
        top_k=1,
    )

    assert len(results) == 1
    assert (
        results[0].node.node_type
        == NodeType.PARAGRAPH
    )


def test_retrieve_rejects_empty_query() -> None:
    retriever = build_retriever()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.retrieve("   ")


def test_retrieve_rejects_invalid_top_k() -> None:
    retriever = build_retriever()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            query="total",
            top_k=0,
        )


def test_minimum_score_filters_results() -> None:
    index = HierarchicalDocumentIndex(
        build_nodes()
    )

    retriever = SemanticBlockRetriever(
        document_index=index,
        model=FakeEncoder(),
        model_name="fake-model",
        minimum_score=0.9,
    )

    results = retriever.retrieve(
        query="What is the total amount?",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].node.node_type == NodeType.TABLE


def test_refresh_rebuilds_index() -> None:
    index = HierarchicalDocumentIndex(
        build_nodes()
    )

    retriever = SemanticBlockRetriever(
        document_index=index,
        model=FakeEncoder(),
        model_name="fake-model",
    )

    assert retriever.indexed_node_count == 3

    retriever.refresh()

    assert retriever.indexed_node_count == 3
    
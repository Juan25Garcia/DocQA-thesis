
"""Tests for retrievable node manifest generation."""

from types import SimpleNamespace

import pytest

from src.evaluation.run_node_manifest import (
    build_node_manifest,
)


def test_build_node_manifest():
    retriever = SimpleNamespace(
        indexed_nodes=[
            SimpleNamespace(
                document_id="doc_a",
                node_id="doc_a:table:1",
            ),
            SimpleNamespace(
                document_id="doc_a",
                node_id="doc_a:section:0",
            ),
            SimpleNamespace(
                document_id="doc_b",
                node_id="doc_b:paragraph:0",
            ),
        ]
    )

    manifest = build_node_manifest(retriever)

    assert manifest == {
        "doc_a": [
            "doc_a:section:0",
            "doc_a:table:1",
        ],
        "doc_b": [
            "doc_b:paragraph:0",
        ],
    }


def test_duplicate_node_id_rejected():
    retriever = SimpleNamespace(
        indexed_nodes=[
            SimpleNamespace(
                document_id="doc_a",
                node_id="doc_a:table:1",
            ),
            SimpleNamespace(
                document_id="doc_a",
                node_id="doc_a:table:1",
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate node_id",
    ):
        build_node_manifest(retriever)

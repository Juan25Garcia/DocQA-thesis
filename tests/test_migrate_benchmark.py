
"""Tests for benchmark annotation migration."""

import pytest

from src.evaluation.migrate_benchmark import (
    migrate_examples,
)


def sample_example():
    return {
        "id": "expense_001",
        "question": "What is the total expense?",
        "document_id": "sample_expense",
        "page_number": 1,
        "relevant_node_ids": [
            "sample_expense:table:1"
        ],
    }


def test_migrate_valid_example():
    manifest = {
        "sample_expense": [
            "sample_expense:table:1"
        ]
    }

    result = migrate_examples(
        [sample_example()],
        manifest,
    )

    assert len(result) == 1
    assert result[0]["example_id"] == "expense_001"
    assert result[0]["positive_node_ids"] == [
        "sample_expense:table:1"
    ]
    assert result[0]["page_number"] == 1
    assert "answer" not in result[0]


def test_reject_unknown_node():
    example = sample_example()

    manifest = {
        "sample_expense": [
            "sample_expense:table:0"
        ]
    }

    with pytest.raises(
        ValueError,
        match="not retrievable",
    ):
        migrate_examples([example], manifest)


def test_reject_duplicate_example_ids():
    example = sample_example()

    manifest = {
        "sample_expense": [
            "sample_expense:table:1"
        ]
    }

    with pytest.raises(
        ValueError,
        match="Duplicate example_id",
    ):
        migrate_examples(
            [example, example],
            manifest,
        )


def test_preserve_multiple_positive_nodes():
    example = sample_example()
    example["relevant_node_ids"] = [
        "sample_expense:table:0",
        "sample_expense:table:1",
    ]

    manifest = {
        "sample_expense": [
            "sample_expense:table:0",
            "sample_expense:table:1",
        ]
    }

    result = migrate_examples(
        [example],
        manifest,
    )

    assert len(
        result[0]["positive_node_ids"]
    ) == 2


"""Tests for benchmark annotation and document splitting."""

import json

import pytest

from src.evaluation.benchmark import (
    BenchmarkQuestion,
    create_document_splits,
    load_questions,
    validate_questions,
    validate_splits,
)


def make_question(
    example_id,
    document_id,
    positive_node_ids=("block:0",),
):
    return BenchmarkQuestion(
        example_id=example_id,
        document_id=document_id,
        question="What is the total amount?",
        positive_node_ids=positive_node_ids,
        answer="$100",
    )


def test_load_questions(tmp_path):
    path = tmp_path / "questions.jsonl"

    path.write_text(
        json.dumps({
            "example_id": "q1",
            "document_id": "invoice_a",
            "question": "What is the total?",
            "positive_node_ids": ["table:1"],
            "answer": "$100",
        }) + "\n",
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert len(questions) == 1
    assert questions[0].example_id == "q1"
    assert questions[0].positive_node_ids == (
        "table:1",
    )


def test_duplicate_question_ids():
    questions = [
        make_question("q1", "doc_a"),
        make_question("q1", "doc_b"),
    ]

    errors = validate_questions(questions)

    assert any(
        "Duplicate example_id" in error
        for error in errors
    )


def test_missing_positive_nodes():
    questions = [
        make_question("q1", "doc_a", ()),
    ]

    errors = validate_questions(questions)

    assert any(
        "no positive nodes" in error
        for error in errors
    )


def test_unknown_positive_node():
    questions = [
        make_question(
            "q1",
            "doc_a",
            ("missing_node",),
        ),
    ]

    errors = validate_questions(
        questions,
        document_nodes={
            "doc_a": {"block:0", "block:1"}
        },
    )

    assert any(
        "not retrievable" in error
        for error in errors
    )


def test_valid_node_references():
    questions = [
        make_question(
            "q1",
            "doc_a",
            ("block:0",),
        ),
    ]

    errors = validate_questions(
        questions,
        document_nodes={
            "doc_a": {"block:0", "block:1"}
        },
    )

    assert errors == []


def test_document_splits_are_disjoint():
    questions = [
        make_question(f"q{i}", f"doc_{i}")
        for i in range(12)
    ]

    splits = create_document_splits(
        questions,
        test_fraction=0.25,
        seed=42,
    )

    development = set(splits["development"])
    test = set(splits["test"])

    assert not development & test
    assert len(test) == 3
    assert validate_splits(questions, splits) == []


def test_splits_are_reproducible():
    questions = [
        make_question(f"q{i}", f"doc_{i}")
        for i in range(10)
    ]

    first = create_document_splits(
        questions,
        seed=42,
    )

    second = create_document_splits(
        list(reversed(questions)),
        seed=42,
    )

    assert first == second


def test_overlapping_splits_rejected():
    questions = [
        make_question("q1", "doc_a"),
        make_question("q2", "doc_b"),
    ]

    splits = {
        "development": ["doc_a", "doc_b"],
        "test": ["doc_b"],
    }

    errors = validate_splits(
        questions,
        splits,
    )

    assert any(
        "both splits" in error
        for error in errors
    )


def test_single_document_cannot_be_split():
    questions = [
        make_question("q1", "doc_a"),
    ]

    with pytest.raises(ValueError):
        create_document_splits(questions)

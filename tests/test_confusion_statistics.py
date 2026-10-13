
"""Tests for Issue #13 confusion statistics."""

import csv

import pytest

from src.analysis.confusion_statistics import (
    clustered_bootstrap,
    confusion_rate,
    load_confusions,
    rate_difference,
    summarize,
)


@pytest.fixture
def rows():
    return [
        {
            "example_id": "q1",
            "confusion": True,
            "same_document": True,
            "same_page": True,
            "same_type": False,
            "same_parent": True,
            "sibling": True,
        },
        {
            "example_id": "q1",
            "confusion": False,
            "same_document": False,
            "same_page": False,
            "same_type": True,
            "same_parent": False,
            "sibling": False,
        },
        {
            "example_id": "q2",
            "confusion": True,
            "same_document": True,
            "same_page": True,
            "same_type": False,
            "same_parent": True,
            "sibling": True,
        },
        {
            "example_id": "q2",
            "confusion": False,
            "same_document": False,
            "same_page": False,
            "same_type": True,
            "same_parent": False,
            "sibling": False,
        },
    ]


def test_confusion_rates(rows):
    assert confusion_rate(
        rows, "same_page", True
    ) == (2, 2, 1.0)

    assert confusion_rate(
        rows, "same_page", False
    ) == (0, 2, 0.0)


def test_rate_difference(rows):
    assert rate_difference(rows, "same_page") == 1.0
    assert rate_difference(rows, "same_type") == -1.0


def test_bootstrap_is_reproducible(rows):
    first = clustered_bootstrap(
        rows, "same_page", iterations=100, seed=42
    )
    second = clustered_bootstrap(
        rows, "same_page", iterations=100, seed=42
    )

    assert first == second
    assert first["lower"] == pytest.approx(1.0)
    assert first["upper"] == pytest.approx(1.0)


def test_summary_contains_all_features(rows):
    results = summarize(rows, iterations=100)

    assert len(results) == 5
    assert results[0]["feature"] == "same_document"
    assert results[1]["feature"] == "same_page"


def test_missing_group_returns_none(rows):
    only_true = [
        row for row in rows if row["same_page"]
    ]

    assert rate_difference(
        only_true, "same_page"
    ) is None


def test_load_csv(tmp_path):
    path = tmp_path / "confusions.csv"

    fields = [
        "example_id", "margin", "confusion",
        "positive_score", "candidate_score",
        "same_document", "same_page", "same_type",
        "same_parent", "sibling",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "example_id": "q1",
            "margin": "-0.1",
            "confusion": "True",
            "positive_score": "0.7",
            "candidate_score": "0.8",
            "same_document": "True",
            "same_page": "True",
            "same_type": "False",
            "same_parent": "True",
            "sibling": "True",
        })

    loaded = load_confusions(path)

    assert len(loaded) == 1
    assert loaded[0]["confusion"] is True
    assert loaded[0]["margin"] == pytest.approx(-0.1)


def test_rejects_inconsistent_confusion(tmp_path):
    path = tmp_path / "invalid.csv"

    path.write_text(
        "example_id,margin,confusion,positive_score,"
        "candidate_score,same_document,same_page,"
        "same_type,same_parent,sibling\n"
        "q1,-0.1,False,0.7,0.8,True,True,"
        "False,True,True\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Confusion flag"):
        load_confusions(path)

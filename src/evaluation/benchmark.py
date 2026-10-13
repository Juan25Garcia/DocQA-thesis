
"""Document-level benchmark schema, validation and splitting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BenchmarkQuestion:
    example_id: str
    document_id: str
    question: str
    positive_node_ids: tuple[str, ...]
    answer: str | None = None


def load_questions(path: Path) -> list[BenchmarkQuestion]:
    """Load questions from a UTF-8 JSONL file."""
    questions = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_number} must be an object"
                )

            required = {
                "example_id",
                "document_id",
                "question",
                "positive_node_ids",
            }

            missing = required - item.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number}: missing {sorted(missing)}"
                )

            positives = item["positive_node_ids"]

            if not isinstance(positives, list):
                raise ValueError(
                    f"Line {line_number}: "
                    "positive_node_ids must be a list"
                )

            answer = item.get("answer")

            if answer is not None and not isinstance(
                answer, str
            ):
                raise ValueError(
                    f"Line {line_number}: answer must be text"
                )

            questions.append(
                BenchmarkQuestion(
                    example_id=item["example_id"],
                    document_id=item["document_id"],
                    question=item["question"],
                    positive_node_ids=tuple(positives),
                    answer=answer,
                )
            )

    return questions


def validate_questions(
    questions: Sequence[BenchmarkQuestion],
    document_nodes: Mapping[str, set[str]] | None = None,
) -> list[str]:
    """
    Return validation errors.

    document_nodes maps document_id to its retrievable node IDs.
    Without it, structural validation still runs, but node
    existence cannot be verified.
    """
    errors = []

    if not questions:
        return ["Benchmark contains no questions"]

    counts = Counter(q.example_id for q in questions)

    for example_id, count in counts.items():
        if count > 1:
            errors.append(
                f"Duplicate example_id: {example_id}"
            )

    for q in questions:
        for field in (
            "example_id",
            "document_id",
            "question",
        ):
            value = getattr(q, field)

            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"{q.example_id!r}: invalid {field}"
                )

        positives = q.positive_node_ids

        if not positives:
            errors.append(
                f"{q.example_id}: no positive nodes"
            )

        if any(
            not isinstance(node_id, str)
            or not node_id.strip()
            for node_id in positives
        ):
            errors.append(
                f"{q.example_id}: invalid positive node ID"
            )

        if len(positives) != len(set(positives)):
            errors.append(
                f"{q.example_id}: duplicate positive nodes"
            )

        if document_nodes is not None:
            known_nodes = document_nodes.get(q.document_id)

            if known_nodes is None:
                errors.append(
                    f"{q.example_id}: unknown document "
                    f"{q.document_id}"
                )
                continue

            for node_id in positives:
                if node_id not in known_nodes:
                    errors.append(
                        f"{q.example_id}: node {node_id!r} "
                        f"is not retrievable in {q.document_id}"
                    )

    return errors


def _stable_order(document_id: str, seed: int) -> str:
    """Stable across Python processes and platforms."""
    value = f"{seed}:{document_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def create_document_splits(
    questions: Sequence[BenchmarkQuestion],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split unique documents, never individual questions.

    This is a development/evaluation split, not a
    stratified or statistically powered partition.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1"
        )

    documents = sorted({
        q.document_id
        for q in questions
    })

    if len(documents) < 2:
        raise ValueError(
            "At least two documents are required"
        )

    documents.sort(
        key=lambda document_id: _stable_order(
            document_id,
            seed,
        )
    )

    n_test = max(
        1,
        min(
            len(documents) - 1,
            round(len(documents) * test_fraction),
        ),
    )

    test_documents = sorted(documents[:n_test])
    development_documents = sorted(documents[n_test:])

    return {
        "development": development_documents,
        "test": test_documents,
    }


def validate_splits(
    questions: Sequence[BenchmarkQuestion],
    splits: Mapping[str, Sequence[str]],
) -> list[str]:
    errors = []

    development = set(splits.get("development", []))
    test = set(splits.get("test", []))

    overlap = development & test

    if overlap:
        errors.append(
            f"Documents in both splits: {sorted(overlap)}"
        )

    all_documents = {
        q.document_id
        for q in questions
    }

    assigned = development | test

    if not development or not test:
        errors.append(
            "Both development and test must be nonempty"
        )

    if all_documents - assigned:
        errors.append(
            "Unassigned documents: "
            f"{sorted(all_documents - assigned)}"
        )

    if assigned - all_documents:
        errors.append(
            "Unknown split documents: "
            f"{sorted(assigned - all_documents)}"
        )

    return errors


def save_splits(
    splits: Mapping[str, Sequence[str]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            dict(splits),
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

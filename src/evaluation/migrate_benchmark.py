
"""Migrate existing retrieval examples into the benchmark format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.benchmark import (
    BenchmarkQuestion,
    validate_questions,
)


def migrate_examples(
    examples: list[dict],
    node_manifest: dict[str, list[str]],
) -> list[dict]:
    """Convert and validate existing retrieval annotations."""
    migrated = []
    questions = []

    for index, example in enumerate(examples, start=1):
        required = {
            "id",
            "question",
            "document_id",
            "relevant_node_ids",
        }

        missing = required - example.keys()

        if missing:
            raise ValueError(
                f"Example {index}: missing {sorted(missing)}"
            )

        positive_ids = example["relevant_node_ids"]

        if not isinstance(positive_ids, list):
            raise ValueError(
                f"Example {index}: relevant_node_ids "
                "must be a list"
            )

        question = BenchmarkQuestion(
            example_id=example["id"],
            document_id=example["document_id"],
            question=example["question"],
            positive_node_ids=tuple(positive_ids),
            answer=example.get("answer"),
        )

        questions.append(question)

        record = {
            "example_id": question.example_id,
            "document_id": question.document_id,
            "question": question.question,
            "positive_node_ids": list(
                question.positive_node_ids
            ),
        }

        # Preserve existing annotations when available.
        if "page_number" in example:
            record["page_number"] = example["page_number"]

        if question.answer is not None:
            record["answer"] = question.answer

        migrated.append(record)

    errors = validate_questions(
        questions,
        document_nodes={
            document_id: set(node_ids)
            for document_id, node_ids in node_manifest.items()
        },
    )

    if errors:
        raise ValueError(
            "Migration validation failed:\n- "
            + "\n- ".join(errors)
        )

    return migrated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/evaluation/retrieval_examples.json"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/benchmark/node_manifest.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/benchmark/questions.jsonl"
        ),
    )

    args = parser.parse_args()

    examples = json.loads(
        args.input.read_text(encoding="utf-8-sig")
    )

    node_manifest = json.loads(
        args.manifest.read_text(encoding="utf-8-sig")
    )

    if not isinstance(examples, list):
        raise ValueError(
            "Retrieval examples must be a JSON array"
        )

    if not isinstance(node_manifest, dict):
        raise ValueError(
            "Node manifest must be a JSON object"
        )

    migrated = migrate_examples(
        examples,
        node_manifest,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Atomic replacement: never leave a partially
    # written benchmark if serialization fails.
    temporary_path = args.output.with_suffix(
        args.output.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in migrated:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )

    temporary_path.replace(args.output)

    print("\n=== BENCHMARK MIGRATION ===")
    print(f"Original examples: {len(examples)}")
    print(f"Migrated questions: {len(migrated)}")
    print(
        "Documents: "
        f"{len({q['document_id'] for q in migrated})}"
    )
    print("Node references: validated")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

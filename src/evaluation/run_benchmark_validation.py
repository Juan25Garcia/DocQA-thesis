
"""Validate benchmark annotations and document splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluation.benchmark import (
    create_document_splits,
    load_questions,
    save_splits,
    validate_questions,
    validate_splits,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/benchmark/questions.jsonl"),
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "data/benchmark/document_splits.json"
        ),
    )

    parser.add_argument(
        "--node-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping document IDs to "
            "lists of retrievable node IDs"
        ),
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    questions = load_questions(args.questions)

    document_nodes = None

    if args.node_manifest is not None:
        raw = json.loads(
            args.node_manifest.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(raw, dict) or any(
            not isinstance(nodes, list)
            for nodes in raw.values()
        ):
            raise ValueError(
                "Node manifest must map document IDs "
                "to lists of node IDs"
            )

        document_nodes = {
            doc_id: set(nodes)
            for doc_id, nodes in raw.items()
        }

    errors = validate_questions(
        questions,
        document_nodes=document_nodes,
    )

    if errors:
        print("\nBENCHMARK VALIDATION FAILED")

        for error in errors:
            print(f"  - {error}")

        raise SystemExit(1)

    if args.splits.exists():
        try:
            splits = json.loads(
                args.splits.read_text(
                    encoding="utf-8-sig"
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid split file: {args.splits}. "
                "The file may be empty or contain invalid JSON. "
                "Fix it before continuing."
            ) from exc
    else:
        splits = create_document_splits(
            questions,
            test_fraction=args.test_fraction,
        )

    split_errors = validate_splits(
        questions,
        splits,
    )

    if split_errors:
        print("\nSPLIT VALIDATION FAILED")

        for error in split_errors:
            print(f"  - {error}")

        raise SystemExit(1)

    if not args.splits.exists():
        save_splits(splits, args.splits)

    counts = Counter(
        q.document_id
        for q in questions
    )

    print("\n=== BENCHMARK VALIDATION ===")
    print("Status: PASS")
    print(f"Questions: {len(questions)}")
    print(f"Documents: {len(counts)}")
    print(
        "Development documents: "
        f"{len(splits['development'])}"
    )
    print(
        "Test documents: "
        f"{len(splits['test'])}"
    )

    for split_name in ("development", "test"):
        n_questions = sum(
            counts[doc_id]
            for doc_id in splits[split_name]
        )

        print(
            f"{split_name} questions: {n_questions}"
        )

    if document_nodes is None:
        print(
            "\nWarning: node existence was NOT checked. "
            "Supply --node-manifest for full validation."
        )
    else:
        print("\nNode existence: validated")

    print(f"\nSplits: {args.splits}")


if __name__ == "__main__":
    main()

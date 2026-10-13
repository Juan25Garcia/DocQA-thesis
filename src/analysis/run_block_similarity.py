
"""Enrich retrieval confusion records with block-to-block similarity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.analysis.block_similarity import BlockSimilarityCalculator
from src.evaluation.run_retrieval_evaluation import (
    build_block_retriever,
)
from src.retrievers.block_retriever import DEFAULT_BLOCK_MODEL


def enrich_rows(rows, calculator):
    """Add pairwise semantic similarity without changing original scores."""
    enriched = []

    for row in rows:
        updated = dict(row)

        updated["block_similarity"] = calculator.similarity(
            row["positive_node_id"],
            row["candidate_node_id"],
        )

        enriched.append(updated)

    return enriched


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/evaluation/retrieval_confusions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/retrieval_confusions_semantic.csv"
        ),
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_BLOCK_MODEL,
    )

    args = parser.parse_args()

    retriever = build_block_retriever(
        pdf_dir=args.pdf_dir,
        model_name=args.model,
    )

    calculator = BlockSimilarityCalculator(
        nodes=retriever.indexed_nodes,
        model=retriever._model,
    )

    with args.input.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required = {
        "positive_node_id",
        "candidate_node_id",
    }

    if not required.issubset(fieldnames):
        raise ValueError(
            f"Missing columns: {sorted(required - set(fieldnames))}"
        )

    enriched = enrich_rows(rows, calculator)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_fields = list(fieldnames)

    if "block_similarity" not in output_fields:
        output_fields.append("block_similarity")

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fields,
        )

        writer.writeheader()
        writer.writerows(enriched)

    print("\n=== BLOCK SIMILARITY ANALYSIS ===")
    print(f"Indexed blocks: {len(retriever.indexed_nodes)}")
    print(f"Comparisons enriched: {len(enriched)}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()

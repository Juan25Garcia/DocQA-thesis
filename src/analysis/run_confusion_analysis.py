
"""Run full-ranking block retrieval confusion analysis."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.analysis.retrieval_confusions import (
    RetrievalConfusionAnalyzer,
)
from src.analysis.structural_relations import (
    StructuralRelationExtractor,
)
from src.evaluation.run_retrieval_evaluation import (
    build_block_retriever,
    load_dataset,
)
from src.retrievers.block_retriever import DEFAULT_BLOCK_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze block retrieval confusions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/retrieval_examples.json"),
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/retrieval_confusions.csv"
        ),
    )
    return parser.parse_args()


def run_analysis(
    dataset: Path,
    pdf_dir: Path,
    model: str,
) -> tuple[list[dict], int]:
    examples = load_dataset(dataset)

    retriever = build_block_retriever(
        pdf_dir=pdf_dir,
        model_name=model,
    )

    # This analysis requires an unfiltered full ranking.
    if retriever.minimum_score is not None:
        raise ValueError(
            "Confusion analysis requires minimum_score=None."
        )

    extractor = StructuralRelationExtractor(
        retriever.document_index
    )

    analyzer = RetrievalConfusionAnalyzer(
        retriever=retriever,
        relation_extractor=extractor,
    )

    rows = []

    for index, example in enumerate(examples, start=1):
        example_id = example.metadata["id"]

        records = analyzer.analyze(example)

        for record in records:
            rows.append({
                "example_id": example_id,
                **record.to_dict(),
            })

        confusions = sum(
            record.confusion for record in records
        )

        ties = sum(
            record.tied for record in records
        )

        print(
            f"[{index}/{len(examples)}] "
            f"{example_id}: "
            f"{confusions} confusions, "
            f"{ties} ties, "
            f"{len(records)} negatives"
        )

    return rows, len(examples)


def save_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(
            "No negative candidates were produced."
        )

    fieldnames = list(rows[0].keys())

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict],
    question_count: int,
) -> None:
    confusion_rows = [
        row for row in rows if row["confusion"]
    ]

    affected_questions = {
        row["example_id"] for row in confusion_rows
    }

    print("\n" + "=" * 55)
    print("CONFUSION ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"Questions analyzed: {question_count}")
    print(f"Negative comparisons: {len(rows)}")
    print(f"Confusing negatives: {len(confusion_rows)}")
    print(f"Questions with confusion: {len(affected_questions)}")
    print(
        "Question-level confusion rate: "
        f"{len(affected_questions) / question_count:.2%}"
    )

    print("\nMost frequent structural relations among confusions:")

    for feature in (
        "same_document",
        "same_page",
        "same_type",
        "same_parent",
        "sibling",
    ):
        counts = Counter(
            row[feature] for row in confusion_rows
        )
        print(f"  {feature}: {dict(counts)}")


def main() -> None:
    args = parse_args()

    rows, question_count = run_analysis(
        dataset=args.dataset,
        pdf_dir=args.pdf_dir,
        model=args.model,
    )

    save_csv(rows, args.output)
    print_summary(rows, question_count)

    print(f"\nCSV saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

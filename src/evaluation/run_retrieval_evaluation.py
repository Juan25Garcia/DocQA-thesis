from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from src.demo import build_clip_retriever
from src.evaluation import (
    RetrievalEvaluationResult,
    RetrievalEvaluator,
    RetrievalExample,
)
from src.indexing import HierarchicalDocumentIndex
from src.models import DocumentNode
from src.parsers.docling_parser import DoclingParser
from src.retrievers.block_retriever import (
    DEFAULT_BLOCK_MODEL,
    SemanticBlockRetriever,
)


DEFAULT_DATASET_PATH = Path(
    "data/evaluation/retrieval_examples.json"
)

DEFAULT_PDF_DIR = Path("data/pdfs")

DEFAULT_OUTPUT_DIR = Path(
    "outputs/evaluation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate page-level or block-level "
            "document retrieval."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the retrieval evaluation dataset.",
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Directory containing evaluation PDFs.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of retrieved results.",
    )

    parser.add_argument(
        "--retriever",
        choices=(
            "block",
            "clip",
        ),
        default="block",
        help="Retriever to evaluate.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BLOCK_MODEL,
        help=(
            "Sentence Transformer model used "
            "for block retrieval."
        ),
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help=(
            "Show diagnostics for every question. "
            "By default only misses are displayed."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation results are saved.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional output name. Defaults to "
            "<retriever>_k<top-k>."
        ),
    )

    return parser.parse_args()


def load_dataset(
    path: Path,
) -> list[RetrievalExample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_examples: list[dict[str, Any]] = (
            json.load(file)
        )

    examples: list[RetrievalExample] = []

    for raw_example in raw_examples:
        document_id = raw_example[
            "document_id"
        ]
        page_number = int(
            raw_example["page_number"]
        )

        examples.append(
            RetrievalExample(
                question=raw_example["question"],
                relevant_node_ids=set(
                    raw_example[
                        "relevant_node_ids"
                    ]
                ),
                relevant_pages={
                    (
                        document_id,
                        page_number,
                    )
                },
                metadata={
                    "id": raw_example["id"],
                    "document_id": document_id,
                    "page_number": page_number,
                },
            )
        )

    return examples


def parse_documents(
    pdf_dir: Path,
) -> list[DocumentNode]:
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_dir}"
        )

    parser = DoclingParser()

    pdf_files = sorted(
        path
        for path in pdf_dir.glob("*.pdf")
        if path.is_file()
    )

    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in {pdf_dir}"
        )

    all_nodes: list[DocumentNode] = []

    for pdf_path in pdf_files:
        print(
            f"Parsing {pdf_path.name}..."
        )

        nodes = parser.parse(pdf_path)

        all_nodes.extend(nodes)

        print(
            f"  nodes: {len(nodes)}"
        )

    return all_nodes


def build_block_retriever(
    pdf_dir: Path,
    model_name: str,
) -> SemanticBlockRetriever:
    print("\nParsing document collection...")

    nodes = parse_documents(
        pdf_dir
    )

    print(
        f"\nTotal structured nodes: "
        f"{len(nodes)}"
    )

    print("\nBuilding hierarchical index...")

    document_index = HierarchicalDocumentIndex(
        nodes
    )

    print(
        f"Documents: "
        f"{len(document_index.document_ids)}"
    )

    print(
        f"Indexed nodes: "
        f"{document_index.node_count}"
    )

    print("\nBuilding semantic block index...")

    retriever = SemanticBlockRetriever(
        document_index=document_index,
        model_name=model_name,
    )

    print(
        f"Retrievable blocks: "
        f"{retriever.indexed_node_count}"
    )

    return retriever


def print_summary(
    result: RetrievalEvaluationResult,
) -> None:
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION")
    print("=" * 50)

    print(
        f"Retriever:        "
        f"{result.retriever_name}"
    )
    print(
        f"Evaluation level: "
        f"{result.evaluation_level}"
    )
    print(
        f"Examples:         "
        f"{result.example_count}"
    )
    print(
        f"Top K:            "
        f"{result.top_k}"
    )
    print(
        f"Hit Rate@K:       "
        f"{result.hit_rate_at_k:.4f}"
    )
    print(
        f"Recall@K:         "
        f"{result.recall_at_k:.4f}"
    )
    print(
        f"MRR@K:            "
        f"{result.mrr:.4f}"
    )
    print(
        f"Avg latency:      "
        f"{result.average_latency_ms:.2f} ms"
    )


def print_rank_distribution(
    result: RetrievalEvaluationResult,
) -> None:
    rank_counts = Counter(
        item.first_relevant_rank
        for item in result.example_results
        if item.first_relevant_rank is not None
    )

    misses = sum(
        item.first_relevant_rank is None
        for item in result.example_results
    )

    print("\n" + "=" * 50)
    print("FIRST RELEVANT RANK DISTRIBUTION")
    print("=" * 50)

    cumulative_hits = 0

    for rank in range(
        1,
        result.top_k + 1,
    ):
        count = rank_counts.get(
            rank,
            0,
        )

        cumulative_hits += count

        cumulative_rate = (
            cumulative_hits
            / result.example_count
        )

        print(
            f"Rank {rank:>2}: "
            f"{count:>2} question(s) "
            f"| cumulative hit rate: "
            f"{cumulative_rate:.4f}"
        )

    print(
        f"Miss:    "
        f"{misses:>2} question(s)"
    )


def print_diagnostics(
    result: RetrievalEvaluationResult,
    show_hits: bool = False,
) -> None:
    print("\n" + "=" * 50)
    print("PER-QUESTION DIAGNOSTICS")
    print("=" * 50)

    misses = 0

    for item in result.example_results:
        if (
            item.is_hit
            and not show_hits
        ):
            continue

        status = (
            "HIT"
            if item.is_hit
            else "MISS"
        )

        if not item.is_hit:
            misses += 1

        example_id = item.metadata.get(
            "id",
            "unknown",
        )

        print(
            f"\n[{status}] {example_id}"
        )

        print(
            f"Question: {item.question}"
        )

        if result.evaluation_level == "node":
            print("Expected nodes:")

            for node_id in sorted(
                item.relevant_node_ids
            ):
                print(
                    f"  - {node_id}"
                )

        else:
            print("Expected pages:")

            for (
                document_id,
                page_number,
            ) in sorted(
                item.relevant_pages
            ):
                print(
                    f"  - {document_id} "
                    f"| page {page_number}"
                )

        print("Retrieved:")

        for rank, (
            node_id,
            page,
            score,
        ) in enumerate(
            zip(
                item.retrieved_node_ids,
                item.retrieved_pages,
                item.retrieved_scores,
            ),
            start=1,
        ):
            document_id, page_number = page

            print(
                f"  {rank}. "
                f"{node_id} "
                f"| {document_id} "
                f"| page {page_number} "
                f"| score={score:.4f}"
            )

        if item.first_relevant_rank is None:
            print(
                "First relevant rank: not found"
            )
        else:
            print(
                "First relevant rank: "
                f"{item.first_relevant_rank}"
            )

        print(
            f"Recall: "
            f"{item.recall:.4f}"
        )

        print(
            f"Latency: "
            f"{item.latency_ms:.2f} ms"
        )

    print(
        f"\nMisses at K={result.top_k}: "
        f"{misses}/{result.example_count}"
    )


def build_run_metadata(
    args: argparse.Namespace,
    result: RetrievalEvaluationResult,
) -> dict[str, object]:
    return {
        "created_at": (
            datetime.now().astimezone().isoformat()
        ),
        "retriever": args.retriever,
        "retriever_name": result.retriever_name,
        "evaluation_level": (
            result.evaluation_level
        ),
        "top_k": result.top_k,
        "dataset": str(
            args.dataset.resolve()
        ),
        "pdf_dir": str(
            args.pdf_dir.resolve()
        ),
        "model": (
            args.model
            if args.retriever == "block"
            else None
        ),
    }


def save_json_result(
    result: RetrievalEvaluationResult,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    payload = {
        "run": metadata,
        "result": result.to_dict(),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_csv_result(
    result: RetrievalEvaluationResult,
    output_path: Path,
) -> None:
    fieldnames = [
        "example_id",
        "question",
        "expected_document_id",
        "expected_page_number",
        "evaluation_level",
        "hit",
        "recall",
        "first_relevant_rank",
        "reciprocal_rank",
        "latency_ms",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in result.example_results:
            writer.writerow(
                {
                    "example_id": (
                        item.metadata.get(
                            "id",
                            "",
                        )
                    ),
                    "question": item.question,
                    "expected_document_id": (
                        item.metadata.get(
                            "document_id",
                            "",
                        )
                    ),
                    "expected_page_number": (
                        item.metadata.get(
                            "page_number",
                            "",
                        )
                    ),
                    "evaluation_level": (
                        result.evaluation_level
                    ),
                    "hit": item.hit,
                    "recall": item.recall,
                    "first_relevant_rank": (
                        item.first_relevant_rank
                        if item.first_relevant_rank
                        is not None
                        else ""
                    ),
                    "reciprocal_rank": (
                        item.reciprocal_rank
                    ),
                    "latency_ms": (
                        item.latency_ms
                    ),
                }
            )


def save_summary_csv(
    result: RetrievalEvaluationResult,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    fieldnames = [
        "created_at",
        "retriever",
        "retriever_name",
        "evaluation_level",
        "top_k",
        "example_count",
        "hit_rate_at_k",
        "recall_at_k",
        "mrr_at_k",
        "average_latency_ms",
        "model",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                "created_at": metadata[
                    "created_at"
                ],
                "retriever": metadata[
                    "retriever"
                ],
                "retriever_name": (
                    result.retriever_name
                ),
                "evaluation_level": (
                    result.evaluation_level
                ),
                "top_k": result.top_k,
                "example_count": (
                    result.example_count
                ),
                "hit_rate_at_k": (
                    result.hit_rate_at_k
                ),
                "recall_at_k": (
                    result.recall_at_k
                ),
                "mrr_at_k": result.mrr,
                "average_latency_ms": (
                    result.average_latency_ms
                ),
                "model": metadata["model"],
            }
        )


def save_results(
    args: argparse.Namespace,
    result: RetrievalEvaluationResult,
) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name = (
        args.run_name
        or f"{args.retriever}_k{args.top_k}"
    )

    json_path = (
        output_dir
        / f"{run_name}.json"
    )

    csv_path = (
        output_dir
        / f"{run_name}_examples.csv"
    )

    summary_path = (
        output_dir
        / f"{run_name}_summary.csv"
    )

    metadata = build_run_metadata(
        args=args,
        result=result,
    )

    save_json_result(
        result=result,
        output_path=json_path,
        metadata=metadata,
    )

    save_csv_result(
        result=result,
        output_path=csv_path,
    )

    save_summary_csv(
        result=result,
        output_path=summary_path,
        metadata=metadata,
    )

    return (
        json_path,
        csv_path,
        summary_path,
    )


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    print("\nLoading evaluation dataset...")

    examples = load_dataset(
        args.dataset
    )

    print(
        f"Examples: {len(examples)}"
    )

    if args.retriever == "block":
        retriever = build_block_retriever(
            pdf_dir=args.pdf_dir,
            model_name=args.model,
        )

        evaluation_level = "node"

    else:
        print(
            "\nBuilding CLIP page retriever..."
        )

        retriever = build_clip_retriever()

        evaluation_level = "page"

    evaluator = RetrievalEvaluator(
        retriever=retriever,
        evaluation_level=evaluation_level,
    )

    print(
        f"\nEvaluating {args.retriever} "
        f"at K={args.top_k}..."
    )

    result = evaluator.evaluate(
        examples=examples,
        top_k=args.top_k,
    )

    print_summary(result)
    print_rank_distribution(result)

    print_diagnostics(
        result=result,
        show_hits=args.show_all,
    )

    (
        json_path,
        csv_path,
        summary_path,
    ) = save_results(
        args=args,
        result=result,
    )

    print("\n" + "=" * 50)
    print("RESULTS SAVED")
    print("=" * 50)

    print(
        f"JSON:     {json_path}"
    )
    print(
        f"Examples: {csv_path}"
    )
    print(
        f"Summary:  {summary_path}"
    )


if __name__ == "__main__":
    main()

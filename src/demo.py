from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.generators.qwen_generator import (
    QwenAnswerGenerator,
)
from src.models import DocumentNode, RetrievedEvidence
from src.parsers.docling_parser import DoclingParser
from src.parsers.page_parser import PageMetadataParser
from src.pipeline import DocumentQAPipeline
from src.retrieve_clip import (
    CLIP_METADATA_PATH,
    retrieve_pages,
)
from src.retrievers.clip_retriever import (
    ClipRetrieverAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CLIP and Qwen-based Document QA baseline "
            "with optional Docling structural parsing."
        )
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help=(
            "Question to ask over the "
            "document collection."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved pages.",
    )

    parser.add_argument(
        "--document",
        type=Path,
        help=(
            "Optional document to parse with Docling before "
            "retrieval. Parsed DocumentNode objects remain "
            "in memory."
        ),
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Only retrieve pages without "
            "generating an answer."
        ),
    )

    return parser.parse_args()


def build_retriever() -> ClipRetrieverAdapter:
    parser = PageMetadataParser(
        metadata_path=CLIP_METADATA_PATH
    )

    nodes = parser.parse_all()

    return ClipRetrieverAdapter(
        nodes=nodes,
        search_function=retrieve_pages,
    )


def print_evidence(
    evidence: Sequence[RetrievedEvidence],
) -> None:
    print("\nTop retrieved pages:")

    if not evidence:
        print("No pages were retrieved.")
        return

    for item in evidence:
        node = item.node

        print(
            f"{item.rank}. "
            f"{node.document_id} "
            f"| page {node.page_number} "
            f"| score={item.score:.4f} "
            f"| {node.image_path}"
        )


def print_structured_summary(
    nodes: Sequence[DocumentNode],
) -> None:
    print("\nStructured document:")

    if not nodes:
        print("No structured nodes were generated.")
        return

    node_counts: dict[str, int] = {}

    for node in nodes:
        node_type = node.node_type.value
        node_counts[node_type] = (
            node_counts.get(node_type, 0) + 1
        )

    print(f"Total nodes: {len(nodes)}")

    for node_type, count in sorted(
        node_counts.items()
    ):
        print(f"- {node_type}: {count}")


def print_hierarchical_index_summary(
    pipeline: DocumentQAPipeline,
) -> None:
    index = pipeline.document_index

    print("\nHierarchical index:")

    if index is None:
        print("No hierarchical index was built.")
        return

    print(f"Indexed nodes: {index.node_count}")
    print(
        "Documents: "
        f"{', '.join(index.document_ids)}"
    )

    for document_id in index.document_ids:
        root = index.get_root(document_id)
        pages = index.get_children(root.node_id)
        content_nodes = index.get_content_nodes(document_id)

        print(f"- {document_id}:")
        print(f"  pages: {len(pages)}")
        print(f"  content nodes: {len(content_nodes)}")

        counts: dict[str, int] = {}

        for node in content_nodes:
            node_type = node.node_type.value
            counts[node_type] = counts.get(node_type, 0) + 1

        for node_type, count in sorted(counts.items()):
            print(f"  {node_type}: {count}")


def main() -> None:
    args = parse_args()

    retriever = build_retriever()
    document_parser = DoclingParser()

    generator = (
        None
        if args.retrieval_only
        else QwenAnswerGenerator()
    )

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        generator=generator,
        document_parser=document_parser,
    )

    if args.document is not None:
        print("\nParsing structured document...")

        structured_nodes = pipeline.ingest(
            args.document
        )

        print_structured_summary(
            structured_nodes
        )

    print_hierarchical_index_summary(
    pipeline
)

    print("\nQuestion:")
    print(args.question)

    print("\nRetrieving pages...")

    if args.retrieval_only:
        evidence = pipeline.retrieve(
            question=args.question,
            top_k=args.top_k,
        )

        print_evidence(evidence)
        return

    result = pipeline.ask(
        question=args.question,
        top_k=args.top_k,
    )

    print_evidence(result.evidence)

    print("\nAnswer:")
    print(result.answer)

    print("\nEvidence pages:")

    for item in result.evidence:
        node = item.node

        print(
            f"- {node.document_id} "
            f"| page {node.page_number}"
        )

    print("\nPipeline metadata:")

    for key, value in result.metadata.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
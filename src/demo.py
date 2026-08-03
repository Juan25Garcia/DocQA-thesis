from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.generators.qwen_generator import (
    QwenAnswerGenerator,
)
from src.indexing import HierarchicalDocumentIndex
from src.models import DocumentNode, RetrievedEvidence
from src.parsers.docling_parser import DoclingParser
from src.parsers.page_parser import PageMetadataParser
from src.pipeline import DocumentQAPipeline
from src.retrieve_clip import (
    CLIP_METADATA_PATH,
    retrieve_pages,
)
from src.retrievers.block_retriever import (
    DEFAULT_BLOCK_MODEL,
    SemanticBlockRetriever,
)
from src.retrievers.clip_retriever import (
    ClipRetrieverAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Document QA pipeline using either "
            "page-level CLIP retrieval or semantic block retrieval."
        )
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask over the document collection.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved results.",
    )

    parser.add_argument(
        "--document",
        type=Path,
        help=(
            "Document to parse with Docling. "
            "Required when using --retriever block."
        ),
    )

    parser.add_argument(
        "--retriever",
        choices=("clip", "block"),
        default="clip",
        help=(
            "Retrieval strategy: CLIP retrieves complete pages; "
            "block retrieves structured document nodes."
        ),
    )

    parser.add_argument(
        "--block-model",
        type=str,
        default=DEFAULT_BLOCK_MODEL,
        help=(
            "Sentence Transformer model used by the "
            "semantic block retriever."
        ),
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help=(
            "Optional minimum similarity score for "
            "semantic block retrieval."
        ),
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Retrieve evidence without generating "
            "an answer with Qwen."
        ),
    )

    return parser.parse_args()


def build_clip_retriever() -> ClipRetrieverAdapter:
    metadata_parser = PageMetadataParser(
        metadata_path=CLIP_METADATA_PATH
    )

    nodes = metadata_parser.parse_all()

    return ClipRetrieverAdapter(
        nodes=nodes,
        search_function=retrieve_pages,
    )


def build_block_retriever(
    document_path: Path,
    model_name: str,
    minimum_score: float | None,
) -> tuple[
    SemanticBlockRetriever,
    list[DocumentNode],
    HierarchicalDocumentIndex,
]:
    parser = DoclingParser()

    print("\nParsing structured document...")

    nodes = parser.parse(document_path)

    document_index = HierarchicalDocumentIndex(
        nodes
    )

    print_structured_summary(nodes)
    print_hierarchical_index_summary(
        document_index
    )

    print("\nBuilding semantic block index...")

    retriever = SemanticBlockRetriever(
        document_index=document_index,
        model_name=model_name,
        minimum_score=minimum_score,
    )

    print(
        "Indexed retrievable blocks: "
        f"{retriever.indexed_node_count}"
    )

    return retriever, nodes, document_index


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
    document_index: HierarchicalDocumentIndex,
) -> None:
    print("\nHierarchical index:")
    print(
        f"Indexed nodes: "
        f"{document_index.node_count}"
    )
    print(
        "Documents: "
        f"{', '.join(document_index.document_ids)}"
    )

    for document_id in document_index.document_ids:
        root = document_index.get_root(
            document_id
        )
        pages = document_index.get_children(
            root.node_id
        )
        content_nodes = (
            document_index.get_content_nodes(
                document_id
            )
        )

        print(f"- {document_id}:")
        print(f"  pages: {len(pages)}")
        print(
            f"  content nodes: "
            f"{len(content_nodes)}"
        )

        counts: dict[str, int] = {}

        for node in content_nodes:
            node_type = node.node_type.value
            counts[node_type] = (
                counts.get(node_type, 0) + 1
            )

        for node_type, count in sorted(
            counts.items()
        ):
            print(f"  {node_type}: {count}")


def print_clip_evidence(
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


def print_block_evidence(
    evidence: Sequence[RetrievedEvidence],
) -> None:
    print("\nTop retrieved blocks:")

    if not evidence:
        print("No blocks were retrieved.")
        return

    for item in evidence:
        node = item.node

        print(
            f"\n{item.rank}. "
            f"{node.node_type.value} "
            f"| {node.document_id} "
            f"| page {node.page_number} "
            f"| score={item.score:.4f}"
        )

        print(f"   node_id: {node.node_id}")

        block_lines = node.text.splitlines()

        for line in block_lines:
            print(f"   {line}")


def run_clip_mode(
    args: argparse.Namespace,
) -> None:
    retriever = build_clip_retriever()

    print("\nQuestion:")
    print(args.question)

    print("\nRetrieving pages...")

    if args.retrieval_only:
        evidence = retriever.retrieve(
            query=args.question,
            top_k=args.top_k,
        )

        print_clip_evidence(evidence)
        return

    generator = QwenAnswerGenerator()

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        generator=generator,
    )

    if args.document is not None:
        pipeline.document_parser = (
            DoclingParser()
        )

        structured_nodes = pipeline.ingest(
            args.document
        )

        print_structured_summary(
            structured_nodes
        )

        if pipeline.document_index is not None:
            print_hierarchical_index_summary(
                pipeline.document_index
            )

    result = pipeline.ask(
        question=args.question,
        top_k=args.top_k,
    )

    print_clip_evidence(result.evidence)

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


def run_block_mode(
    args: argparse.Namespace,
) -> None:
    if args.document is None:
        raise ValueError(
            "--document is required when using "
            "--retriever block."
        )

    if not args.retrieval_only:
        raise ValueError(
            "Block retrieval currently supports only "
            "--retrieval-only. Textual block evidence "
            "has not yet been connected to the generator."
        )

    retriever, _, _ = build_block_retriever(
        document_path=args.document,
        model_name=args.block_model,
        minimum_score=args.minimum_score,
    )

    print("\nQuestion:")
    print(args.question)

    print("\nRetrieving structured blocks...")

    evidence = retriever.retrieve(
        query=args.question,
        top_k=args.top_k,
    )

    print_block_evidence(evidence)


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if args.retriever == "block":
        run_block_mode(args)
        return

    run_clip_mode(args)


if __name__ == "__main__":
    main()
import argparse

from src.generators.qwen_generator import (
    QwenAnswerGenerator,
)
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
            "Run the CLIP and Qwen-based "
            "Document QA baseline."
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


def print_evidence(evidence) -> None:
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


def main() -> None:
    args = parse_args()

    retriever = build_retriever()

    print("\nQuestion:")
    print(args.question)

    print("\nRetrieving pages...")

    if args.retrieval_only:
        evidence = retriever.retrieve(
            query=args.question,
            top_k=args.top_k,
        )

        print_evidence(evidence)
        return

    generator = QwenAnswerGenerator()

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        generator=generator,
    )

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


if __name__ == "__main__":
    main()

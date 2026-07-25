import argparse

from src.answer import QwenVLAnswerer
from src.retrieve_clip import retrieve_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        help="Number of retrieved pages.",
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only retrieve pages without generating an answer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\nRetrieving pages...")

    retrieved_pages = retrieve_pages(
        query=args.question,
        top_k=args.top_k,
    )

    print("\nQuestion:")
    print(args.question)

    print("\nTop retrieved pages:")

    for item in retrieved_pages:
        print(
            f"{item['rank']}. {item['doc_id']} | page {item['page_num']} "
            f"| score={item['score']:.4f} | {item['image_path']}"
        )

    if args.retrieval_only:
        return

    image_paths = [item["image_path"] for item in retrieved_pages]

    print("\nGenerating answer...")

    answerer = QwenVLAnswerer()
    answer = answerer.answer(
        question=args.question,
        image_paths=image_paths,
    )

    print("\nAnswer:")
    print(answer)

    print("\nEvidence pages:")
    for item in retrieved_pages:
        print(f"- {item['doc_id']} | page {item['page_num']}")


if __name__ == "__main__":
    main()

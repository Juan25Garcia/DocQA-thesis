
"""Export a manifest of retrievable document nodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.run_retrieval_evaluation import (
    build_block_retriever,
)


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_node_manifest(retriever):
    """Map each document to its retrievable node IDs."""
    manifest = defaultdict(list)
    seen = set()

    for node in retriever.indexed_nodes:
        document_id = node.document_id
        node_id = node.node_id

        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Node has an invalid document_id")

        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Node has an invalid node_id")

        if node_id in seen:
            raise ValueError(f"Duplicate node_id: {node_id}")

        seen.add(node_id)
        manifest[document_id].append(node_id)

    return {
        document_id: sorted(node_ids)
        for document_id, node_ids in sorted(manifest.items())
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/node_manifest.json"),
    )

    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        raise ValueError(
            f"PDF directory not found: {args.pdf_dir}"
        )

    retriever = build_block_retriever(
        pdf_dir=args.pdf_dir,
        model_name=args.model,
    )

    manifest = build_node_manifest(retriever)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n=== NODE MANIFEST ===")
    print(f"Documents: {len(manifest)}")

    for document_id, node_ids in manifest.items():
        print(f"  {document_id}: {len(node_ids)} blocks")

    print(
        f"Total retrievable blocks: "
        f"{sum(len(nodes) for nodes in manifest.values())}"
    )

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()

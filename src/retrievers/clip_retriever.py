from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.models import DocumentNode, RetrievedEvidence


ClipSearchFunction = Callable[
    [str, int],
    list[dict],
]


class ClipRetrieverAdapter:
    """
    Adapter around the existing retrieve_pages function.

    The existing CLIP retriever returns dictionaries such as:

    {
        "doc_id": "invoice",
        "page_num": 2,
        "image_path": "...",
        "rank": 1,
        "score": 0.94
    }

    This adapter converts them into RetrievedEvidence objects.
    """

    def __init__(
        self,
        nodes: list[DocumentNode],
        search_function: ClipSearchFunction,
    ) -> None:
        self.search_function = search_function

        self.nodes_by_id = {
            node.node_id: node
            for node in nodes
        }

        self.nodes_by_image_path = {
            self._normalize_path(node.image_path): node
            for node in nodes
            if node.image_path is not None
        }

    @property
    def name(self) -> str:
        return "clip_page_retriever"

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        raw_results = self.search_function(
            normalized_query,
            top_k,
        )

        evidence: list[RetrievedEvidence] = []

        for fallback_rank, raw_result in enumerate(
            raw_results,
            start=1,
        ):
            node = self._resolve_node(raw_result)

            rank = int(
                raw_result.get(
                    "rank",
                    fallback_rank,
                )
            )

            score = float(raw_result["score"])

            evidence.append(
                RetrievedEvidence(
                    node=node,
                    score=score,
                    rank=rank,
                    retriever_name=self.name,
                    metadata={
                        "original_result": dict(raw_result),
                    },
                )
            )

        return evidence

    def _resolve_node(
        self,
        result: dict,
    ) -> DocumentNode:
        document_id = str(
            result.get("doc_id", "")
        ).strip()

        page_value = result.get("page_num")

        if document_id and page_value is not None:
            page_number = int(page_value)

            node_id = (
                f"{document_id}:page:{page_number}"
            )

            node = self.nodes_by_id.get(node_id)

            if node is not None:
                return node

        image_path_value = result.get("image_path")

        if image_path_value:
            normalized_path = self._normalize_path(
                Path(str(image_path_value))
            )

            node = self.nodes_by_image_path.get(
                normalized_path
            )

            if node is not None:
                return node

        raise KeyError(
            "A CLIP retrieval result could not be matched "
            "to a DocumentNode. "
            f"Result: {result}"
        )

    @staticmethod
    def _normalize_path(
        path: Path | None,
    ) -> str:
        if path is None:
            return ""

        return str(
            path.resolve()
        ).replace("\\", "/").lower()

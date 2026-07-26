from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.models import DocumentNode, RetrievedEvidence


class ClipRetrieverAdapter:
    def __init__(
        self,
        nodes: list[DocumentNode],
        search_function: Callable[
            [str, int],
            list[dict],
        ],
    ):
        self.nodes_by_id = {
            node.node_id: node
            for node in nodes
        }
        self.search_function = search_function

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
            raise ValueError("top_k must be greater than zero.")

        raw_results = self.search_function(
            normalized_query,
            top_k,
        )

        evidence: list[RetrievedEvidence] = []

        for rank, raw_result in enumerate(
            raw_results,
            start=1,
        ):
            node = self._resolve_node(raw_result)

            evidence.append(
                RetrievedEvidence(
                    node=node,
                    score=float(raw_result["score"]),
                    rank=rank,
                    retriever_name=self.name,
                    metadata={
                        key: value
                        for key, value in raw_result.items()
                        if key not in {
                            "node_id",
                            "score",
                            "image_path",
                            "page_number",
                            "document_id",
                        }
                    },
                )
            )

        return evidence

    def _resolve_node(
        self,
        result: dict,
    ) -> DocumentNode:
        node_id = result.get("node_id")

        if node_id and node_id in self.nodes_by_id:
            return self.nodes_by_id[node_id]

        document_id = result.get("document_id")
        page_number = result.get("page_number")

        if document_id is not None and page_number is not None:
            generated_id = (
                f"{Path(str(document_id)).stem}"
                f":page:{int(page_number)}"
            )

            if generated_id in self.nodes_by_id:
                return self.nodes_by_id[generated_id]

        image_path = result.get("image_path")

        if image_path:
            normalized_path = str(image_path).replace("\\", "/")

            for node in self.nodes_by_id.values():
                if node.image_path is None:
                    continue

                node_path = str(node.image_path).replace("\\", "/")

                if node_path == normalized_path:
                    return node

        raise KeyError(
            "The retrieved result could not be associated "
            "with a metadata node."
        )

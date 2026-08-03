from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np

from src.indexing import HierarchicalDocumentIndex
from src.models import (
    DocumentNode,
    NodeType,
    RetrievedEvidence,
)


DEFAULT_BLOCK_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_RETRIEVABLE_TYPES = frozenset(
    {
        NodeType.TITLE,
        NodeType.SECTION,
        NodeType.PARAGRAPH,
        NodeType.TABLE,
        NodeType.FIGURE,
        NodeType.KEY_VALUE,
    }
)


class TextEncoderProtocol(Protocol):
    def encode(
        self,
        sentences: str | Sequence[str],
        **kwargs: Any,
    ) -> Any:
        ...


class SemanticBlockRetriever:
    """
    Retrieve structured document blocks using semantic similarity.

    The retriever indexes textual content from DocumentNode objects and
    returns RetrievedEvidence instances ranked by cosine similarity.
    """

    def __init__(
        self,
        document_index: HierarchicalDocumentIndex,
        model_name: str = DEFAULT_BLOCK_MODEL,
        model: TextEncoderProtocol | None = None,
        retrievable_types: Iterable[NodeType] | None = None,
        minimum_score: float | None = None,
    ) -> None:
        self.document_index = document_index
        self.model_name = model_name
        self.minimum_score = minimum_score

        self.retrievable_types = frozenset(
            retrievable_types
            if retrievable_types is not None
            else DEFAULT_RETRIEVABLE_TYPES
        )

        self._model = (
            model
            if model is not None
            else self._load_model(model_name)
        )

        self._nodes: list[DocumentNode] = []
        self._embeddings = np.empty(
            shape=(0, 0),
            dtype=np.float32,
        )

        self.build()

    @property
    def name(self) -> str:
        return "semantic_block_retriever"

    @property
    def indexed_node_count(self) -> int:
        return len(self._nodes)

    @property
    def indexed_nodes(self) -> list[DocumentNode]:
        return list(self._nodes)

    def build(self) -> None:
        """
        Build the block embedding index from the hierarchical index.
        """
        candidate_nodes = (
            self.document_index.get_content_nodes()
        )

        self._nodes = [
            node
            for node in candidate_nodes
            if self._is_retrievable(node)
        ]

        if not self._nodes:
            self._embeddings = np.empty(
                shape=(0, 0),
                dtype=np.float32,
            )
            return

        texts = [
            self._prepare_node_text(node)
            for node in self._nodes
        ]

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self._embeddings = self._ensure_matrix(
            embeddings
        )

        if len(self._embeddings) != len(self._nodes):
            raise RuntimeError(
                "The encoder returned a different number of "
                "embeddings than indexed nodes."
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        """
        Return the blocks most semantically similar to a query.
        """
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not self._nodes:
            return []

        query_embedding = self._model.encode(
            normalized_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_vector = self._ensure_vector(
            query_embedding
        )

        if query_vector.shape[0] != self._embeddings.shape[1]:
            raise RuntimeError(
                "Query embedding dimension does not match "
                "the block index dimension."
            )

        # Embeddings are normalized, therefore their dot product
        # is equivalent to cosine similarity.
        scores = self._embeddings @ query_vector

        ranked_indexes = np.argsort(
            scores
        )[::-1]

        evidence: list[RetrievedEvidence] = []

        for node_index in ranked_indexes:
            score = float(scores[node_index])

            if (
                self.minimum_score is not None
                and score < self.minimum_score
            ):
                continue

            node = self._nodes[int(node_index)]

            evidence.append(
                RetrievedEvidence(
                    node=node,
                    score=score,
                    rank=len(evidence) + 1,
                    retriever_name=self.name,
                    metadata={
                        "model_name": self.model_name,
                        "node_type": node.node_type.value,
                        "page_number": node.page_number,
                    },
                )
            )

            if len(evidence) >= top_k:
                break

        return evidence

    def refresh(self) -> None:
        """
        Rebuild embeddings after the hierarchical index changes.
        """
        self.build()

    def _is_retrievable(
        self,
        node: DocumentNode,
    ) -> bool:
        if node.node_type not in self.retrievable_types:
            return False

        return bool(
            self._prepare_node_text(node)
        )

    @staticmethod
    def _prepare_node_text(
        node: DocumentNode,
    ) -> str:
        text = node.text.strip()

        if not text:
            return ""

        node_type = node.node_type.value.replace(
            "_",
            " ",
        )

        return f"{node_type}: {text}"

    @staticmethod
    def _load_model(
        model_name: str,
    ) -> TextEncoderProtocol:
        from sentence_transformers import (
            SentenceTransformer,
        )

        return SentenceTransformer(model_name)

    @staticmethod
    def _ensure_matrix(
        embeddings: Any,
    ) -> np.ndarray:
        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise RuntimeError(
                "Block embeddings must be a two-dimensional array."
            )

        return array

    @staticmethod
    def _ensure_vector(
        embedding: Any,
    ) -> np.ndarray:
        array = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if array.ndim == 2:
            if array.shape[0] != 1:
                raise RuntimeError(
                    "The query encoder returned multiple embeddings."
                )

            array = array[0]

        if array.ndim != 1:
            raise RuntimeError(
                "The query embedding must be one-dimensional."
            )

        return array

"""Compute semantic similarity between pairs of document blocks."""

from __future__ import annotations

import numpy as np

from src.models import DocumentNode
from src.retrievers.block_retriever import (
    DEFAULT_BLOCK_MODEL,
    SemanticBlockRetriever,
)


class BlockSimilarityCalculator:
    """Compute cosine similarities using cached block embeddings."""

    def __init__(
        self,
        nodes: list[DocumentNode],
        model_name: str = DEFAULT_BLOCK_MODEL,
        model=None,
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}

        if len(self.nodes) != len(nodes):
            raise ValueError("Duplicate node IDs detected.")

        self.model = (
            model
            if model is not None
            else SemanticBlockRetriever._load_model(model_name)
        )

        self.embeddings: dict[str, np.ndarray] = {}

        if nodes:
            texts = [
                SemanticBlockRetriever._prepare_node_text(node)
                for node in nodes
            ]

            vectors = np.asarray(
                self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )

            if vectors.ndim != 2 or len(vectors) != len(nodes):
                raise RuntimeError(
                    "Invalid encoder output dimensions."
                )

            norms = np.linalg.norm(vectors, axis=1)

            if not np.all(np.isfinite(vectors)):
                raise ValueError("Non-finite embeddings detected.")

            if np.any(norms == 0):
                raise ValueError("Zero-length embedding detected.")

            vectors = vectors / norms[:, None]

            self.embeddings = {
                node.node_id: vector
                for node, vector in zip(nodes, vectors)
            }

    def similarity(
        self,
        positive_node_id: str,
        candidate_node_id: str,
    ) -> float:
        """Cosine similarity between two indexed blocks."""
        positive = self.embeddings[positive_node_id]
        candidate = self.embeddings[candidate_node_id]

        return float(np.dot(positive, candidate))

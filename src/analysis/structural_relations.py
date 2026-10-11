
"""Deterministic structural relationships between indexed document nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.indexing import HierarchicalDocumentIndex
from src.models import DocumentNode


@dataclass(frozen=True)
class StructuralRelations:
    same_document: bool
    same_page: bool | None
    same_type: bool
    same_parent: bool
    sibling: bool
    hierarchical_distance: int | None

    def to_dict(self) -> dict[str, bool | int | None]:
        return asdict(self)


class StructuralRelationExtractor:
    """Compare nodes using validated hierarchical relationships."""

    def __init__(self, index: HierarchicalDocumentIndex) -> None:
        self.index = index

    def compare(
        self,
        positive_node_id: str,
        candidate_node_id: str,
    ) -> dict[str, bool | int | None]:
        positive = self.index.get_node(positive_node_id)
        candidate = self.index.get_node(candidate_node_id)

        same_document = positive.document_id == candidate.document_id

        same_parent = (
            same_document
            and positive.parent_id is not None
            and positive.parent_id == candidate.parent_id
        )

        if not same_document:
            same_page = False
        elif positive.page_number is None or candidate.page_number is None:
            same_page = None
        else:
            same_page = positive.page_number == candidate.page_number

        relations = StructuralRelations(
            same_document=same_document,
            same_page=same_page,
            same_type=positive.node_type == candidate.node_type,
            same_parent=same_parent,
            sibling=(
                same_parent
                and positive.node_id != candidate.node_id
            ),
            hierarchical_distance=self._distance(
                positive, candidate
            ),
        )

        return relations.to_dict()

    def _distance(
        self,
        positive: DocumentNode,
        candidate: DocumentNode,
    ) -> int | None:
        if positive.document_id != candidate.document_id:
            return None

        positive_path = [
            positive,
            *self.index.get_ancestors(positive.node_id),
        ]
        candidate_path = [
            candidate,
            *self.index.get_ancestors(candidate.node_id),
        ]

        candidate_depths = {
            node.node_id: depth
            for depth, node in enumerate(candidate_path)
        }

        for depth, node in enumerate(positive_path):
            if node.node_id in candidate_depths:
                return depth + candidate_depths[node.node_id]

        return None

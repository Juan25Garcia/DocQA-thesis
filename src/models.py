from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeType(str, Enum):
    DOCUMENT = "document"
    PAGE = "page"
    SECTION = "section"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    KEY_VALUE = "key_value"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class DocumentNode:
    node_id: str
    document_id: str
    node_type: NodeType

    page_number: int | None = None
    text: str = ""
    image_path: Path | None = None
    bbox: BoundingBox | None = None

    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "document_id": self.document_id,
            "node_type": self.node_type.value,
            "page_number": self.page_number,
            "text": self.text,
            "image_path": (
                str(self.image_path)
                if self.image_path is not None
                else None
            ),
            "bbox": (
                self.bbox.to_list()
                if self.bbox is not None
                else None
            ),
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
        }


@dataclass
class RetrievedEvidence:
    node: DocumentNode
    score: float
    rank: int
    retriever_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "retriever_name": self.retriever_name,
            "node": self.node.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    evidence: list[RetrievedEvidence]
    generator_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "generator_name": self.generator_name,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "metadata": self.metadata,
        }

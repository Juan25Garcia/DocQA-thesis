from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from src.indexing import HierarchicalDocumentIndex
from src.models import (
    AnswerResult,
    DocumentNode,
    NodeType,
    RetrievedEvidence,
)


class DocumentParserProtocol(Protocol):
    def parse(
        self,
        input_path: Path,
    ) -> list[DocumentNode]:
        ...


class RetrieverProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        ...


class GeneratorProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> AnswerResult:
        ...


class DocumentQAPipeline:
    def __init__(
        self,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol | None = None,
        document_parser: DocumentParserProtocol | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.document_parser = document_parser

        self.document_nodes: list[DocumentNode] = []
        self.document_index: HierarchicalDocumentIndex | None = None

    def ingest(
        self,
        document_path: Path | str,
    ) -> list[DocumentNode]:
        """
        Parse and structurally index a document.

        The parsed DocumentNode objects and the hierarchical index are kept
        in memory. No intermediate JSON file is required.
        """
        if self.document_parser is None:
            raise RuntimeError(
                "A document parser must be configured before ingestion."
            )

        normalized_path = Path(document_path).resolve()

        if not normalized_path.exists():
            raise FileNotFoundError(
                f"Input document not found: {normalized_path}"
            )

        if not normalized_path.is_file():
            raise ValueError(
                f"Input path is not a file: {normalized_path}"
            )

        nodes = list(
            self.document_parser.parse(normalized_path)
        )

        index = HierarchicalDocumentIndex(nodes)

        self.document_nodes = nodes
        self.document_index = index

        return self.document_nodes

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        normalized_question = self._validate_question(
            question
        )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        return self.retriever.retrieve(
            query=normalized_question,
            top_k=top_k,
        )

    def ask(
        self,
        question: str,
        top_k: int = 3,
        document_path: Path | str | None = None,
    ) -> AnswerResult:
        normalized_question = self._validate_question(
            question
        )

        if document_path is not None:
            self.ingest(document_path)

        evidence = self.retrieve(
            question=normalized_question,
            top_k=top_k,
        )

        structured_metadata = self._structured_metadata()

        if not evidence:
            generator_name = (
                self.generator.name
                if self.generator is not None
                else "not-configured"
            )

            return AnswerResult(
                question=normalized_question,
                answer=(
                    "No relevant evidence was found "
                    "in the indexed documents."
                ),
                evidence=[],
                generator_name=generator_name,
                metadata={
                    "retriever": self.retriever.name,
                    "top_k": top_k,
                    "retrieval_empty": True,
                    **structured_metadata,
                },
            )

        if self.generator is None:
            raise RuntimeError(
                "A generator must be configured before asking "
                "for an answer."
            )

        result = self.generator.generate(
            question=normalized_question,
            evidence=evidence,
        )

        result.metadata.update(
            {
                "retriever": self.retriever.name,
                "top_k": top_k,
                **structured_metadata,
            }
        )

        return result

    def get_structured_node(
        self,
        node_id: str,
    ) -> DocumentNode:
        return self._require_document_index().get_node(
            node_id
        )

    def get_structured_root(
        self,
        document_id: str,
    ) -> DocumentNode:
        return self._require_document_index().get_root(
            document_id
        )

    def get_structured_children(
        self,
        node_id: str,
    ) -> list[DocumentNode]:
        return self._require_document_index().get_children(
            node_id
        )

    def get_structured_ancestors(
        self,
        node_id: str,
        include_self: bool = False,
    ) -> list[DocumentNode]:
        return self._require_document_index().get_ancestors(
            node_id=node_id,
            include_self=include_self,
        )

    def get_structured_descendants(
        self,
        node_id: str,
        include_self: bool = False,
        max_depth: int | None = None,
    ) -> list[DocumentNode]:
        return self._require_document_index().get_descendants(
            node_id=node_id,
            include_self=include_self,
            max_depth=max_depth,
        )

    def get_structured_nodes_by_type(
        self,
        node_type: NodeType,
        document_id: str | None = None,
    ) -> list[DocumentNode]:
        return self._require_document_index().get_nodes_by_type(
            node_type=node_type,
            document_id=document_id,
        )

    def get_structured_page_nodes(
        self,
        document_id: str,
        page_number: int,
    ) -> list[DocumentNode]:
        return self._require_document_index().get_page_nodes(
            document_id=document_id,
            page_number=page_number,
        )

    def get_structured_content_nodes(
        self,
        document_id: str | None = None,
    ) -> list[DocumentNode]:
        return self._require_document_index().get_content_nodes(
            document_id=document_id,
        )

    def clear_document_nodes(self) -> None:
        self.document_nodes.clear()
        self.document_index = None

    def _require_document_index(
        self,
    ) -> HierarchicalDocumentIndex:
        if self.document_index is None:
            raise RuntimeError(
                "No hierarchical document index is loaded. "
                "Call ingest() first."
            )

        return self.document_index

    def _structured_metadata(
        self,
    ) -> dict[str, object]:
        if (
            not self.document_nodes
            or self.document_index is None
        ):
            return {
                "structured_document_loaded": False,
                "structured_node_count": 0,
                "hierarchical_index_loaded": False,
                "hierarchical_index_node_count": 0,
            }

        return {
            "structured_document_loaded": True,
            "structured_node_count": len(
                self.document_nodes
            ),
            "structured_document_ids": (
                self.document_index.document_ids
            ),
            "hierarchical_index_loaded": True,
            "hierarchical_index_node_count": (
                self.document_index.node_count
            ),
        }

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty."
            )

        return normalized_question
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from src.models import (
    AnswerResult,
    DocumentNode,
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

        # Structured nodes remain in memory.
        # The hierarchical index will consume them in Issue #4.
        self.document_nodes: list[DocumentNode] = []

    def ingest(
        self,
        document_path: Path | str,
    ) -> list[DocumentNode]:
        """
        Parse a document into the internal DocumentNode representation.

        The nodes are kept in memory and returned to the caller. No
        intermediate JSON file is required.

        Args:
            document_path: Path to the source PDF or supported document.

        Returns:
            Parsed DocumentNode objects.

        Raises:
            RuntimeError: If no document parser was configured.
            FileNotFoundError: If the input document does not exist.
            ValueError: If the input path is not a file.
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

        nodes = self.document_parser.parse(normalized_path)

        self.document_nodes = list(nodes)

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
        """
        Answer a question using the configured retriever and generator.

        When document_path is supplied, the document is first parsed with
        Docling and the resulting DocumentNode objects are kept in memory.
        The current baseline still retrieves pages using CLIP.
        """
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

    def get_nodes_by_type(
        self,
        node_type: str,
    ) -> list[DocumentNode]:
        """
        Return in-memory structured nodes matching a NodeType value.

        Example:
            pipeline.get_nodes_by_type("table")
        """
        normalized_type = node_type.strip().lower()

        return [
            node
            for node in self.document_nodes
            if node.node_type.value == normalized_type
        ]

    def clear_document_nodes(self) -> None:
        """Remove the currently ingested structured document."""
        self.document_nodes.clear()

    def _structured_metadata(self) -> dict[str, object]:
        if not self.document_nodes:
            return {
                "structured_document_loaded": False,
                "structured_node_count": 0,
            }

        document_ids = sorted(
            {
                node.document_id
                for node in self.document_nodes
            }
        )

        return {
            "structured_document_loaded": True,
            "structured_node_count": len(
                self.document_nodes
            ),
            "structured_document_ids": document_ids,
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
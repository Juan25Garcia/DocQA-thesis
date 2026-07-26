from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import DocumentNode, NodeType


class PageMetadataParser:
    """
    Converts the existing page metadata CSV into DocumentNode objects.

    Expected columns:
        doc_id
        page_num
        image_path
    """

    REQUIRED_COLUMNS = {
        "doc_id",
        "page_num",
        "image_path",
    }

    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path

    @property
    def name(self) -> str:
        return "page_metadata_parser"

    def parse_all(self) -> list[DocumentNode]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        metadata = pd.read_csv(self.metadata_path)

        missing_columns = self.REQUIRED_COLUMNS.difference(
            metadata.columns
        )

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))

            raise ValueError(
                "The metadata CSV is missing required columns: "
                f"{missing}"
            )

        nodes = [
            self._row_to_node(row)
            for _, row in metadata.iterrows()
        ]

        if not nodes:
            raise ValueError(
                f"No page records were found in {self.metadata_path}"
            )

        return nodes

    def _row_to_node(
        self,
        row: pd.Series,
    ) -> DocumentNode:
        document_id = str(row["doc_id"]).strip()
        page_number = int(row["page_num"])
        image_path = self._resolve_image_path(
            str(row["image_path"])
        )

        return DocumentNode(
            node_id=self.build_node_id(
                document_id=document_id,
                page_number=page_number,
            ),
            document_id=document_id,
            node_type=NodeType.PAGE,
            page_number=page_number,
            image_path=image_path,
            parent_id=document_id,
            metadata={
                "source": self.name,
                "doc_id": document_id,
                "page_num": page_number,
            },
        )

    def _resolve_image_path(
        self,
        image_path_value: str,
    ) -> Path:
        image_path = Path(image_path_value)

        if image_path.is_absolute():
            return image_path

        project_root = Path(__file__).resolve().parents[2]

        return project_root / image_path

    @staticmethod
    def build_node_id(
        document_id: str,
        page_number: int,
    ) -> str:
        return f"{document_id}:page:{page_number}"

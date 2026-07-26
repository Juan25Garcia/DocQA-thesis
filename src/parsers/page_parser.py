from __future__ import annotations

import csv
from pathlib import Path

from src.models import DocumentNode, NodeType


class PageMetadataParser:
    def __init__(self, metadata_csv: Path):
        self.metadata_csv = metadata_csv

    @property
    def name(self) -> str:
        return "page_metadata_parser"

    def parse_all(self) -> list[DocumentNode]:
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_csv}"
            )

        nodes: list[DocumentNode] = []

        with self.metadata_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError("Metadata CSV has no headers.")

            for row_number, row in enumerate(reader, start=2):
                node = self._row_to_node(row, row_number)
                nodes.append(node)

        return nodes

    def _row_to_node(
        self,
        row: dict[str, str],
        row_number: int,
    ) -> DocumentNode:
        document_id = self._get_first_value(
            row,
            "document_id",
            "document_name",
            "pdf_name",
            "document_path",
        )

        if not document_id:
            raise ValueError(
                f"Missing document identifier at row {row_number}."
            )

        document_id = Path(document_id).stem

        page_value = self._get_first_value(
            row,
            "page_number",
            "page",
            "page_index",
        )

        if not page_value:
            raise ValueError(
                f"Missing page number at row {row_number}."
            )

        page_number = int(page_value)

        image_value = self._get_first_value(
            row,
            "image_path",
            "image",
            "page_image",
        )

        text = self._get_first_value(
            row,
            "text",
            "ocr_text",
            "content",
        )

        return DocumentNode(
            node_id=f"{document_id}:page:{page_number}",
            document_id=document_id,
            node_type=NodeType.PAGE,
            page_number=page_number,
            text=text,
            image_path=Path(image_value) if image_value else None,
            parent_id=document_id,
            metadata={
                "source": self.name,
                "csv_row": row_number,
            },
        )

    @staticmethod
    def _get_first_value(
        row: dict[str, str],
        *keys: str,
    ) -> str:
        for key in keys:
            value = row.get(key, "")

            if value and value.strip():
                return value.strip()

        return ""

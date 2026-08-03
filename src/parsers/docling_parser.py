from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from src.models import BoundingBox, DocumentNode, NodeType


class DoclingParser:
    """Parse documents with Docling into the project's DocumentNode model."""

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def parse(self, input_path: Path) -> list[DocumentNode]:
        """
        Parse a document and convert it into hierarchical DocumentNode objects.

        The returned list is flat, but nodes are connected through parent_id
        and children_ids.

        Args:
            input_path: Path to the source document.

        Returns:
            Parsed nodes ordered by document, page and visual reading order.

        Raises:
            FileNotFoundError: If the input document does not exist.
            ValueError: If the input path is not a file.
            RuntimeError: If Docling cannot convert the document.
        """
        input_path = input_path.resolve()

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input document not found: {input_path}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"Input path is not a file: {input_path}"
            )

        try:
            conversion_result = self.converter.convert(input_path)
            raw_document = conversion_result.document.model_dump(
                mode="json"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Docling could not parse document: {input_path}"
            ) from exc

        return self._build_nodes(
            raw_document=raw_document,
            input_path=input_path,
        )

    def parse_to_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> Path:
        """Parse a document and write its internal node representation to JSON."""
        input_path = input_path.resolve()
        nodes = self.parse(input_path)

        output_path = output_path.resolve()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        document_node = next(
            (
                node
                for node in nodes
                if node.node_type == NodeType.DOCUMENT
            ),
            None,
        )

        document_id = (
            document_node.document_id
            if document_node is not None
            else input_path.stem
        )

        payload = {
            "document_id": document_id,
            "source_filename": input_path.name,
            "nodes": [
                node.to_dict()
                for node in nodes
            ],
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                payload,
                output_file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path

    def _build_nodes(
        self,
        raw_document: dict[str, Any],
        input_path: Path,
    ) -> list[DocumentNode]:
        document_id = raw_document.get(
            "name",
            input_path.stem,
        )

        document_node = self._build_document_node(
            raw_document=raw_document,
            input_path=input_path,
            document_id=document_id,
        )

        page_nodes = self._build_page_nodes(
            raw_document=raw_document,
            document_id=document_id,
            document_node_id=document_node.node_id,
        )

        content_nodes: list[DocumentNode] = []

        for text_index, text_item in enumerate(
            raw_document.get("texts", [])
        ):
            content_nodes.append(
                self._build_text_node(
                    document_id=document_id,
                    text_item=text_item,
                    text_index=text_index,
                )
            )

        for table_index, table_item in enumerate(
            raw_document.get("tables", [])
        ):
            content_nodes.append(
                self._build_table_node(
                    document_id=document_id,
                    table_item=table_item,
                    table_index=table_index,
                )
            )

        for picture_index, picture_item in enumerate(
            raw_document.get("pictures", [])
        ):
            content_nodes.append(
                self._build_picture_node(
                    document_id=document_id,
                    picture_item=picture_item,
                    picture_index=picture_index,
                )
            )

        nodes = [
            document_node,
            *page_nodes,
            *content_nodes,
        ]

        nodes.sort(key=self._node_sort_key)

        self._rebuild_children_in_reading_order(nodes)

        return nodes

    def _build_document_node(
        self,
        raw_document: dict[str, Any],
        input_path: Path,
        document_id: str,
    ) -> DocumentNode:
        origin = raw_document.get("origin", {})

        return DocumentNode(
            node_id=f"{document_id}:document",
            document_id=document_id,
            node_type=NodeType.DOCUMENT,
            text="",
            parent_id=None,
            metadata={
                "source_filename": origin.get(
                    "filename",
                    input_path.name,
                ),
                "mimetype": origin.get("mimetype"),
                "binary_hash": origin.get("binary_hash"),
                "schema_name": raw_document.get("schema_name"),
                "schema_version": raw_document.get("version"),
                "parser": "docling",
            },
        )

    def _build_page_nodes(
        self,
        raw_document: dict[str, Any],
        document_id: str,
        document_node_id: str,
    ) -> list[DocumentNode]:
        page_nodes: list[DocumentNode] = []

        for page_key, page_data in raw_document.get(
            "pages",
            {},
        ).items():
            page_number = int(
                page_data.get(
                    "page_no",
                    page_key,
                )
            )

            page_size = page_data.get("size", {})

            page_nodes.append(
                DocumentNode(
                    node_id=f"{document_id}:page:{page_number}",
                    document_id=document_id,
                    node_type=NodeType.PAGE,
                    page_number=page_number,
                    parent_id=document_node_id,
                    metadata={
                        "width": page_size.get("width"),
                        "height": page_size.get("height"),
                    },
                )
            )

        page_nodes.sort(
            key=lambda node: node.page_number or 0
        )

        return page_nodes

    def _build_text_node(
        self,
        document_id: str,
        text_item: dict[str, Any],
        text_index: int,
    ) -> DocumentNode:
        provenance = self._first_provenance(text_item)
        page_number = self._page_number_from_provenance(
            provenance
        )

        node_type = self._map_text_label(
            text_item.get(
                "label",
                "text",
            )
        )

        return DocumentNode(
            node_id=(
                f"{document_id}:"
                f"{node_type.value}:"
                f"{text_index}"
            ),
            document_id=document_id,
            node_type=node_type,
            page_number=page_number,
            text=text_item.get(
                "text",
                "",
            ).strip(),
            bbox=self._parse_bbox(
                provenance.get("bbox")
            ),
            parent_id=self._content_parent_id(
                document_id=document_id,
                page_number=page_number,
            ),
            metadata={
                "source_ref": text_item.get("self_ref"),
                "label": text_item.get("label"),
                "content_layer": text_item.get(
                    "content_layer"
                ),
                "level": text_item.get("level"),
                "original_text": text_item.get("orig"),
            },
        )

    def _build_table_node(
        self,
        document_id: str,
        table_item: dict[str, Any],
        table_index: int,
    ) -> DocumentNode:
        provenance = self._first_provenance(table_item)
        page_number = self._page_number_from_provenance(
            provenance
        )

        table_data = table_item.get("data", {})

        num_rows = int(
            table_data.get(
                "num_rows",
                0,
            )
        )
        num_cols = int(
            table_data.get(
                "num_cols",
                0,
            )
        )

        cells = table_data.get(
            "table_cells",
            [],
        )

        grid = self._build_table_grid(
            cells=cells,
            rows=num_rows,
            columns=num_cols,
        )

        return DocumentNode(
            node_id=f"{document_id}:table:{table_index}",
            document_id=document_id,
            node_type=NodeType.TABLE,
            page_number=page_number,
            text=self._table_to_text(grid),
            bbox=self._parse_bbox(
                provenance.get("bbox")
            ),
            parent_id=self._content_parent_id(
                document_id=document_id,
                page_number=page_number,
            ),
            metadata={
                "source_ref": table_item.get("self_ref"),
                "label": table_item.get("label"),
                "content_layer": table_item.get(
                    "content_layer"
                ),
                "num_rows": num_rows,
                "num_cols": num_cols,
                "grid": grid,
                "cells": [
                    self._serialize_table_cell(cell)
                    for cell in cells
                ],
            },
        )

    def _build_picture_node(
        self,
        document_id: str,
        picture_item: dict[str, Any],
        picture_index: int,
    ) -> DocumentNode:
        provenance = self._first_provenance(picture_item)
        page_number = self._page_number_from_provenance(
            provenance
        )

        return DocumentNode(
            node_id=f"{document_id}:figure:{picture_index}",
            document_id=document_id,
            node_type=NodeType.FIGURE,
            page_number=page_number,
            text="",
            bbox=self._parse_bbox(
                provenance.get("bbox")
            ),
            parent_id=self._content_parent_id(
                document_id=document_id,
                page_number=page_number,
            ),
            metadata={
                "source_ref": picture_item.get("self_ref"),
                "label": picture_item.get("label"),
                "content_layer": picture_item.get(
                    "content_layer"
                ),
            },
        )

    @staticmethod
    def _content_parent_id(
        document_id: str,
        page_number: int | None,
    ) -> str:
        if page_number is None:
            return f"{document_id}:document"

        return f"{document_id}:page:{page_number}"

    @staticmethod
    def _map_text_label(
        label: str,
    ) -> NodeType:
        normalized_label = label.lower()

        mapping = {
            "title": NodeType.TITLE,
            "section_header": NodeType.SECTION,
            "paragraph": NodeType.PARAGRAPH,
            "text": NodeType.PARAGRAPH,
            "list_item": NodeType.PARAGRAPH,
            "caption": NodeType.PARAGRAPH,
            "footnote": NodeType.FOOTER,
            "page_header": NodeType.HEADER,
            "page_footer": NodeType.FOOTER,
            "key_value": NodeType.KEY_VALUE,
        }

        return mapping.get(
            normalized_label,
            NodeType.UNKNOWN,
        )

    @staticmethod
    def _first_provenance(
        item: dict[str, Any],
    ) -> dict[str, Any]:
        provenance = item.get(
            "prov",
            [],
        )

        if not provenance:
            return {}

        return provenance[0]

    @staticmethod
    def _page_number_from_provenance(
        provenance: dict[str, Any],
    ) -> int | None:
        page_number = provenance.get("page_no")

        if page_number is None:
            return None

        return int(page_number)

    @staticmethod
    def _parse_bbox(
        bbox: dict[str, Any] | None,
    ) -> BoundingBox | None:
        if not bbox:
            return None

        return BoundingBox(
            x0=float(bbox["l"]),
            y0=float(bbox["b"]),
            x1=float(bbox["r"]),
            y1=float(bbox["t"]),
        )

    @staticmethod
    def _build_table_grid(
        cells: list[dict[str, Any]],
        rows: int,
        columns: int,
    ) -> list[list[str]]:
        grid = [
            [
                ""
                for _ in range(columns)
            ]
            for _ in range(rows)
        ]

        for cell in cells:
            row_index = int(
                cell.get(
                    "start_row_offset_idx",
                    0,
                )
            )
            column_index = int(
                cell.get(
                    "start_col_offset_idx",
                    0,
                )
            )

            if (
                0 <= row_index < rows
                and 0 <= column_index < columns
            ):
                grid[row_index][column_index] = (
                    cell.get(
                        "text",
                        "",
                    ).strip()
                )

        return grid

    @staticmethod
    def _table_to_text(
        grid: list[list[str]],
    ) -> str:
        return "\n".join(
            " | ".join(row)
            for row in grid
        )

    def _serialize_table_cell(
        self,
        cell: dict[str, Any],
    ) -> dict[str, Any]:
        bbox = self._parse_bbox(
            cell.get("bbox")
        )

        return {
            "text": cell.get(
                "text",
                "",
            ).strip(),
            "row_start": int(
                cell.get(
                    "start_row_offset_idx",
                    0,
                )
            ),
            "row_end": int(
                cell.get(
                    "end_row_offset_idx",
                    0,
                )
            ),
            "column_start": int(
                cell.get(
                    "start_col_offset_idx",
                    0,
                )
            ),
            "column_end": int(
                cell.get(
                    "end_col_offset_idx",
                    0,
                )
            ),
            "row_span": int(
                cell.get(
                    "row_span",
                    1,
                )
            ),
            "column_span": int(
                cell.get(
                    "col_span",
                    1,
                )
            ),
            "column_header": bool(
                cell.get(
                    "column_header",
                    False,
                )
            ),
            "row_header": bool(
                cell.get(
                    "row_header",
                    False,
                )
            ),
            "bbox": (
                bbox.to_list()
                if bbox is not None
                else None
            ),
        }

    @staticmethod
    def _node_sort_key(
        node: DocumentNode,
    ) -> tuple[int, int, float, float, str]:
        type_order = {
            NodeType.DOCUMENT: 0,
            NodeType.PAGE: 1,
        }

        base_order = type_order.get(
            node.node_type,
            2,
        )

        page_number = node.page_number or 0

        if node.bbox is None:
            return (
                base_order,
                page_number,
                float("inf"),
                float("inf"),
                node.node_id,
            )

        return (
            base_order,
            page_number,
            -node.bbox.y1,
            node.bbox.x0,
            node.node_id,
        )

    @staticmethod
    def _rebuild_children_in_reading_order(
        nodes: list[DocumentNode],
    ) -> None:
        """
        Rebuild every children_ids list following the current node order.

        This ensures page children follow visual reading order instead of
        being grouped by Docling element type, such as all texts first and
        all tables afterward.
        """
        nodes_by_id = {
            node.node_id: node
            for node in nodes
        }

        for node in nodes:
            node.children_ids.clear()

        for node in nodes:
            if node.parent_id is None:
                continue

            parent = nodes_by_id.get(node.parent_id)

            if parent is not None:
                parent.children_ids.append(node.node_id)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a document with Docling and export "
            "DocumentNode objects as JSON."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source document.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON path. Defaults to "
            "data/structured/<document-name>.json."
        ),
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    output_path = args.output or (
        Path("data")
        / "structured"
        / f"{args.input.stem}.json"
    )

    parser = DoclingParser()

    generated_path = parser.parse_to_file(
        input_path=args.input,
        output_path=output_path,
    )

    print(
        f"Structured document written to: "
        f"{generated_path}"
    )


if __name__ == "__main__":
    main()
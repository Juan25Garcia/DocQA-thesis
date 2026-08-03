from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models import NodeType
from src.parsers.docling_parser import DoclingParser


def test_missing_file_raises() -> None:
    parser = DoclingParser()

    with pytest.raises(FileNotFoundError):
        parser.parse(Path("missing.pdf"))


def test_text_label_mapping() -> None:
    parser = DoclingParser()

    assert parser._map_text_label("section_header") == NodeType.SECTION
    assert parser._map_text_label("title") == NodeType.TITLE
    assert parser._map_text_label("text") == NodeType.PARAGRAPH
    assert parser._map_text_label("paragraph") == NodeType.PARAGRAPH
    assert parser._map_text_label("page_header") == NodeType.HEADER
    assert parser._map_text_label("page_footer") == NodeType.FOOTER
    assert parser._map_text_label("unknown") == NodeType.UNKNOWN


def test_build_table_grid() -> None:
    parser = DoclingParser()

    cells = [
        {
            "text": "A",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
        },
        {
            "text": "B",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 1,
        },
        {
            "text": "C",
            "start_row_offset_idx": 1,
            "start_col_offset_idx": 0,
        },
        {
            "text": "D",
            "start_row_offset_idx": 1,
            "start_col_offset_idx": 1,
        },
    ]

    grid = parser._build_table_grid(
        cells=cells,
        rows=2,
        columns=2,
    )

    assert grid == [
        ["A", "B"],
        ["C", "D"],
    ]


def test_table_to_text() -> None:
    parser = DoclingParser()

    table = [
        ["A", "B"],
        ["C", "D"],
    ]

    text = parser._table_to_text(table)

    assert text == "A | B\nC | D"


def test_rebuild_children_order() -> None:
    from src.models import DocumentNode

    parser = DoclingParser()

    document = DocumentNode(
        node_id="doc",
        document_id="doc",
        node_type=NodeType.DOCUMENT,
    )

    page = DocumentNode(
        node_id="page",
        document_id="doc",
        node_type=NodeType.PAGE,
        page_number=1,
        parent_id="doc",
    )

    section = DocumentNode(
        node_id="section",
        document_id="doc",
        node_type=NodeType.SECTION,
        page_number=1,
        parent_id="page",
    )

    table = DocumentNode(
        node_id="table",
        document_id="doc",
        node_type=NodeType.TABLE,
        page_number=1,
        parent_id="page",
    )

    paragraph = DocumentNode(
        node_id="paragraph",
        document_id="doc",
        node_type=NodeType.PARAGRAPH,
        page_number=1,
        parent_id="page",
    )

    nodes = [
        document,
        page,
        section,
        table,
        paragraph,
    ]

    parser._rebuild_children_in_reading_order(nodes)

    assert document.children_ids == ["page"]

    assert page.children_ids == [
        "section",
        "table",
        "paragraph",
    ]


def test_parse_to_file(tmp_path: Path) -> None:
    parser = DoclingParser()

    parser.parse = MagicMock(
        return_value=[]
    )

    input_pdf = tmp_path / "invoice.pdf"
    input_pdf.write_bytes(b"dummy")

    output = tmp_path / "invoice.json"

    parser.parse_to_file(
        input_pdf,
        output,
    )

    assert output.exists()
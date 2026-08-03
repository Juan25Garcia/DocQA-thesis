from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import demo
from src.models import (
    DocumentNode,
    NodeType,
    RetrievedEvidence,
)


def test_block_mode_requires_document() -> None:
    args = MagicMock()
    args.document = None
    args.retrieval_only = True

    with pytest.raises(
        ValueError,
        match="--document is required",
    ):
        demo.run_block_mode(args)


def test_block_mode_requires_retrieval_only(
    tmp_path: Path,
) -> None:
    args = MagicMock()
    args.document = tmp_path / "document.pdf"
    args.retrieval_only = False

    with pytest.raises(
        ValueError,
        match="supports only --retrieval-only",
    ):
        demo.run_block_mode(args)


def test_print_block_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    node = DocumentNode(
        node_id="invoice:table:0",
        document_id="invoice",
        node_type=NodeType.TABLE,
        page_number=1,
        text="Total Amount | $1,245.50",
    )

    evidence = RetrievedEvidence(
        node=node,
        score=0.95,
        rank=1,
        retriever_name="semantic_block_retriever",
    )

    demo.print_block_evidence([evidence])

    output = capsys.readouterr().out

    assert "Top retrieved blocks" in output
    assert "invoice:table:0" in output
    assert "Total Amount | $1,245.50" in output
from pathlib import Path

from src.models import DocumentNode, NodeType
from src.retrievers.clip_retriever import (
    ClipRetrieverAdapter,
)


def fake_retrieve_pages(
    query: str,
    top_k: int,
) -> list[dict]:
    assert query == "What is the total amount?"
    assert top_k == 3

    return [
        {
            "doc_id": "invoice",
            "page_num": 2,
            "image_path": (
                "data/pages/invoice/page_002.png"
            ),
            "rank": 1,
            "score": 0.93,
        }
    ]


def test_clip_result_is_converted_to_evidence() -> None:
    node = DocumentNode(
        node_id="invoice:page:2",
        document_id="invoice",
        node_type=NodeType.PAGE,
        page_number=2,
        image_path=Path(
            "data/pages/invoice/page_002.png"
        ),
    )

    retriever = ClipRetrieverAdapter(
        nodes=[node],
        search_function=fake_retrieve_pages,
    )

    evidence = retriever.retrieve(
        query="What is the total amount?",
        top_k=3,
    )

    assert len(evidence) == 1
    assert evidence[0].node.node_id == "invoice:page:2"
    assert evidence[0].node.document_id == "invoice"
    assert evidence[0].node.page_number == 2
    assert evidence[0].rank == 1
    assert evidence[0].score == 0.93
    assert (
        evidence[0].retriever_name
        == "clip_page_retriever"
    )

from pathlib import Path

from src.models import DocumentNode, NodeType
from src.pipeline import DocumentQAPipeline
from src.retrievers.clip_retriever import ClipRetrieverAdapter


def fake_clip_search(
    query: str,
    top_k: int,
) -> list[dict]:
    return [
        {
            "node_id": "invoice:page:2",
            "score": 0.94,
        }
    ][:top_k]


def fake_answer(
    question,
    evidence,
) -> str:
    assert question == "What is the invoice total?"
    assert evidence[0].node.page_number == 2

    return "The invoice total is 1,250.00."


def test_existing_pipeline_can_be_wrapped() -> None:
    nodes = [
        DocumentNode(
            node_id="invoice:page:2",
            document_id="invoice",
            node_type=NodeType.PAGE,
            page_number=2,
            text="Invoice total: 1,250.00",
            image_path=Path(
                "data/pages/invoice/page_002.png"
            ),
        )
    ]

    retriever = ClipRetrieverAdapter(
        nodes=nodes,
        search_function=fake_clip_search,
    )

    pipeline = DocumentQAPipeline(
        retriever=retriever,
        answer_function=fake_answer,
    )

    result = pipeline.ask(
        "What is the invoice total?",
        top_k=3,
    )

    assert result.answer == "The invoice total is 1,250.00."
    assert len(result.evidence) == 1
    assert result.evidence[0].score == 0.94
    assert result.evidence[0].node.page_number == 2

from __future__ import annotations

from pathlib import Path

from src.parsers.docling_parser import DoclingParser


PDF_DIR = Path("data/pdfs")


def main() -> None:
    parser = DoclingParser()

    pdf_files = sorted(
        path
        for path in PDF_DIR.glob("*.pdf")
        if path.is_file()
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:
        print("\n" + "=" * 80)
        print(f"DOCUMENT: {pdf_path.name}")
        print("=" * 80)

        nodes = parser.parse(pdf_path)

        for node in nodes:
            if not node.text.strip():
                continue

            print()
            print(f"node_id: {node.node_id}")
            print(f"type: {node.node_type.value}")
            print(f"page: {node.page_number}")
            print("text:")

            for line in node.text.splitlines():
                print(f"  {line}")


if __name__ == "__main__":
    main()

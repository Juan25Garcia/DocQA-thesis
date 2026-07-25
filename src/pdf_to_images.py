from pdf2image import convert_from_path
from tqdm import tqdm

from src.config import PDF_DIR, PAGE_DIR, ensure_dirs


def convert_pdfs_to_images(dpi: int = 150, overwrite: bool = False) -> None:
    """Converts each PDF in data/pdfs into page images under data/pages/<doc_id>/."""
    ensure_dirs()

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}. Add PDFs and run again.")
        return

    for pdf_path in tqdm(pdf_paths, desc="Converting PDFs"):
        doc_id = pdf_path.stem
        output_doc_dir = PAGE_DIR / doc_id
        output_doc_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and any(output_doc_dir.glob("page_*.png")):
            print(f"Skipping {pdf_path.name}: pages already exist.")
            continue

        images = convert_from_path(str(pdf_path), dpi=dpi)

        for i, image in enumerate(images, start=1):
            page_path = output_doc_dir / f"page_{i:03d}.png"
            image.save(page_path, "PNG")

        print(f"Saved {len(images)} pages for {pdf_path.name}")


if __name__ == "__main__":
    convert_pdfs_to_images()

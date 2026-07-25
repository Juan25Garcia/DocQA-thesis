import pandas as pd

from src.config import PAGE_DIR, PAGES_METADATA_PATH, ensure_dirs


def create_page_metadata() -> pd.DataFrame:
    """Creates a metadata CSV with one row per rendered page image."""
    ensure_dirs()

    rows = []

    for doc_dir in sorted(PAGE_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue

        for page_path in sorted(doc_dir.glob("page_*.png")):
            try:
                page_num = int(page_path.stem.split("_")[-1])
            except ValueError:
                continue

            rows.append(
                {
                    "doc_id": doc_dir.name,
                    "page_num": page_num,
                    "image_path": str(page_path),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"No page images found in {PAGE_DIR}. Run pdf_to_images.py first.")
        return df

    df.to_csv(PAGES_METADATA_PATH, index=False)
    print(f"Saved {len(df)} pages to {PAGES_METADATA_PATH}")

    return df


if __name__ == "__main__":
    create_page_metadata()

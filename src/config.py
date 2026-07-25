from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PAGE_DIR = DATA_DIR / "pages"
INDEX_DIR = DATA_DIR / "index"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RETRIEVAL_OUTPUT_DIR = OUTPUT_DIR / "retrieval_results"
ANSWERS_OUTPUT_DIR = OUTPUT_DIR / "answers"

PAGES_METADATA_PATH = INDEX_DIR / "pages_metadata.csv"
CLIP_FAISS_INDEX_PATH = INDEX_DIR / "clip_pages.faiss"
CLIP_EMBEDDINGS_PATH = INDEX_DIR / "clip_embeddings.npy"

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")


def ensure_dirs() -> None:
    for path in [
        PDF_DIR,
        PAGE_DIR,
        INDEX_DIR,
        RETRIEVAL_OUTPUT_DIR,
        ANSWERS_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

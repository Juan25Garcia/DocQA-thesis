from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGES_METADATA_PATH = PROJECT_ROOT / "data" / "index" / "pages_metadata.csv"
CLIP_FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "clip_pages.faiss"
CLIP_METADATA_PATH = PROJECT_ROOT / "data" / "index" / "clip_pages_metadata.csv"

CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


def embed_images(image_paths: list[str], batch_size: int = 8) -> np.ndarray:
    model = SentenceTransformer(CLIP_MODEL_NAME)

    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding pages"):
        batch_paths = image_paths[i : i + batch_size]

        images = [
            Image.open(path).convert("RGB")
            for path in batch_paths
        ]

        embeddings = model.encode(
            images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_embeddings.append(embeddings.astype("float32"))

    return np.ascontiguousarray(np.vstack(all_embeddings).astype("float32"))


def build_clip_index(batch_size: int = 8) -> None:
    if not PAGES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {PAGES_METADATA_PATH}. Run create_metadata.py first."
        )

    df = pd.read_csv(PAGES_METADATA_PATH)

    if df.empty:
        raise ValueError("pages_metadata.csv is empty.")

    image_paths = []

    for path in df["image_path"].tolist():
        path_obj = Path(path)

        if path_obj.is_absolute():
            image_paths.append(str(path_obj))
        else:
            image_paths.append(str((PROJECT_ROOT / path_obj).resolve()))

    embeddings = embed_images(image_paths, batch_size=batch_size)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    CLIP_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(CLIP_FAISS_INDEX_PATH))
    df.to_csv(CLIP_METADATA_PATH, index=False)

    print(f"Saved FAISS index to: {CLIP_FAISS_INDEX_PATH}")
    print(f"Saved CLIP metadata to: {CLIP_METADATA_PATH}")
    print(f"Indexed {len(df)} pages.")
    print(f"Embedding dimension: {dim}")


if __name__ == "__main__":
    build_clip_index()
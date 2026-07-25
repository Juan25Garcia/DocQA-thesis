from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLIP_FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "clip_pages.faiss"
CLIP_METADATA_PATH = PROJECT_ROOT / "data" / "index" / "clip_pages_metadata.csv"

CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


def embed_query(query: str) -> np.ndarray:
    model = SentenceTransformer(CLIP_MODEL_NAME)

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.ascontiguousarray(embedding.astype("float32"))


def retrieve_pages(query: str, top_k: int = 3) -> list[dict]:
    if not CLIP_FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found: {CLIP_FAISS_INDEX_PATH}. Run build_clip_index.py first."
        )

    if not CLIP_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {CLIP_METADATA_PATH}. Run build_clip_index.py first."
        )

    index = faiss.read_index(str(CLIP_FAISS_INDEX_PATH))
    df = pd.read_csv(CLIP_METADATA_PATH)

    query_embedding = embed_query(query)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue

        row = df.iloc[int(idx)].to_dict()
        row["rank"] = rank
        row["score"] = float(score)
        results.append(row)

    return results


if __name__ == "__main__":
    results = retrieve_pages("What is the total amount?", top_k=3)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['doc_id']}")
        print(f"Page: {result['page_num']}")
        print(f"Image: {result['image_path']}")
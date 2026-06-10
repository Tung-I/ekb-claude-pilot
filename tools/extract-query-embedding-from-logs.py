"""
Extract and save query embeddings for plan-caching-study traces.

Usage:
    python tools/extract-query-embedding-from-logs.py [--split {knowledge-base,test,both}]

For each task folder under traces/plan-caching-study/<split>/, reads
normalized_trace.json, encodes the query_text with all-MiniLM-L6-v2,
and saves the embedding as query_embedding.npy alongside the trace.

Falls back to TF-IDF + SVD if sentence-transformers is unavailable.
"""

import argparse
import json
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACES_ROOT = REPO_ROOT / "traces" / "plan-caching-study"

EMBEDDING_BACKEND: str = ""


def make_embeddings(texts: list[str]) -> np.ndarray:
    global EMBEDDING_BACKEND
    texts = [str(t) for t in texts]

    try:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        X = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        EMBEDDING_BACKEND = model_name
        return np.asarray(X, dtype=np.float32)
    except Exception as e:
        print("SentenceTransformer unavailable; falling back to TF-IDF + SVD")
        print("Reason:", repr(e))
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X_tfidf = vectorizer.fit_transform(texts)
        n_components = min(128, max(2, X_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X = svd.fit_transform(X_tfidf)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        EMBEDDING_BACKEND = "tfidf+svd"
        return np.asarray(X, dtype=np.float32)


def collect_tasks(split: str) -> list[pathlib.Path]:
    split_dir = TRACES_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and (p / "normalized_trace.json").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Extract query embeddings from trace logs.")
    parser.add_argument(
        "--split",
        choices=["knowledge-base", "test", "both"],
        default="both",
        help="Which split to process (default: both)",
    )
    args = parser.parse_args()

    splits = ["knowledge-base", "test"] if args.split == "both" else [args.split]

    for split in splits:
        task_dirs = collect_tasks(split)
        print(f"\n[{split}] Found {len(task_dirs)} tasks")

        queries: list[str] = []
        for td in task_dirs:
            trace = json.loads((td / "normalized_trace.json").read_text())
            queries.append(trace["query_text"])

        embeddings = make_embeddings(queries)
        print(f"[{split}] Backend: {EMBEDDING_BACKEND}  |  Shape: {embeddings.shape}")

        for td, emb in zip(task_dirs, embeddings):
            out_path = td / "query_embedding.npy"
            np.save(out_path, emb)

        print(f"[{split}] Saved embeddings to each task subfolder")


if __name__ == "__main__":
    main()

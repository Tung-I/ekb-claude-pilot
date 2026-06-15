"""
Extract and save query embeddings for gaia_lv1_x4 / gaia_lv2_x3 traces.

For each task folder under traces/claude_native/<run>/, reads
normalized_trace.json, encodes query_text, and saves the embedding as
query_embedding.npy under the mirrored path:

    traces/claude_native/<run>/<task_id>/normalized_trace.json
    →  embeddings/claude_native/<run>/<task_id>/query_embedding.npy

Usage:
    python tools/extract-query-embedding.py
    python tools/extract-query-embedding.py --run gaia_lv1_x4 gaia_lv2_x3
    python tools/extract-query-embedding.py --run gaia_lv1_x4 --model sentence-transformers/all-mpnet-base-v2
    python tools/extract-query-embedding.py --overwrite
"""

import argparse
import json
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACES_ROOT = REPO_ROOT / "traces" / "claude_native"
EMBEDDINGS_ROOT = REPO_ROOT / "embeddings" / "claude_native"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_BACKEND: str = ""


def make_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    global EMBEDDING_BACKEND
    texts = [str(t) for t in texts]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        X = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        EMBEDDING_BACKEND = model_name
        return np.asarray(X, dtype=np.float32)
    except Exception as e:
        print(f"SentenceTransformer unavailable (reason: {e!r}); falling back to TF-IDF + SVD")
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


def collect_task_dirs(run_name: str) -> list[pathlib.Path]:
    run_dir = TRACES_ROOT / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and (p / "normalized_trace.json").exists()
    )


def embedding_output_path(task_dir: pathlib.Path) -> pathlib.Path:
    """Mirror traces/claude_native/<run>/<task_id> → embeddings/claude_native/<run>/<task_id>."""
    rel = task_dir.relative_to(TRACES_ROOT)
    return EMBEDDINGS_ROOT / rel / "query_embedding.npy"


def main():
    parser = argparse.ArgumentParser(description="Extract query embeddings from trace logs.")
    parser.add_argument(
        "--run",
        nargs="+",
        metavar="RUN_NAME",
        default=None,
        help=(
            "One or more run names under traces/claude_native/ to process. "
            "Defaults to all runs found in that directory."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode and overwrite existing query_embedding.npy files.",
    )
    args = parser.parse_args()

    if args.run is not None:
        run_names = args.run
    else:
        if not TRACES_ROOT.exists():
            raise FileNotFoundError(f"Traces root not found: {TRACES_ROOT}")
        run_names = sorted(p.name for p in TRACES_ROOT.iterdir() if p.is_dir())
        if not run_names:
            raise RuntimeError(f"No run directories found under {TRACES_ROOT}")

    print(f"Runs to process : {run_names}")
    print(f"Embedding model : {args.model}")
    print(f"Output root     : {EMBEDDINGS_ROOT}")
    print()

    for run_name in run_names:
        task_dirs = collect_task_dirs(run_name)
        print(f"[{run_name}] {len(task_dirs)} tasks with normalized_trace.json")

        # Filter out already-done tasks unless --overwrite
        if not args.overwrite:
            pending = [td for td in task_dirs if not embedding_output_path(td).exists()]
            skipped = len(task_dirs) - len(pending)
            if skipped:
                print(f"[{run_name}] Skipping {skipped} already-embedded tasks (use --overwrite to redo)")
            task_dirs = pending

        if not task_dirs:
            print(f"[{run_name}] Nothing to do.\n")
            continue

        # Read query texts
        queries: list[str] = []
        for td in task_dirs:
            trace = json.loads((td / "normalized_trace.json").read_text())
            queries.append(trace["query_text"])

        # Encode
        embeddings = make_embeddings(queries, args.model)
        print(f"[{run_name}] Backend: {EMBEDDING_BACKEND}  |  Shape: {embeddings.shape}")

        # Save to mirrored output paths
        saved = 0
        for td, emb in zip(task_dirs, embeddings):
            out_path = embedding_output_path(td)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, emb)
            saved += 1

        print(f"[{run_name}] Saved {saved} embeddings under {EMBEDDINGS_ROOT / run_name}\n")


if __name__ == "__main__":
    main()

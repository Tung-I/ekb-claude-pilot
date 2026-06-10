"""
For each test task, find its 3 nearest neighbors (by query embedding cosine
similarity) among completed knowledge-base tasks and save metadata to
traces/plan-caching-study/test/<task_id>/3nn_meta.json.

Knowledge-base tasks with final_answer_pred=null are excluded (incomplete
traces). Tasks with success=false but a valid final_answer_pred are kept.

Usage:
    python tools/find-3nn-for-test-tasks.py
"""

import json
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
KB_DIR    = REPO_ROOT / "traces" / "plan-caching-study" / "knowledge-base"
TEST_DIR  = REPO_ROOT / "traces" / "plan-caching-study" / "test"


def load_split(split_dir: pathlib.Path, require_complete: bool = False):
    """
    Returns:
        task_ids  : list[str]
        embeddings: np.ndarray  shape (N, D)
        metas     : list[dict]  keys: task_id, total_tool_calls,
                                      total_latency_ms, total_tokens
    """
    task_ids, embeddings, metas = [], [], []
    for td in sorted(split_dir.iterdir()):
        if not td.is_dir():
            continue
        trace_path = td / "normalized_trace.json"
        emb_path   = td / "query_embedding.npy"
        if not trace_path.exists() or not emb_path.exists():
            continue

        trace = json.loads(trace_path.read_text())

        if require_complete and trace.get("final_answer_pred") is None:
            continue

        task_ids.append(trace["query_id"])
        embeddings.append(np.load(emb_path))
        metas.append({
            "task_id":          trace["query_id"],
            "total_tool_calls": trace.get("total_tool_calls"),
            "total_latency_ms": trace.get("total_latency_ms"),
            "total_tokens":     trace.get("total_tokens"),
        })

    return task_ids, np.stack(embeddings, axis=0), metas


def main():
    print("Loading knowledge-base embeddings (filtering incomplete traces)…")
    kb_ids, kb_embs, kb_metas = load_split(KB_DIR, require_complete=True)
    print(f"  {len(kb_ids)} valid knowledge-base tasks loaded")

    print("Loading test embeddings…")
    test_ids, test_embs, _ = load_split(TEST_DIR, require_complete=False)
    print(f"  {len(test_ids)} test tasks loaded")

    # Embeddings are already L2-normalised, so dot product = cosine similarity.
    # Shape: (n_test, n_kb)
    sim_matrix = test_embs @ kb_embs.T

    saved = 0
    for i, (test_id, sims) in enumerate(zip(test_ids, sim_matrix)):
        top3_idx = np.argsort(sims)[::-1][:3]
        neighbors = []
        for rank, kb_idx in enumerate(top3_idx):
            entry = dict(kb_metas[kb_idx])
            entry["rank"]       = rank + 1
            entry["similarity"] = float(sims[kb_idx])
            neighbors.append(entry)

        out = {
            "query_id":  test_id,
            "neighbors": neighbors,
        }

        out_path = TEST_DIR / test_id / "3nn_meta.json"
        out_path.write_text(json.dumps(out, indent=2))
        saved += 1

    print(f"Saved 3nn_meta.json for {saved} test tasks.")


if __name__ == "__main__":
    main()

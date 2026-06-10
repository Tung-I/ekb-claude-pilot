#!/usr/bin/env python3
"""
Build data/gaia_test/test.jsonl from traces/plan-caching-study/test/.

For each task directory under traces/plan-caching-study/test/ the script reads
normalized_trace.json and cross-references the original benchmark record from
data/gaia_paraphrases/gaia_level1_expanded.jsonl to recover file_name,
file_path, and metadata fields.  Tasks not found in that JSONL (e.g. tasks
added to the test split after the paraphrase expansion was generated) are
reconstructed from the trace alone.

Output: data/gaia_test/test.jsonl — one JSON object per line, schema matching
gaia_level1_expanded.jsonl so run_claude_task_w_plan_reuse.py can consume it
directly.

Usage:
    python tools/filter_test_set.py
    python tools/filter_test_set.py --traces-dir traces/plan-caching-study/test \
        --source-jsonl data/gaia_paraphrases/gaia_level1_expanded.jsonl \
        --output data/gaia_test/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{lineno}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row_from_original(orig: dict) -> dict:
    """Return a clean JSONL row from a gaia_level1_expanded.jsonl record."""
    return {
        "task_id":        orig["task_id"],
        "benchmark":      orig.get("benchmark", "gaia"),
        "split":          orig.get("split", "validation"),
        "level":          orig.get("level", 1),
        "question":       orig["question"],
        "final_answer":   orig.get("final_answer", ""),
        "file_name":      orig.get("file_name", ""),
        "file_path":      orig.get("file_path"),
        "metadata":       orig.get("metadata", {}),
        "source_task_id": orig.get("source_task_id", orig["task_id"]),
        "variant":        orig.get("variant", "original"),
        "is_paraphrase":  orig.get("is_paraphrase", False),
        "paraphrase_index": orig.get("paraphrase_index", 0),
    }


def build_row_from_trace(trace: dict) -> dict:
    """Fallback: reconstruct a JSONL row from a normalized_trace.json."""
    return {
        "task_id":        trace["query_id"],
        "benchmark":      trace.get("benchmark", "gaia"),
        "split":          trace.get("split", "validation"),
        "level":          trace.get("level", 1),
        "question":       trace["query_text"],
        "final_answer":   trace.get("ground_truth_answer", ""),
        "file_name":      "",
        "file_path":      None,
        "metadata":       {},
        "source_task_id": trace["query_id"],
        "variant":        "original",
        "is_paraphrase":  False,
        "paraphrase_index": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build test.jsonl from plan-caching-study test traces.")
    parser.add_argument(
        "--traces-dir",
        default=str(REPO_ROOT / "traces" / "plan-caching-study" / "test"),
        help="Directory containing per-task trace subdirectories.",
    )
    parser.add_argument(
        "--source-jsonl",
        default=str(REPO_ROOT / "data" / "gaia_paraphrases" / "gaia_level1_expanded.jsonl"),
        help="Original expanded JSONL to cross-reference for full task metadata.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "gaia_test" / "test.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    traces_dir   = Path(args.traces_dir).expanduser().resolve()
    source_path  = Path(args.source_jsonl).expanduser().resolve()
    output_path  = Path(args.output).expanduser().resolve()

    # Load original JSONL index (non-paraphrase rows only, keyed by task_id)
    orig_index: dict[str, dict] = {}
    if source_path.exists():
        for row in load_jsonl(source_path):
            if not row.get("is_paraphrase", False):
                orig_index[row["task_id"]] = row
        print(f"Loaded {len(orig_index)} original records from {source_path}")
    else:
        print(f"Warning: source JSONL not found at {source_path}; will reconstruct all rows from traces.")

    # Walk test task directories
    task_dirs = sorted(p for p in traces_dir.iterdir() if p.is_dir())
    print(f"Found {len(task_dirs)} task directories in {traces_dir}")

    rows: list[dict] = []
    n_from_orig  = 0
    n_from_trace = 0
    for td in task_dirs:
        trace_path = td / "normalized_trace.json"
        if not trace_path.exists():
            print(f"  skip {td.name}: no normalized_trace.json")
            continue

        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        task_id = trace["query_id"]

        if task_id in orig_index:
            row = build_row_from_original(orig_index[task_id])
            n_from_orig += 1
        else:
            print(f"  {task_id}: not in source JSONL — reconstructing from trace")
            row = build_row_from_trace(trace)
            n_from_trace += 1

        rows.append(row)

    rows.sort(key=lambda r: r["task_id"])
    write_jsonl(output_path, rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")
    print(f"  From original JSONL : {n_from_orig}")
    print(f"  Reconstructed       : {n_from_trace}")


if __name__ == "__main__":
    main()

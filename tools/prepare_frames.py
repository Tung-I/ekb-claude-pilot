#!/usr/bin/env python3
"""
Prepare FRAMES benchmark into EKB JSONL format.

FRAMES (Factuality, Retrieval, And Multi-step Evidenced Summarization) is a
multi-hop factual research benchmark released by Google DeepMind.

HuggingFace: google/frames-benchmark (single "test" split, 824 tasks)

Output (under data/frames/ by default):
  frames.jsonl  — all 824 tasks, normalized to EKB schema

Each output record:
  {
    "task_id":         "frames_test_00042",
    "benchmark":       "frames",
    "split":           "test",
    "question":        "...",
    "final_answer":    "...",
    "file_name":       null,
    "file_path":       null,
    "metadata": {
      "reasoning_types": "Multiple constraints",
      "reasoning_types_primary": "Multiple constraints",
      "wiki_links": [...],
      "frames_idx": 42
    }
  }

Usage:
  python tools/prepare_frames.py
  python tools/prepare_frames.py --output-dir data/frames
  python tools/prepare_frames.py --hf-split test
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_DATASET = "google/frames-benchmark"
DEFAULT_HF_SPLIT = "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "frames"


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_hf(dataset_name: str, hf_split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets not installed. Run: pip install datasets") from exc
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset(dataset_name, split=hf_split, token=hf_token or None)
    return [dict(row) for row in ds]


def parse_wiki_links(raw: Dict[str, Any]) -> List[str]:
    """Collect all non-null wiki link columns into a clean list."""
    links: List[str] = []

    # First try the consolidated wiki_links field (stored as a stringified Python list)
    wiki_links_field = raw.get("wiki_links")
    if wiki_links_field:
        try:
            parsed = ast.literal_eval(str(wiki_links_field))
            if isinstance(parsed, list):
                links = [str(u) for u in parsed if u]
                return links
        except Exception:
            pass

    # Fall back to individual wikipedia_link_N columns
    for col in [
        "wikipedia_link_1", "wikipedia_link_2", "wikipedia_link_3",
        "wikipedia_link_4", "wikipedia_link_5", "wikipedia_link_6",
        "wikipedia_link_7", "wikipedia_link_8", "wikipedia_link_9",
        "wikipedia_link_10", "wikipedia_link_11+",
    ]:
        val = raw.get(col)
        if val and str(val).strip():
            links.append(str(val).strip())
    return links


def primary_reasoning_type(reasoning_types: str) -> str:
    """Return the first (primary) reasoning type from a pipe-separated string."""
    if not reasoning_types:
        return "Unknown"
    return reasoning_types.split("|")[0].strip()


def normalize(raw: Dict[str, Any], idx: int, hf_split: str) -> Optional[Dict[str, Any]]:
    question = raw.get("Prompt") or raw.get("prompt") or raw.get("question")
    if not question or not str(question).strip():
        eprint(f"[WARN] Skipping record {idx}: missing question")
        return None

    final_answer = raw.get("Answer") or raw.get("answer")
    reasoning_types = str(raw.get("reasoning_types") or "").strip()
    wiki_links = parse_wiki_links(raw)

    frames_idx = raw.get("Unnamed: 0")
    if frames_idx is not None:
        task_id = f"frames_{hf_split}_{int(frames_idx):05d}"
    else:
        task_id = f"frames_{hf_split}_{idx:05d}"

    return {
        "task_id": task_id,
        "benchmark": "frames",
        "split": hf_split,
        "question": str(question).strip(),
        "final_answer": str(final_answer).strip() if final_answer is not None else None,
        "file_name": None,
        "file_path": None,
        "metadata": {
            "reasoning_types": reasoning_types,
            "reasoning_types_primary": primary_reasoning_type(reasoning_types),
            "wiki_links": wiki_links,
            "frames_idx": int(frames_idx) if frames_idx is not None else idx,
        },
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FRAMES into EKB JSONL format.")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT,
                        help="HuggingFace split name (default: test)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-file", default=None,
                        help="Override output filename (default: frames.jsonl)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_file = args.output_file or "frames.jsonl"
    out_path = out_dir / out_file

    print(f"Loading {args.hf_dataset} / split={args.hf_split} ...")
    raw_records = load_hf(args.hf_dataset, args.hf_split)
    print(f"  {len(raw_records)} records loaded")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for idx, raw in enumerate(raw_records):
        row = normalize(raw, idx, args.hf_split)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["task_id"])
    write_jsonl(out_path, rows)

    from collections import Counter
    type_counts = Counter(r["metadata"]["reasoning_types"] for r in rows)
    print(f"\nreasoning_types distribution ({len(type_counts)} unique values):")
    for rt, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:>4}  {rt}")

    print(f"\nTotal loaded : {len(raw_records)}")
    print(f"Skipped      : {skipped}")
    print(f"Written      : {len(rows)}")
    print(f"Output       : {out_path}")


if __name__ == "__main__":
    main()

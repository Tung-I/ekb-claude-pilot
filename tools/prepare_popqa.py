#!/usr/bin/env python3
"""
Download and normalize the PopQA benchmark into JSONL format.

PopQA (akariasai/PopQA on HuggingFace) is a single-hop QA dataset with
14,267 questions spanning 16 property types (e.g. director, author, genre).

Output:
  data/popqa/popqa.jsonl  — one record per task, normalized schema

Each record:
  task_id          str   "popqa_<id>"
  benchmark        str   "popqa"
  split            str   "test"
  question         str   natural-language question
  final_answer     str   canonical answer (obj field)
  possible_answers list  all acceptable answer strings (parsed from JSON)
  prop             str   property group label (e.g. "director")
  prop_id          int   Wikidata property numeric ID
  subj             str   subject entity name
  subj_id          int   Wikidata subject QID
  obj              str   object entity name
  obj_id           int   Wikidata object QID
  s_pop            int   subject popularity
  o_pop            int   object popularity
  metadata         dict  remaining raw fields

NOTE: `prop` is stored for downstream analysis and test-set stratification
only. It must NOT be used during KB retrieval or semantic-execution analysis
— the agent sees only the question text.

Usage:
    python tools/prepare_popqa.py
    python tools/prepare_popqa.py --output-dir data/popqa
    python tools/prepare_popqa.py --hf-dataset akariasai/PopQA --hf-split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "popqa"
DEFAULT_HF_DATASET = "akariasai/PopQA"
DEFAULT_HF_SPLIT = "test"


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def load_popqa_hf(dataset_name: str, split_name: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets not installed. Run: pip install datasets") from exc
    ds = load_dataset(dataset_name, split=split_name)
    return [dict(row) for row in ds]


def parse_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [value]
    return []


def normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    task_id = f"popqa_{raw['id']}"
    possible_answers = parse_json_list(raw.get("possible_answers", []))
    return {
        "task_id": task_id,
        "benchmark": "popqa",
        "split": "test",
        "question": str(raw["question"]).strip(),
        "final_answer": str(raw["obj"]).strip(),
        "possible_answers": possible_answers,
        "prop": str(raw["prop"]),
        "prop_id": int(raw["prop_id"]),
        "subj": str(raw["subj"]),
        "subj_id": int(raw["subj_id"]),
        "obj": str(raw["obj"]),
        "obj_id": int(raw["obj_id"]),
        "s_pop": int(raw["s_pop"]),
        "o_pop": int(raw["o_pop"]),
        "metadata": {
            "s_aliases": parse_json_list(raw.get("s_aliases", [])),
            "o_aliases": parse_json_list(raw.get("o_aliases", [])),
            "s_uri": raw.get("s_uri"),
            "o_uri": raw.get("o_uri"),
            "s_wiki_title": raw.get("s_wiki_title"),
            "o_wiki_title": raw.get("o_wiki_title"),
        },
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize PopQA into JSONL.")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for output file (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_path = out_dir / "popqa.jsonl"

    print(f"Loading {args.hf_dataset} split={args.hf_split} ...")
    raw_records = load_popqa_hf(args.hf_dataset, args.hf_split)
    print(f"Loaded {len(raw_records)} records")

    rows: List[Dict[str, Any]] = []
    errors = 0
    for raw in raw_records:
        try:
            rows.append(normalize(raw))
        except Exception as exc:
            eprint(f"[WARN] Skipping record id={raw.get('id')}: {exc}")
            errors += 1

    rows.sort(key=lambda r: r["task_id"])
    write_jsonl(out_path, rows)

    from collections import Counter
    prop_counts = Counter(r["prop"] for r in rows)
    print(f"\nWrote {len(rows)} tasks → {out_path}")
    print(f"Errors skipped: {errors}")
    print(f"\nProp group distribution:")
    for prop, n in sorted(prop_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {prop}")


if __name__ == "__main__":
    main()

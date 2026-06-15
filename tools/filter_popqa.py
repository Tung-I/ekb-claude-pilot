#!/usr/bin/env python3
"""
Filter PopQA to remove all tasks whose question text appears more than once.

PopQA generates questions from fixed per-property templates applied to
(subject, object) pairs. Multiple Wikidata entities can share the same
surface name, producing identical question strings with different answers.
These ambiguous tasks are removed entirely (no representative is kept) to
ensure every question in the output has exactly one correct answer.

Input : data/popqa/popqa.jsonl          (14,267 tasks, 16 prop groups)
Output: data/popqa/popqa_filtered.jsonl (~12,187 tasks, unambiguous only)

Usage:
    python tools/filter_popqa.py
    python tools/filter_popqa.py --input data/popqa/popqa.jsonl
    python tools/filter_popqa.py --output data/popqa/popqa_filtered.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = REPO_ROOT / "data" / "popqa" / "popqa.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "popqa" / "popqa_filtered.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove all PopQA tasks whose question appears more than once."
    )
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)
    print(f"Loaded {len(rows)} tasks from {input_path}")

    question_counts = Counter(r["question"] for r in rows)
    ambiguous_questions = {q for q, n in question_counts.items() if n > 1}

    filtered = [r for r in rows if r["question"] not in ambiguous_questions]
    removed  = [r for r in rows if r["question"] in ambiguous_questions]

    write_jsonl(output_path, filtered)

    # ── Stats ────────────────────────────────────────────────────────────────
    prop_before = Counter(r["prop"] for r in rows)
    prop_after  = Counter(r["prop"] for r in filtered)

    print(f"\nAmbiguous question groups : {len(ambiguous_questions)}")
    print(f"Tasks removed             : {len(removed)} "
          f"({100 * len(removed) / len(rows):.1f}%)")
    print(f"Tasks kept                : {len(filtered)} "
          f"({100 * len(filtered) / len(rows):.1f}%)")
    print(f"\n{'prop':<20}  {'before':>7}  {'after':>7}  {'removed':>8}")
    print("-" * 50)
    for prop in sorted(prop_before, key=lambda p: -prop_before[p]):
        b = prop_before[prop]
        a = prop_after.get(prop, 0)
        print(f"{prop:<20}  {b:>7}  {a:>7}  {b - a:>8}")
    print("-" * 50)
    b = len(rows)
    a = len(filtered)
    print(f"{'TOTAL':<20}  {b:>7}  {a:>7}  {b - a:>8}")
    print(f"\nWrote {len(filtered)} tasks → {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split PopQA into a KB set and a test set using stratified sampling by `prop`.

Strategy:
  - Group tasks by `prop` label.
  - Within each group, randomly sample `--test-ratio` (default 0.20) of tasks
    as the test set; the remaining tasks form the KB.
  - A fixed random seed ensures reproducibility.

IMPORTANT: `prop` is used ONLY for stratification here. It must not be
exposed to the retrieval system or used during semantic-execution analysis.
After this split, treat popqa_kb.jsonl and popqa_test.jsonl as flat lists
of tasks — the `prop` field is metadata, not a retrieval signal.

Output (default: data/popqa/):
  popqa_test.jsonl   — ~20% of tasks, balanced across prop groups
  popqa_kb.jsonl     — ~80% of tasks, the knowledge base

Usage:
    python tools/split_popqa_kb_test.py
    python tools/split_popqa_kb_test.py --input data/popqa/popqa.jsonl
    python tools/split_popqa_kb_test.py --test-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "popqa" / "popqa.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "popqa"
DEFAULT_TEST_RATIO = 0.20
DEFAULT_SEED = 42


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


def stratified_split(
    rows: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kb_rows, test_rows) via stratified sampling on `prop`."""
    rng = random.Random(seed)

    by_prop: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_prop[r["prop"]].append(r)

    kb_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for prop, group in sorted(by_prop.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_test = max(1, math.floor(len(shuffled) * test_ratio))
        test_rows.extend(shuffled[:n_test])
        kb_rows.extend(shuffled[n_test:])

    kb_rows.sort(key=lambda r: r["task_id"])
    test_rows.sort(key=lambda r: r["task_id"])
    return kb_rows, test_rows


def print_split_table(
    by_prop_all: Dict[str, int],
    by_prop_test: Dict[str, int],
    by_prop_kb: Dict[str, int],
) -> None:
    header = f"{'prop':<20}  {'total':>6}  {'test':>6}  {'kb':>6}  {'test%':>6}"
    print(header)
    print("-" * len(header))
    for prop in sorted(by_prop_all, key=lambda p: -by_prop_all[p]):
        n = by_prop_all[prop]
        t = by_prop_test.get(prop, 0)
        k = by_prop_kb.get(prop, 0)
        print(f"{prop:<20}  {n:>6}  {t:>6}  {k:>6}  {100*t/n:>5.1f}%")
    total = sum(by_prop_all.values())
    total_t = sum(by_prop_test.values())
    total_k = sum(by_prop_kb.values())
    print("-" * len(header))
    print(f"{'TOTAL':<20}  {total:>6}  {total_t:>6}  {total_k:>6}  {100*total_t/total:>5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified KB/test split of PopQA by prop group."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Fraction of each prop group to assign to test (default: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--kb-output",
        default=None,
        help="Override KB output filename (default: popqa_kb.jsonl)",
    )
    parser.add_argument(
        "--test-output",
        default=None,
        help="Override test output filename (default: popqa_test.jsonl)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    kb_path = out_dir / (args.kb_output or "popqa_kb.jsonl")
    test_path = out_dir / (args.test_output or "popqa_test.jsonl")

    print(f"Input  : {input_path}")
    print(f"KB out : {kb_path}")
    print(f"Test out: {test_path}")
    print(f"Test ratio: {args.test_ratio:.0%}  Seed: {args.seed}")
    print()

    rows = load_jsonl(input_path)
    print(f"Loaded {len(rows)} tasks")

    kb_rows, test_rows = stratified_split(rows, args.test_ratio, args.seed)

    write_jsonl(kb_path, kb_rows)
    write_jsonl(test_path, test_rows)

    from collections import Counter
    by_prop_all  = Counter(r["prop"] for r in rows)
    by_prop_test = Counter(r["prop"] for r in test_rows)
    by_prop_kb   = Counter(r["prop"] for r in kb_rows)

    print_split_table(by_prop_all, by_prop_test, by_prop_kb)
    print(f"\nWrote {len(kb_rows)} KB tasks   → {kb_path}")
    print(f"Wrote {len(test_rows)} test tasks → {test_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split FRAMES into a KB set and a test set using stratified sampling by reasoning_types.

Strategy:
  - Group tasks by `reasoning_types` (the full pipe-separated label, e.g.
    "Multiple constraints | Temporal reasoning").
  - Within each group, randomly sample `--test-ratio` (default 0.25) of tasks
    as the test set; the remaining tasks form the KB.
  - Groups with fewer than 4 tasks get at least 1 task in test (floor → max(1, ...)).
  - A fixed random seed ensures reproducibility.

IMPORTANT: `reasoning_types` is used ONLY for stratification. It must not be
exposed to the retrieval system or used during semantic-execution analysis.
After this split, treat frames_kb.jsonl and frames_test.jsonl as flat lists
of tasks — the `reasoning_types` field is metadata, not a retrieval signal.

Output (default: data/frames/):
  frames_kb.jsonl    — ~75% of tasks, the knowledge base
  frames_test.jsonl  — ~25% of tasks, evaluation set

Usage:
    python tools/split_frames_kb_test.py
    python tools/split_frames_kb_test.py --input data/frames/frames.jsonl
    python tools/split_frames_kb_test.py --test-ratio 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "frames" / "frames.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "frames"
DEFAULT_TEST_RATIO = 0.25
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


def get_stratum_key(row: Dict[str, Any]) -> str:
    """Extract the full reasoning_types string for stratification."""
    return str(row.get("metadata", {}).get("reasoning_types") or "Unknown").strip()


def stratified_split(
    rows: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kb_rows, test_rows) via stratified sampling on reasoning_types."""
    rng = random.Random(seed)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[get_stratum_key(r)].append(r)

    kb_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for rt, group in sorted(by_type.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_test = max(1, math.floor(len(shuffled) * test_ratio))
        test_rows.extend(shuffled[:n_test])
        kb_rows.extend(shuffled[n_test:])

    kb_rows.sort(key=lambda r: r["task_id"])
    test_rows.sort(key=lambda r: r["task_id"])
    return kb_rows, test_rows


def print_split_table(rows: List[Dict[str, Any]], kb_ids: set, test_ids: set) -> None:
    by_type_all  = Counter(get_stratum_key(r) for r in rows)
    by_type_test = Counter(get_stratum_key(r) for r in rows if r["task_id"] in test_ids)
    by_type_kb   = Counter(get_stratum_key(r) for r in rows if r["task_id"] in kb_ids)

    col = 55
    header = f"{'reasoning_types':<{col}}  {'total':>6}  {'test':>5}  {'kb':>5}  {'test%':>6}"
    print(header)
    print("-" * len(header))
    for rt in sorted(by_type_all, key=lambda t: -by_type_all[t]):
        n = by_type_all[rt]
        t = by_type_test.get(rt, 0)
        k = by_type_kb.get(rt, 0)
        label = rt if len(rt) <= col else rt[: col - 3] + "..."
        print(f"{label:<{col}}  {n:>6}  {t:>5}  {k:>5}  {100 * t / n:>5.1f}%")
    total   = sum(by_type_all.values())
    total_t = sum(by_type_test.values())
    total_k = sum(by_type_kb.values())
    print("-" * len(header))
    print(
        f"{'TOTAL':<{col}}  {total:>6}  {total_t:>5}  {total_k:>5}"
        f"  {100 * total_t / total:>5.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified KB/test split of FRAMES by reasoning_types."
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
        help=f"Fraction of each reasoning_types group to assign to test (default: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--kb-output",
        default=None,
        help="Override KB output filename (default: frames_kb.jsonl)",
    )
    parser.add_argument(
        "--test-output",
        default=None,
        help="Override test output filename (default: frames_test.jsonl)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.output_dir)
    kb_path    = out_dir / (args.kb_output   or "frames_kb.jsonl")
    test_path  = out_dir / (args.test_output or "frames_test.jsonl")

    print(f"Input    : {input_path}")
    print(f"KB out   : {kb_path}")
    print(f"Test out : {test_path}")
    print(f"Test ratio: {args.test_ratio:.0%}  Seed: {args.seed}")
    print()

    rows = load_jsonl(input_path)
    print(f"Loaded {len(rows)} tasks")

    kb_rows, test_rows = stratified_split(rows, args.test_ratio, args.seed)

    write_jsonl(kb_path,   kb_rows)
    write_jsonl(test_path, test_rows)

    print()
    kb_ids   = {r["task_id"] for r in kb_rows}
    test_ids = {r["task_id"] for r in test_rows}
    print_split_table(rows, kb_ids, test_ids)

    print(f"\nWrote {len(kb_rows)} KB tasks   → {kb_path}")
    print(f"Wrote {len(test_rows)} test tasks → {test_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Recompute and patch exact_match in saved normalized_trace.json files.

Default target:
  results/claude_native/gaia_level1_expanded_claude_native_maxturn12

What it does:
- traverses */normalized_trace.json under the given run directory
- recomputes exact_match from final_answer_pred and ground_truth_answer
- overwrites normalized_trace.json only if exact_match changed
- optionally rewrites run-level results.jsonl for consistency

Usage:
  python scripts/fix_saved_exact_match.py

  python scripts/fix_saved_exact_match.py \
    --run-root results/claude_native/gaia_level1_expanded_claude_native_maxturn12

  python scripts/fix_saved_exact_match.py \
    --run-root results/claude_native/gaia_level1_expanded_claude_native_maxturn12 \
    --rewrite-results-jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional


LEADING_STRIP_CHARS = "\"'`“”‘’([{<"
TRAILING_STRIP_CHARS = "\"'`“”‘’.,;:!?)]}>"
CURRENCY_CHARS = "$€£¥₹₩₽₪฿₫₴₦₱₲₵₡₺₸₼₭₮₨"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_basic_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(LEADING_STRIP_CHARS + TRAILING_STRIP_CHARS)
    s = re.sub(rf"[{re.escape(TRAILING_STRIP_CHARS)}]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canonicalize_decimal_string(x: str) -> Optional[str]:
    s = unicodedata.normalize("NFKC", str(x)).strip()
    s = s.strip(LEADING_STRIP_CHARS + TRAILING_STRIP_CHARS)
    s = s.replace(" ", "")

    while s and s[0] in CURRENCY_CHARS:
        s = s[1:]
    while s and s[-1] in CURRENCY_CHARS:
        s = s[:-1]

    if not s:
        return None

    s_no_commas = s.replace(",", "")

    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s_no_commas):
        return None

    try:
        d = Decimal(s_no_commas)
    except InvalidOperation:
        return None

    out = format(d, "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    if out == "-0":
        out = "0"
    return out


def exact_match(pred: Optional[str], gold: Optional[str]) -> Optional[bool]:
    if pred is None or gold is None:
        return None

    pred_num = _canonicalize_decimal_string(pred)
    gold_num = _canonicalize_decimal_string(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num

    pred_norm = _normalize_basic_text(pred)
    gold_norm = _normalize_basic_text(gold)
    return pred_norm == gold_norm


def iter_normalized_trace_paths(run_root: Path):
    for path in sorted(run_root.glob("*/normalized_trace.json")):
        yield path


def maybe_rewrite_results_jsonl(run_root: Path) -> int:
    """
    Rewrite run-level results.jsonl from the patched normalized_trace.json files.
    """
    results_jsonl = run_root / "results.jsonl"
    if not results_jsonl.exists():
        return 0

    records = []
    for path in iter_normalized_trace_paths(run_root):
        try:
            obj = load_json(path)
            records.append(obj)
        except Exception:
            continue

    with results_jsonl.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=str,
        default="results/claude_native/gaia_level1_expanded_claude_native_maxturn12",
        help="Run directory containing per-task subfolders with normalized_trace.json",
    )
    parser.add_argument(
        "--rewrite-results-jsonl",
        action="store_true",
        help="Also rewrite run-root/results.jsonl from patched normalized_trace.json files.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    n_total = 0
    n_changed = 0
    n_true_to_false = 0
    n_false_to_true = 0
    n_none_to_true = 0
    n_none_to_false = 0

    changed_examples = []

    for path in iter_normalized_trace_paths(run_root):
        n_total += 1
        obj: Dict[str, Any] = load_json(path)

        pred = obj.get("final_answer_pred")
        gold = obj.get("ground_truth_answer")
        old = obj.get("exact_match")
        new = exact_match(pred, gold)

        if old != new:
            obj["exact_match"] = new
            write_json(path, obj)
            n_changed += 1

            if old is True and new is False:
                n_true_to_false += 1
            elif old is False and new is True:
                n_false_to_true += 1
            elif old is None and new is True:
                n_none_to_true += 1
            elif old is None and new is False:
                n_none_to_false += 1

            if len(changed_examples) < 20:
                changed_examples.append({
                    "query_id": obj.get("query_id"),
                    "old_exact_match": old,
                    "new_exact_match": new,
                    "pred": pred,
                    "gold": gold,
                    "path": str(path),
                })

    rewritten = 0
    if args.rewrite_results_jsonl:
        rewritten = maybe_rewrite_results_jsonl(run_root)

    print(f"Run root: {run_root}")
    print(f"Total normalized traces checked: {n_total}")
    print(f"Files changed: {n_changed}")
    print(f"False -> True: {n_false_to_true}")
    print(f"True -> False: {n_true_to_false}")
    print(f"None -> True: {n_none_to_true}")
    print(f"None -> False: {n_none_to_false}")
    if args.rewrite_results_jsonl:
        print(f"results.jsonl records rewritten: {rewritten}")

    if changed_examples:
        print("\nSample changed cases:")
        for ex in changed_examples:
            print(json.dumps(ex, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Prepare GAIA benchmark into level-split JSONL files.

Outputs (under data/gaia/ by default):
  gaia_lv1.jsonl  — Level 1 tasks
  gaia_lv2.jsonl  — Level 2 tasks
  gaia_lv3.jsonl  — Level 3 tasks

Each HF config covers a single level, so run once per config:
  python scripts/prepare_gaia.py --from-hf --hf-config 2023_level1
  python scripts/prepare_gaia.py --from-hf --hf-config 2023_level2
  python scripts/prepare_gaia.py --from-hf --hf-config 2023_level3

From a local file containing mixed levels:
  python scripts/prepare_gaia.py --input path/to/gaia.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "records", "examples", "validation", "train", "test"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"{path} is a JSON object with no recognized record-list key.")
    raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


def load_local(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .json or .jsonl")


def load_hf(dataset_name: str, config_name: str, split_name: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets not installed. Run: pip install datasets") from exc
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset(dataset_name, config_name, split=split_name, token=hf_token or None)
    return [dict(row) for row in ds]


def first_present(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default


def coerce_level(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip().lower()
    mapping = {"1": 1, "level 1": 1, "l1": 1, "easy": 1,
               "2": 2, "level 2": 2, "l2": 2, "medium": 2,
               "3": 3, "level 3": 3, "l3": 3, "hard": 3}
    if s in mapping:
        return mapping[s]
    try:
        return int(s)
    except ValueError:
        return None


def normalize(raw: Dict[str, Any], idx: int, benchmark: str, split_name: str,
              raw_root: Path, keep_raw: bool) -> Dict[str, Any]:
    task_id = first_present(raw, ["task_id", "id", "uid", "question_id", "instance_id"],
                            f"{benchmark}_{split_name}_{idx:05d}")
    question = first_present(raw, ["Question", "question", "prompt", "input", "query", "task", "problem"])
    if question is None:
        raise ValueError(f"Record {idx} has no question field. Keys={list(raw.keys())}")
    final_answer = first_present(raw, ["Final answer", "final_answer", "answer", "target", "gold", "label"])
    level = coerce_level(first_present(raw, ["Level", "level", "difficulty"]))
    file_name = first_present(raw, ["file_name", "filename", "attachment", "file"])
    asset_path = None
    if file_name:
        candidate = raw_root / str(file_name)
        if candidate.exists():
            asset_path = str(candidate.resolve())
    row: Dict[str, Any] = {
        "task_id": str(task_id),
        "benchmark": benchmark,
        "split": split_name,
        "level": level,
        "question": str(question).strip(),
        "final_answer": None if final_answer is None else str(final_answer).strip(),
        "file_name": None if file_name is None else str(file_name),
        "file_path": asset_path,
        "metadata": {"source_keys": sorted(raw.keys())},
    }
    if keep_raw:
        row["raw_record"] = raw
    return row


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GAIA into level-split JSONL files.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Local .json or .jsonl file.")
    src.add_argument("--from-hf", action="store_true", help="Load from Hugging Face.")
    parser.add_argument("--hf-dataset", type=str, default="gaia-benchmark/GAIA")
    parser.add_argument("--hf-config", type=str, default="2023_level1",
                        help="HF config name, e.g. 2023_level1 / 2023_level2 / 2023_level3")
    parser.add_argument("--hf-split", type=str, default="validation")
    parser.add_argument("--benchmark", type=str, default="gaia")
    parser.add_argument("--split-name", type=str, default=None,
                        help="Logical split name written to output records. Defaults to hf-split or file stem.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files. Defaults to $EKB_ROOT/data/gaia.")
    parser.add_argument("--keep-raw-record", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    ekb_root = Path(os.environ.get("EKB_ROOT", Path.cwd()))
    out_dir = Path(args.output_dir) if args.output_dir else ekb_root / "data" / "gaia"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_hf:
        split_name = args.split_name or args.hf_split
        gaia_raw = ekb_root / "data" / "gaia" / "raw"
        raw_records = load_hf(args.hf_dataset, args.hf_config, args.hf_split)
        raw_root = gaia_raw
    else:
        input_path = Path(args.input).expanduser().resolve()
        split_name = args.split_name or input_path.stem
        raw_records = load_local(input_path)
        raw_root = input_path.parent

    by_level: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for idx, raw in enumerate(raw_records):
        try:
            row = normalize(raw, idx, args.benchmark, split_name, raw_root, args.keep_raw_record)
        except Exception as exc:
            skipped += 1
            eprint(f"[WARN] Skipping record {idx}: {exc}")
            continue
        level = row.get("level")
        if level not in (1, 2, 3):
            skipped += 1
            eprint(f"[WARN] Skipping record {idx}: unrecognized level={level!r}")
            continue
        by_level[level].append(row)

    for level, rows in sorted(by_level.items()):
        rows.sort(key=lambda x: x["task_id"])
        out_path = out_dir / f"gaia_lv{level}.jsonl"
        write_jsonl(out_path, rows)
        print(f"gaia_lv{level}.jsonl: {len(rows)} tasks → {out_path}")

    print(f"\nTotal loaded: {len(raw_records)}  skipped: {skipped}  written: {sum(len(v) for v in by_level.values())}")


if __name__ == "__main__":
    main()

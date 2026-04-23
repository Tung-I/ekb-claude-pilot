#!/usr/bin/env python3
"""
Prepare GAIA Level 1 into normalized JSONL and split into shards.

Outputs:
  - {GAIA_PREPARED_PATH}/gaia_level1_prepared.jsonl
  - {GAIA_SHARD_PATH}/gaia_level1_shard_00.jsonl
  - ...
  - {GAIA_SHARD_PATH}/gaia_level1_shard_09.jsonl

Supported inputs:
  1) Local JSONL / JSON file via --input
  2) Hugging Face dataset via --from-hf

Examples:
python scripts/prepare_gaia_level1.py \
  --from-hf \
  --hf-dataset gaia-benchmark/GAIA \
  --hf-config 2023_level1 \
  --hf-split validation \
  --num-shards 10 \
  --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def mkdirp(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(obj)}")
            rows.append(obj)
    return rows


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise ValueError(f"{path} contains a list, but not all entries are JSON objects.")
        return data

    if isinstance(data, dict):
        # Common patterns:
        # {"data": [...]} or {"validation": [...]} or {"items": [...]}
        for key in ("data", "items", "records", "examples", "validation", "train", "test"):
            if key in data and isinstance(data[key], list):
                if not all(isinstance(x, dict) for x in data[key]):
                    raise ValueError(f"{path}[{key}] is not a list of JSON objects.")
                return data[key]
        raise ValueError(
            f"{path} is a JSON object, but no obvious record list was found "
            "(expected keys like data/items/records/examples/validation)."
        )

    raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


def load_local_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .json or .jsonl")


def load_hf_records(dataset_name: str, config_name: str, split_name: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not installed. Please run: pip install datasets"
        ) from exc

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        ds = load_dataset(dataset_name, config_name, split=split_name, token=hf_token)
    else:
        ds = load_dataset(dataset_name, config_name, split=split_name)
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
    if s in {"1", "level 1", "l1", "easy"}:
        return 1
    if s in {"2", "level 2", "l2", "medium"}:
        return 2
    if s in {"3", "level 3", "l3", "hard"}:
        return 3
    try:
        return int(s)
    except ValueError:
        return None


def normalize_record(
    raw: Dict[str, Any],
    idx: int,
    benchmark: str,
    split_name: str,
    raw_root: Path,
) -> Dict[str, Any]:
    task_id = first_present(raw, ["task_id", "id", "uid", "question_id", "instance_id"], None)
    if task_id is None:
        task_id = f"{benchmark}_{split_name}_{idx:05d}"

    question = first_present(
        raw,
        ["Question", "question", "prompt", "input", "query", "task", "problem"],
        None,
    )
    if question is None:
        raise ValueError(f"Record {idx} is missing a question-like field. Keys={list(raw.keys())}")

    final_answer = first_present(
        raw,
        ["Final answer", "final_answer", "answer", "target", "gold", "label"],
        None,
    )

    level = coerce_level(first_present(raw, ["Level", "level", "difficulty"], None))
    file_name = first_present(raw, ["file_name", "filename", "attachment", "file"], None)

    asset_path = None
    if file_name:
        candidate = raw_root / str(file_name)
        if candidate.exists():
            asset_path = str(candidate.resolve())

    normalized: Dict[str, Any] = {
        "task_id": str(task_id),
        "benchmark": benchmark,
        "split": split_name,
        "level": level,
        "question": str(question).strip(),
        "final_answer": None if final_answer is None else str(final_answer).strip(),
        "file_name": None if file_name is None else str(file_name),
        "file_path": asset_path,
        "metadata": {
            "source_keys": sorted(list(raw.keys())),
        },
        "raw_record": raw,
    }
    return normalized


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_evenly(rows: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    n = len(rows)
    base = n // num_shards
    extra = n % num_shards
    shards: List[List[Dict[str, Any]]] = []
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        end = start + size
        shards.append(rows[start:end])
        start = end
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to local GAIA .json or .jsonl file.",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Load from Hugging Face dataset instead of local file.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="gaia-benchmark/GAIA",
        help="HF dataset name if --from-hf is used.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="validation",
        help="HF split name if --from-hf is used.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gaia",
        help="Benchmark name to write into normalized records.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Logical split name written to output records. Defaults to hf split or file stem.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of output shards.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle Level-1 tasks before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--keep-raw-record",
        action="store_true",
        help="Keep the full raw record in each normalized row. Useful for debugging but increases file size.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="2023_level1",
        help="HF dataset config, e.g. 2023_level1",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    ekb_root = Path(os.environ.get("EKB_ROOT", Path.cwd()))
    gaia_raw_path = Path(os.environ.get("GAIA_RAW_PATH", ekb_root / "data" / "gaia" / "raw"))
    gaia_prepared_path = Path(
        os.environ.get("GAIA_PREPARED_PATH", ekb_root / "data" / "gaia" / "prepared")
    )
    gaia_shard_path = Path(os.environ.get("GAIA_SHARD_PATH", ekb_root / "data" / "gaia" / "shards"))

    mkdirp(gaia_raw_path)
    mkdirp(gaia_prepared_path)
    mkdirp(gaia_shard_path)

    if args.from_hf and args.input:
        raise ValueError("Use either --input or --from-hf, not both.")
    if not args.from_hf and not args.input:
        raise ValueError("Please provide either --input or --from-hf.")

    if args.from_hf:
        split_name = args.split_name or args.hf_split
        raw_records = load_hf_records(args.hf_dataset, args.hf_config, args.hf_split)
        raw_root = gaia_raw_path
    else:
        input_path = Path(args.input).expanduser().resolve()
        split_name = args.split_name or input_path.stem
        raw_records = load_local_records(input_path)
        raw_root = input_path.parent

    normalized_all: List[Dict[str, Any]] = []
    skipped = 0

    for idx, raw in enumerate(raw_records):
        try:
            row = normalize_record(
                raw=raw,
                idx=idx,
                benchmark=args.benchmark,
                split_name=split_name,
                raw_root=raw_root,
            )
        except Exception as exc:
            skipped += 1
            eprint(f"[WARN] Skipping record {idx}: {exc}")
            continue

        if row["level"] != 1:
            continue

        if not args.keep_raw_record:
            row.pop("raw_record", None)

        normalized_all.append(row)

    if not normalized_all:
        raise RuntimeError("No Level-1 records were found after normalization.")

    # Stable sort before optional shuffle for reproducibility.
    normalized_all.sort(key=lambda x: x["task_id"])

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(normalized_all)

    prepared_file = gaia_prepared_path / "gaia_level1_prepared.jsonl"
    write_jsonl(prepared_file, normalized_all)

    shards = split_evenly(normalized_all, args.num_shards)
    for shard_idx, shard_rows in enumerate(shards):
        shard_file = gaia_shard_path / f"gaia_level1_shard_{shard_idx:02d}.jsonl"
        write_jsonl(shard_file, shard_rows)

    print("Done.")
    print(f"Total raw records loaded: {len(raw_records)}")
    print(f"Skipped during normalization: {skipped}")
    print(f"Level-1 records written: {len(normalized_all)}")
    print(f"Prepared file: {prepared_file}")
    print(f"Shard dir: {gaia_shard_path}")
    for shard_idx, shard_rows in enumerate(shards):
        print(f"  shard_{shard_idx:02d}: {len(shard_rows)} rows")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate paraphrases for GAIA prompts using Claude.

For each source task, generates --copies-per-task paraphrases so the output
contains (1 original + N paraphrases) = (N+1) variants per task, e.g.
--copies-per-task 3 → gaia_lv1_x4.jsonl.

Features
--------
- Gap-aware: reads the existing output file and skips tasks that already have
  a full set of paraphrases; only generates for missing or incomplete tasks.
- Cache-first: per-task intermediate results are stored under
  traces/paraphrases/<input-stem>/<task-id>/paraphrases.json; if the cache
  already holds enough paraphrases they are reused without an API call.
- Resume-safe: the output file is rewritten after every completed task.
- Quota guard: stops on suspected rate-limit / usage-limit errors.

Typical usage
-------------
Dry run (2 tasks):
  python tools/generate_gaia_paraphrases.py \
    --input data/gaia/gaia_lv1.jsonl \
    --limit 2 --model sonnet --effort low --overwrite

Full run:
  python tools/generate_gaia_paraphrases.py \
    --input data/gaia/gaia_lv1.jsonl \
    --model sonnet --effort medium

Fill gaps in an existing output file (default behaviour without --overwrite):
  python tools/generate_gaia_paraphrases.py \
    --input data/gaia/gaia_lv1.jsonl

Level 2 with 4 paraphrases per task (x5):
  python tools/generate_gaia_paraphrases.py \
    --input data/gaia/gaia_lv2.jsonl \
    --copies-per-task 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


RATE_LIMIT_PATTERNS = [
    "rate limit", "usage limit", "limit reached", "quota",
    "too many requests", "try again later", "credit balance",
    "exceeded your", "daily limit", "message limit", "capacity",
]

SUCCESS_MARKER = "completed"


# --------------------------------------------------------------------------- #
# Schema / prompt (copies_per_task-aware)                                     #
# --------------------------------------------------------------------------- #

def build_schema(copies_per_task: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "paraphrases": {
                "type": "array",
                "minItems": copies_per_task,
                "maxItems": copies_per_task,
                "items": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            "checks": {
                "type": "object",
                "properties": {
                    "answer_should_remain_same": {"type": "boolean"},
                    "constraints_preserved": {"type": "boolean"},
                    "answer_format_preserved": {"type": "boolean"},
                    "file_reference_preserved": {"type": "boolean"},
                },
                "required": [
                    "answer_should_remain_same", "constraints_preserved",
                    "answer_format_preserved", "file_reference_preserved",
                ],
                "additionalProperties": False,
            },
            "notes": {"type": "string"},
        },
        "required": ["paraphrases", "checks", "notes"],
        "additionalProperties": False,
    }


def build_system_prompt(copies_per_task: int) -> str:
    return textwrap.dedent(f"""
        You are creating paraphrases for a benchmark dataset.

        Rules:
        - Do NOT solve the task.
        - Do NOT reveal or guess the answer.
        - Do NOT use external tools such as WebSearch, WebFetch, Bash, Read, Glob, Grep, Edit, or Write.
        - Return the required structured output directly.
        - Preserve meaning exactly.
        - Preserve every named entity, number, date, unit, constraint, and answer format requirement.
        - If the original question references an attached/provided file, each paraphrase must still
          clearly refer to that same supporting file type.
        - Make the wording genuinely different across the {copies_per_task} paraphrases.
        - Return only structured output.
    """).strip()


def infer_file_type_hint(task: Dict[str, Any]) -> str:
    joined = " ".join([
        str(task.get("question", "")),
        str(task.get("file_name") or ""),
        str(task.get("file_path") or ""),
    ]).casefold()
    hints = [
        ("spreadsheet", [".xlsx", ".xls", ".csv", "excel", "spreadsheet"]),
        ("image",       [".png", ".jpg", ".jpeg", ".gif", ".webp", "image", "photo", "picture"]),
        ("audio",       [".mp3", ".wav", ".m4a", "audio"]),
        ("video",       [".mp4", ".mov", ".avi", "video"]),
        ("pdf/document",[".pdf", "pdf", "document"]),
        ("presentation",[".ppt", ".pptx", "powerpoint", "slides"]),
    ]
    for label, keys in hints:
        if any(k in joined for k in keys):
            return label
    return "supporting file"


def build_prompt(task: Dict[str, Any], copies_per_task: int) -> str:
    file_hint = infer_file_type_hint(task)
    parts = [
        "Create paraphrases for the following GAIA task.",
        f"Task ID: {task['task_id']}",
        f"Original question:\n{str(task['question']).strip()}",
        f"Number of paraphrases required: {copies_per_task}",
        "",
        "Paraphrase requirements:",
        "1) Preserve the exact meaning.",
        "2) Preserve all factual constraints, named entities, dates, numbers, units, and comparison conditions.",
        "3) Preserve the expected answer format.",
        "4) Do not add hints, extra facts, or solution steps.",
        f"5) If the question depends on a {file_hint}, each paraphrase must still clearly mention that same kind of supporting file.",
        "6) Make the wording materially different across the paraphrases.",
        "7) Do not solve the question.",
        "8) Do not use any external tools or perform research.",
        "9) Respond directly with the structured output schema in one shot.",
        "",
        "Return only structured output.",
    ]
    if task.get("file_name"):
        parts.append(f"Dataset metadata file_name: {task['file_name']}")
    if task.get("file_path"):
        parts.append("A local file_path exists in the dataset metadata, but do not inspect it.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(task_id))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Gap detection                                                               #
# --------------------------------------------------------------------------- #

def load_existing_para_counts(output_path: Path) -> Dict[str, int]:
    """Return {source_task_id: number_of_paraphrases} from an existing output file."""
    if not output_path.exists():
        return {}
    counts: Dict[str, int] = defaultdict(int)
    try:
        rows = load_jsonl(output_path)
    except Exception as exc:
        print(f"[warn] could not parse existing output {output_path}: {exc}", file=sys.stderr)
        return {}
    for row in rows:
        if row.get("is_paraphrase"):
            src = str(row.get("source_task_id", ""))
            if src:
                counts[src] += 1
    return dict(counts)


def seed_cache_from_output(
    output_path: Path,
    cache_root: Path,
    copies_per_task: int,
    all_tasks: List[Dict[str, Any]],
) -> int:
    """
    One-time migration: for tasks that are complete in the output file but have no
    cache entry yet, write a paraphrases.json cache file so materialize_output()
    can reconstruct them without an API call.

    Returns the number of tasks seeded.
    """
    if not output_path.exists():
        return 0

    try:
        rows = load_jsonl(output_path)
    except Exception:
        return 0

    # Group paraphrase questions by source_task_id, in index order
    para_map: Dict[str, List[str]] = defaultdict(list)
    index_map: Dict[str, Dict[int, str]] = defaultdict(dict)
    for row in rows:
        if not row.get("is_paraphrase"):
            continue
        src = str(row.get("source_task_id", ""))
        idx = int(row.get("paraphrase_index", 0))
        q = str(row.get("question", ""))
        if src and idx > 0 and q:
            index_map[src][idx] = q

    # Build task lookup for source_task records
    task_by_id = {str(t["task_id"]): t for t in all_tasks}

    seeded = 0
    for src_id, idx_to_q in index_map.items():
        if len(idx_to_q) < copies_per_task:
            continue  # incomplete — skip, let gap detection handle it
        task_dir = cache_root / safe_task_id(src_id)
        para_file = task_dir / "paraphrases.json"
        if para_file.exists():
            continue  # cache already present
        questions = [idx_to_q[i] for i in sorted(idx_to_q) if i <= copies_per_task]
        if len(questions) < copies_per_task:
            continue
        questions = questions[:copies_per_task]
        source_task = task_by_id.get(src_id, {"task_id": src_id})
        cache_payload = {
            "status": SUCCESS_MARKER,
            "query_id": src_id,
            "source_task": source_task,
            "paraphrase_questions": questions,
            "copies_per_task": copies_per_task,
            "usage": {},
            "total_tokens": None,
            "model_requested": "seeded_from_output",
            "effort": "seeded_from_output",
            "generated_at": utcnow(),
        }
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(para_file, cache_payload)
        seeded += 1

    return seeded


# --------------------------------------------------------------------------- #
# Claude invocation                                                           #
# --------------------------------------------------------------------------- #

def parse_cli_json(stdout_text: str) -> Optional[Dict[str, Any]]:
    stdout_text = stdout_text.strip()
    if not stdout_text:
        return None
    try:
        obj = json.loads(stdout_text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def extract_usage(parsed_cli: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not parsed_cli:
        return {}
    usage = parsed_cli.get("usage")
    return usage if isinstance(usage, dict) else {}


def extract_total_tokens(usage: Dict[str, Any]) -> Optional[int]:
    if not usage:
        return None
    total = 0
    found = False
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        v = usage.get(key)
        if isinstance(v, (int, float)):
            total += int(v)
            found = True
    return total if found else None


def looks_like_rate_limit(stdout: str, stderr: str) -> Optional[str]:
    # Scan stderr only — stdout contains paraphrase text and can produce false positives
    # (e.g. "quotation" matching "quota").
    hay = stderr.casefold()
    for pat in RATE_LIMIT_PATTERNS:
        if pat in hay:
            return pat
    return None


def explain_cli_failure(parsed_cli: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(parsed_cli, dict):
        return None
    subtype = parsed_cli.get("subtype")
    if subtype == "error_max_turns":
        return "Claude hit max_turns before returning valid structured output."
    if parsed_cli.get("is_error"):
        return (f"Claude CLI error: subtype={subtype!r}, "
                f"stop_reason={parsed_cli.get('stop_reason')!r}, "
                f"errors={parsed_cli.get('errors')!r}")
    return None


def normalize_text(x: str) -> str:
    return " ".join(str(x).strip().split()).casefold()


def validate_paraphrase_payload(
    original_question: str,
    payload: Dict[str, Any],
    copies_per_task: int,
) -> Tuple[bool, str, List[str]]:
    if not isinstance(payload, dict):
        return False, "structured output is not a dict", []
    paraphrases = payload.get("paraphrases")
    checks = payload.get("checks")
    if not isinstance(paraphrases, list) or len(paraphrases) != copies_per_task:
        return False, f"expected exactly {copies_per_task} paraphrases, got {len(paraphrases) if isinstance(paraphrases, list) else type(paraphrases)}", []
    if not isinstance(checks, dict):
        return False, "missing checks object", []
    for k in ("answer_should_remain_same", "constraints_preserved", "answer_format_preserved", "file_reference_preserved"):
        if checks.get(k) is not True:
            return False, f"check {k!r} is not true", []
    orig_norm = normalize_text(original_question)
    questions: List[str] = []
    for item in paraphrases:
        if not isinstance(item, dict):
            return False, "paraphrase item is not a dict", []
        q = str(item.get("question", "")).strip()
        if not q:
            return False, "empty paraphrase question", []
        questions.append(q)
    norms = [normalize_text(q) for q in questions]
    if len(set(norms)) != len(norms):
        return False, "duplicate paraphrases detected", []
    if any(n == orig_norm for n in norms):
        return False, "a paraphrase is identical to the original", []
    return True, "ok", questions


def run_claude(
    task: Dict[str, Any],
    task_dir: Path,
    copies_per_task: int,
    model: str,
    effort: str,
    max_turns: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    prompt = build_prompt(task, copies_per_task)
    write_text(task_dir / "prompt.txt", prompt)
    write_json(task_dir / "task_record.json", task)

    schema = build_schema(copies_per_task)
    system_prompt = build_system_prompt(copies_per_task)
    session_id = str(uuid.uuid4())

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(schema, separators=(",", ":")),
        "--append-system-prompt", system_prompt,
        "--session-id", session_id,
        "--name", str(task["task_id"]),
        "--max-turns", str(max_turns),
        "--model", model,
        "--effort", effort,
    ]
    write_json(task_dir / "cli_command.json", {"cmd": cmd})

    started_at = utcnow()
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    ended_at = utcnow()
    duration_ms = int((time.time() - t0) * 1000)

    write_text(task_dir / "claude_stdout.txt", proc.stdout)
    write_text(task_dir / "claude_stderr.txt", proc.stderr)

    parsed_cli = parse_cli_json(proc.stdout)
    if parsed_cli is not None:
        write_json(task_dir / "claude_output.json", parsed_cli)

    usage = extract_usage(parsed_cli)
    total_tokens = extract_total_tokens(usage)
    quota_pattern = looks_like_rate_limit(proc.stdout, proc.stderr)
    cli_failure_msg = explain_cli_failure(parsed_cli)

    structured_output = None
    if isinstance(parsed_cli, dict):
        structured_output = parsed_cli.get("structured_output")
        if structured_output is None and "paraphrases" in parsed_cli:
            structured_output = parsed_cli

    ok = False
    validation_msg = "no structured output"
    questions: List[str] = []
    if isinstance(structured_output, dict):
        ok, validation_msg, questions = validate_paraphrase_payload(
            str(task["question"]), structured_output, copies_per_task,
        )
    elif cli_failure_msg:
        validation_msg = cli_failure_msg

    result = {
        "query_id": task["task_id"],
        "session_id": session_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "returncode": proc.returncode,
        "usage": usage,
        "total_tokens": total_tokens,
        "quota_pattern": quota_pattern,
        "cli_failure_msg": cli_failure_msg,
        "structured_output_valid": ok,
        "validation_msg": validation_msg,
        "paraphrase_questions": questions,
    }
    write_json(task_dir / "result.json", result)

    if ok:
        cache_payload = {
            "status": SUCCESS_MARKER,
            "query_id": task["task_id"],
            "source_task": task,
            "paraphrase_questions": questions,
            "copies_per_task": copies_per_task,
            "usage": usage,
            "total_tokens": total_tokens,
            "model_requested": model,
            "effort": effort,
            "generated_at": ended_at,
        }
        write_json(task_dir / "paraphrases.json", cache_payload)

    return result


# --------------------------------------------------------------------------- #
# Output materialization                                                      #
# --------------------------------------------------------------------------- #

def materialize_output(
    all_tasks: List[Dict[str, Any]],
    cache_root: Path,
    output_path: Path,
    copies_per_task: int,
) -> Tuple[int, int]:
    """Rebuild output JSONL from per-task cache. Returns (n_originals, n_paraphrases)."""
    rows: List[Dict[str, Any]] = []
    n_orig = 0
    n_para = 0

    for task in all_tasks:
        orig = {**task, "source_task_id": task["task_id"], "variant": "original",
                "is_paraphrase": False, "paraphrase_index": 0}
        rows.append(orig)
        n_orig += 1

        task_dir = cache_root / safe_task_id(task["task_id"])
        para_file = task_dir / "paraphrases.json"
        if not para_file.exists():
            continue
        try:
            payload = load_json(para_file)
        except Exception:
            continue
        if payload.get("status") != SUCCESS_MARKER:
            continue
        questions = payload.get("paraphrase_questions", [])
        for idx, q in enumerate(questions, start=1):
            rec = {**task,
                   "task_id": f"{task['task_id']}__para{idx}",
                   "source_task_id": task["task_id"],
                   "variant": f"paraphrase_{idx}",
                   "is_paraphrase": True,
                   "paraphrase_index": idx,
                   "question": q}
            rows.append(rec)
            n_para += 1

    write_jsonl(output_path, rows)
    return n_orig, n_para


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class RunStats:
    selected: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    skipped: int = 0
    failed: int = 0
    stopped_on_limit: bool = False
    stop_reason: Optional[str] = None
    cumulative_tokens: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GAIA paraphrases with gap-filling support.")
    parser.add_argument("--input", required=True, help="Path to gaia_lv{N}.jsonl")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path. Defaults to data/gaia_paraphrased/gaia_lv{N}_x{M}.jsonl")
    parser.add_argument("--cache-root", default=None,
                        help="Root dir for per-task intermediate files. Defaults to traces/paraphrases/<input-stem>/")
    parser.add_argument("--copies-per-task", type=int, default=3,
                        help="Number of paraphrases per original task (default: 3 → x4 output)")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--task-id", default=None, help="Run a single task by ID.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even for tasks that already have a full set of paraphrases.")
    parser.add_argument("--max-cumulative-tokens", type=int, default=None)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    ekb_root = Path(os.environ.get("EKB_ROOT", Path.cwd()))
    input_path = Path(args.input).expanduser().resolve()
    all_tasks = load_jsonl(input_path)

    # Derive output path from input filename and copies_per_task if not given
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        x_label = f"x{args.copies_per_task + 1}"
        stem = input_path.stem  # e.g. "gaia_lv1"
        output_path = ekb_root / "data" / "gaia_paraphrased" / f"{stem}_{x_label}.jsonl"

    # Derive cache root from input stem if not given
    if args.cache_root:
        cache_root = Path(args.cache_root).expanduser().resolve()
    else:
        cache_root = ekb_root / "traces" / "paraphrases" / input_path.stem

    print(f"Input:      {input_path}  ({len(all_tasks)} tasks)")
    print(f"Output:     {output_path}")
    print(f"Cache root: {cache_root}")
    print(f"Copies/task: {args.copies_per_task}  (output label: x{args.copies_per_task + 1})")

    # Filter by task-id / limit
    tasks = all_tasks
    if args.task_id:
        tasks = [t for t in tasks if str(t.get("task_id")) == args.task_id]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if not tasks:
        print("No tasks selected.")
        return

    # Gap detection: find which tasks already have a complete set of paraphrases
    existing_counts = load_existing_para_counts(output_path)
    n_already_complete = sum(1 for t in tasks if existing_counts.get(str(t["task_id"]), 0) >= args.copies_per_task)
    n_need_work = len(tasks) - n_already_complete
    print(f"\nGap analysis: {n_already_complete}/{len(tasks)} tasks already complete, "
          f"{n_need_work} need generation.")

    # Seed per-task cache from the existing output file so materialize_output()
    # can reconstruct complete tasks without re-running Claude.
    n_seeded = seed_cache_from_output(output_path, cache_root, args.copies_per_task, all_tasks)
    if n_seeded:
        print(f"Seeded {n_seeded} task(s) into cache from existing output.")

    stats = RunStats(selected=len(tasks))
    started_at = utcnow()

    for idx, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        existing_count = existing_counts.get(task_id, 0)

        # Skip if already complete (unless --overwrite)
        if existing_count >= args.copies_per_task and not args.overwrite:
            print(f"[{idx}/{len(tasks)}] skip {task_id}  ({existing_count} paraphrases, complete)")
            stats.skipped += 1
            continue

        task_dir = cache_root / safe_task_id(task_id)
        para_file = task_dir / "paraphrases.json"

        # Cache-first: check if per-task cache already has enough paraphrases
        cache_ok = False
        if para_file.exists() and not args.overwrite:
            try:
                cached = load_json(para_file)
                cached_count = len(cached.get("paraphrase_questions", []))
                if cached.get("status") == SUCCESS_MARKER and cached_count >= args.copies_per_task:
                    print(f"[{idx}/{len(tasks)}] cache-hit {task_id}  ({cached_count} in cache, rematerializing)")
                    stats.cache_hits += 1
                    cache_ok = True
            except Exception:
                pass

        if not cache_ok:
            # Need to call Claude
            reason = "overwrite" if args.overwrite else f"only {existing_count}/{args.copies_per_task} paraphrases"
            print(f"[{idx}/{len(tasks)}] generate {task_id}  ({reason})")
            task_dir.mkdir(parents=True, exist_ok=True)
            stats.api_calls += 1

            try:
                result = run_claude(task, task_dir, args.copies_per_task,
                                    args.model, args.effort, args.max_turns, args.timeout_sec)
            except subprocess.TimeoutExpired as exc:
                write_json(task_dir / "result.json", {"status": "failed", "query_id": task_id,
                           "failure_type": "timeout", "detail": str(exc), "generated_at": utcnow()})
                print("  -> timeout")
                stats.failed += 1
                continue
            except Exception as exc:
                write_json(task_dir / "result.json", {"status": "failed", "query_id": task_id,
                           "failure_type": "exception", "detail": repr(exc), "generated_at": utcnow()})
                print(f"  -> exception: {exc!r}")
                stats.failed += 1
                continue

            if result.get("total_tokens"):
                stats.cumulative_tokens += int(result["total_tokens"])

            if result.get("quota_pattern"):
                print(f"  -> stop: suspected usage/rate limit ({result['quota_pattern']})")
                stats.stopped_on_limit = True
                stats.stop_reason = f"rate limit pattern: {result['quota_pattern']}"
                break

            if result.get("structured_output_valid"):
                print("  -> success")
            else:
                print(f"  -> failed ({result.get('validation_msg')})")
                stats.failed += 1
                continue

        # Rebuild output after every processed task
        n_orig, n_para = materialize_output(all_tasks, cache_root, output_path, args.copies_per_task)
        print(f"  -> output: {n_orig} originals + {n_para} paraphrases = {n_orig + n_para} total")

        if (args.max_cumulative_tokens is not None
                and stats.cumulative_tokens >= args.max_cumulative_tokens):
            stats.stopped_on_limit = True
            stats.stop_reason = f"max cumulative tokens reached: {args.max_cumulative_tokens}"
            print(f"  -> stop: {stats.stop_reason}")
            break

        if args.sleep_sec > 0 and not cache_ok:
            time.sleep(args.sleep_sec)

    # Final materialization
    n_orig, n_para = materialize_output(all_tasks, cache_root, output_path, args.copies_per_task)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "cache_root": str(cache_root),
        "copies_per_task": args.copies_per_task,
        "started_at": started_at,
        "ended_at": utcnow(),
        "tasks_selected": stats.selected,
        "skipped_already_complete": stats.skipped,
        "cache_hits_rematerialized": stats.cache_hits,
        "api_calls_made": stats.api_calls,
        "failed": stats.failed,
        "stopped_on_limit": stats.stopped_on_limit,
        "stop_reason": stats.stop_reason,
        "cumulative_tokens_this_run": stats.cumulative_tokens,
        "output_originals": n_orig,
        "output_paraphrases": n_para,
        "output_total": n_orig + n_para,
    }
    write_json(cache_root / "summary.json", summary)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

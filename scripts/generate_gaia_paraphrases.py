#!/usr/bin/env python3
"""
Generate paraphrases for GAIA prompts using Claude Code.

Goal
----
For each input task, ask Claude Code to produce N paraphrases (default: 3)
that preserve meaning, constraints, and expected answer format.

Features
--------
- Resume-safe: skips tasks that already have completed outputs
- Graceful stop on suspected quota / rate-limit / usage-limit errors
- Optional manual stop by cumulative total_tokens
- Rebuilds:
    1) paraphrases_only.partial.jsonl
    2) expanded_with_originals.partial.jsonl
  after every successful task, so you can resume next day safely

Typical usage
-------------
Dry run on 2 tasks:
  python scripts/generate_gaia_paraphrases.py \
  --input data/gaia/prepared/gaia_level1_prepared.jsonl \
  --output-root results/gaia_paraphrases \
  --limit 2 \
  --model sonnet \
  --effort low \
  --overwrite

Full run:
  python scripts/generate_gaia_paraphrases.py \
    --input data/gaia/prepared/gaia_level1_prepared.jsonl \
    --output-root results/gaia_paraphrases \
    --model sonnet \
    --effort low

Resume next day:
  run the exact same command again; completed tasks will be skipped.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "paraphrases": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
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
                "answer_should_remain_same",
                "constraints_preserved",
                "answer_format_preserved",
                "file_reference_preserved",
            ],
            "additionalProperties": False,
        },
        "notes": {"type": "string"},
    },
    "required": ["paraphrases", "checks", "notes"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are creating paraphrases for a benchmark dataset.

    Rules:
    - Do NOT solve the task.
    - Do NOT reveal or guess the answer.
    - Do NOT use external tools such as WebSearch, WebFetch, Bash, Read, Glob, Grep, Edit, or Write.
    - Return the required structured output directly.
    - Preserve meaning exactly.
    - Preserve every named entity, number, date, unit, constraint, and answer format requirement.
    - If the original question references an attached/provided file, spreadsheet, image, audio, video, or document,
      each paraphrase must still clearly refer to that same supporting file type.
    - Make the wording genuinely different across the 3 paraphrases.
    - Return only structured output.
    """
).strip()


RATE_LIMIT_PATTERNS = [
    "rate limit",
    "usage limit",
    "limit reached",
    "quota",
    "too many requests",
    "try again later",
    "credit balance",
    "exceeded your",
    "daily limit",
    "message limit",
    "capacity",
]

SUCCESS_MARKER = "completed"


@dataclass
class RunStats:
    selected_tasks: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    stopped_on_limit: bool = False
    stop_reason: Optional[str] = None
    cumulative_total_tokens: int = 0


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(task_id))


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
    if isinstance(usage, dict):
        return usage
    return {}


def extract_total_tokens(usage: Dict[str, Any]) -> Optional[int]:
    if not usage:
        return None
    total = 0
    found = False
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
            found = True
    return total if found else None


def normalize_text(x: str) -> str:
    return " ".join(str(x).strip().split()).casefold()


def looks_like_rate_limit(stdout_text: str, stderr_text: str) -> Optional[str]:
    hay = f"{stdout_text}\n{stderr_text}".casefold()
    for pat in RATE_LIMIT_PATTERNS:
        if pat in hay:
            return pat
    return None


def infer_file_type_hint(task: Dict[str, Any]) -> str:
    q = str(task.get("question", "")).casefold()
    file_name = str(task.get("file_name") or "")
    file_path = str(task.get("file_path") or "")
    joined = f"{q} {file_name} {file_path}".casefold()

    hints = [
        ("spreadsheet", ["spreadsheet", ".xlsx", ".xls", ".csv", "excel"]),
        ("image", [".png", ".jpg", ".jpeg", ".gif", ".webp", "image", "photo", "picture"]),
        ("audio", [".mp3", ".wav", ".m4a", "audio"]),
        ("video", [".mp4", ".mov", ".avi", "video"]),
        ("pdf/document", [".pdf", "pdf", "document"]),
        ("presentation", [".ppt", ".pptx", "powerpoint", "slides"]),
    ]
    for label, keys in hints:
        if any(k in joined for k in keys):
            return label
    return "supporting file"


def build_prompt(task: Dict[str, Any], copies_per_task: int) -> str:
    question = str(task["question"]).strip()
    file_hint = infer_file_type_hint(task)

    parts = [
        "Create paraphrases for the following GAIA task.",
        f"Task ID: {task['task_id']}",
        f"Original question:\n{question}",
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


def explain_cli_failure(parsed_cli: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(parsed_cli, dict):
        return None

    subtype = parsed_cli.get("subtype")
    stop_reason = parsed_cli.get("stop_reason")
    terminal_reason = parsed_cli.get("terminal_reason")
    errors = parsed_cli.get("errors")

    if subtype == "error_max_turns" and stop_reason == "tool_use":
        return (
            "Claude hit max_turns before completing structured output. "
            "This usually means max_turns was set too low."
        )

    if subtype == "error_max_turns":
        return "Claude hit max_turns before returning valid structured output."

    if parsed_cli.get("is_error"):
        return (
            f"Claude CLI returned an error object: "
            f"subtype={subtype!r}, stop_reason={stop_reason!r}, "
            f"terminal_reason={terminal_reason!r}, errors={errors!r}"
        )

    return None

def validate_paraphrase_payload(
    original_question: str,
    payload: Dict[str, Any],
    copies_per_task: int,
) -> Tuple[bool, str, List[str]]:
    if not isinstance(payload, dict):
        return False, "parsed structured output is not a dict", []

    paraphrases = payload.get("paraphrases")
    checks = payload.get("checks")

    if not isinstance(paraphrases, list) or len(paraphrases) != copies_per_task:
        return False, f"expected exactly {copies_per_task} paraphrases", []

    if not isinstance(checks, dict):
        return False, "missing checks object", []

    required_checks = [
        "answer_should_remain_same",
        "constraints_preserved",
        "answer_format_preserved",
        "file_reference_preserved",
    ]
    for k in required_checks:
        if checks.get(k) is not True:
            return False, f"check {k!r} is not true", []

    original_norm = normalize_text(original_question)
    questions: List[str] = []

    for item in paraphrases:
        if not isinstance(item, dict):
            return False, "paraphrase item is not a dict", []
        q = str(item.get("question", "")).strip()
        if not q:
            return False, "empty paraphrase", []
        questions.append(q)

    norms = [normalize_text(q) for q in questions]
    if len(set(norms)) != len(norms):
        return False, "duplicate paraphrases detected", []
    if any(n == original_norm for n in norms):
        return False, "a paraphrase is identical to the original question", []

    return True, "ok", questions


def build_claude_cmd(
    prompt: str,
    model: str,
    effort: str,
    max_turns: int,
    session_id: str,
    task_name: str,
) -> List[str]:
    return [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(PARAPHRASE_SCHEMA, separators=(",", ":")),
        "--append-system-prompt",
        SYSTEM_PROMPT,
        "--session-id",
        session_id,
        "--name",
        task_name,
        "--max-turns",
        str(max_turns),
        "--model",
        model,
        "--effort",
        effort,
    ]


def run_one_task(
    task: Dict[str, Any],
    args: argparse.Namespace,
    task_out_dir: Path,
) -> Dict[str, Any]:
    prompt = build_prompt(task, copies_per_task=args.copies_per_task)
    write_text(task_out_dir / "prompt.txt", prompt)
    write_json(task_out_dir / "task_record.json", task)

    session_id = str(uuid.uuid4())
    cmd = build_claude_cmd(
        prompt=prompt,
        model=args.model,
        effort=args.effort,
        max_turns=args.max_turns,
        session_id=session_id,
        task_name=str(task["task_id"]),
    )
    write_json(task_out_dir / "cli_command.json", {"cmd": cmd})

    started_at = utcnow()
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=args.timeout_sec,
    )
    ended_at = utcnow()
    duration_ms = int((time.time() - t0) * 1000)

    write_text(task_out_dir / "claude_stdout.txt", proc.stdout)
    write_text(task_out_dir / "claude_stderr.txt", proc.stderr)

    parsed_cli = parse_cli_json(proc.stdout)
    if parsed_cli is not None:
        write_json(task_out_dir / "claude_output.json", parsed_cli)

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
    paraphrase_questions: List[str] = []

    if isinstance(structured_output, dict):
        ok, validation_msg, paraphrase_questions = validate_paraphrase_payload(
            original_question=str(task["question"]),
            payload=structured_output,
            copies_per_task=args.copies_per_task,
        )
    elif cli_failure_msg is not None:
        validation_msg = cli_failure_msg

    usage = extract_usage(parsed_cli)
    total_tokens = extract_total_tokens(usage)

    quota_pattern = looks_like_rate_limit(proc.stdout, proc.stderr)

    structured_output = None
    if isinstance(parsed_cli, dict):
        structured_output = parsed_cli.get("structured_output")
        if structured_output is None and "paraphrases" in parsed_cli:
            structured_output = parsed_cli

    ok = False
    validation_msg = "no structured output"
    paraphrase_questions: List[str] = []

    if isinstance(structured_output, dict):
        ok, validation_msg, paraphrase_questions = validate_paraphrase_payload(
            original_question=str(task["question"]),
            payload=structured_output,
            copies_per_task=args.copies_per_task,
        )

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
        "paraphrase_questions": paraphrase_questions,
        "raw_paths": {
            "prompt": str(task_out_dir / "prompt.txt"),
            "stdout": str(task_out_dir / "claude_stdout.txt"),
            "stderr": str(task_out_dir / "claude_stderr.txt"),
            "parsed_output_json": str(task_out_dir / "claude_output.json") if parsed_cli else None,
        },
    }

    write_json(task_out_dir / "result.json", result)

    if ok:
        final_payload = {
            "status": SUCCESS_MARKER,
            "query_id": task["task_id"],
            "source_task": task,
            "paraphrase_questions": paraphrase_questions,
            "usage": usage,
            "total_tokens": total_tokens,
            "model_requested": args.model,
            "effort": args.effort,
            "generated_at": ended_at,
        }
        write_json(task_out_dir / "paraphrases.json", final_payload)

    return result


def materialize_outputs(
    tasks: List[Dict[str, Any]],
    run_root: Path,
    copies_per_task: int,
) -> Tuple[int, int]:
    paraphrase_only_path = run_root / "paraphrases_only.partial.jsonl"
    expanded_path = run_root / "expanded_with_originals.partial.jsonl"

    paraphrase_only_path.parent.mkdir(parents=True, exist_ok=True)

    n_orig = 0
    n_para = 0

    with paraphrase_only_path.open("w", encoding="utf-8") as f_para, \
         expanded_path.open("w", encoding="utf-8") as f_exp:

        for task in tasks:
            orig = dict(task)
            orig["source_task_id"] = task["task_id"]
            orig["variant"] = "original"
            orig["is_paraphrase"] = False
            orig["paraphrase_index"] = 0
            f_exp.write(json.dumps(orig, ensure_ascii=False) + "\n")
            n_orig += 1

            task_dir = run_root / "tasks" / safe_task_id(task["task_id"])
            para_file = task_dir / "paraphrases.json"
            if not para_file.exists():
                continue

            payload = load_json(para_file)
            if payload.get("status") != SUCCESS_MARKER:
                continue

            paraphrases = payload.get("paraphrase_questions", [])
            for idx, q in enumerate(paraphrases, start=1):
                rec = dict(task)
                rec["task_id"] = f"{task['task_id']}__para{idx}"
                rec["source_task_id"] = task["task_id"]
                rec["variant"] = f"paraphrase_{idx}"
                rec["is_paraphrase"] = True
                rec["paraphrase_index"] = idx
                rec["question"] = q

                f_para.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_exp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_para += 1

    return n_orig, n_para


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to GAIA prepared JSONL or shard JSONL.")
    parser.add_argument("--output-root", type=str, default="results/gaia_paraphrases")
    parser.add_argument("--model", type=str, default="sonnet")
    parser.add_argument("--effort", type=str, default="low")
    parser.add_argument("--copies-per-task", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument(
        "--max-cumulative-tokens",
        type=int,
        default=None,
        help="Optional manual stop after this many accumulated total_tokens in a single run.",
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    if args.copies_per_task != 3:
        raise ValueError("This script currently expects --copies-per-task 3 to match the JSON schema.")

    input_path = Path(args.input).expanduser().resolve()
    tasks = load_jsonl(input_path)

    if args.task_id:
        tasks = [t for t in tasks if str(t.get("task_id")) == str(args.task_id)]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if not tasks:
        print("No tasks selected.")
        return

    run_name = input_path.stem
    run_root = Path(args.output_root).expanduser().resolve() / run_name
    tasks_root = run_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    stats = RunStats(selected_tasks=len(tasks))
    started_at = utcnow()

    for idx, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        task_dir = tasks_root / safe_task_id(task_id)
        para_path = task_dir / "paraphrases.json"

        if para_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(tasks)}] skip {task_id} (already done)")
            stats.skipped += 1
            continue

        print(f"[{idx}/{len(tasks)}] run {task_id}")
        task_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_one_task(task, args, task_dir)
        except subprocess.TimeoutExpired as exc:
            fail_obj = {
                "status": "failed",
                "query_id": task_id,
                "failure_type": "timeout",
                "detail": str(exc),
                "generated_at": utcnow(),
            }
            write_json(task_dir / "result.json", fail_obj)
            print("  -> timeout")
            stats.failed += 1
            continue
        except Exception as exc:
            fail_obj = {
                "status": "failed",
                "query_id": task_id,
                "failure_type": "exception",
                "detail": repr(exc),
                "generated_at": utcnow(),
            }
            write_json(task_dir / "result.json", fail_obj)
            print(f"  -> exception: {exc!r}")
            stats.failed += 1
            continue

        if result.get("total_tokens") is not None:
            stats.cumulative_total_tokens += int(result["total_tokens"])

        quota_pattern = result.get("quota_pattern")
        if quota_pattern:
            print(f"  -> stop: suspected usage/rate limit ({quota_pattern})")
            stats.stopped_on_limit = True
            stats.stop_reason = f"suspected limit pattern: {quota_pattern}"
            break

        if result.get("structured_output_valid"):
            print("  -> success")
            stats.completed += 1
        else:
            print(f"  -> invalid structured output ({result.get('validation_msg')})")
            stats.failed += 1

        # rebuild partial outputs after every successful/attempted task
        n_orig, n_para = materialize_outputs(tasks, run_root, args.copies_per_task)
        print(f"  -> materialized partial benchmark: originals={n_orig}, paraphrases={n_para}, total={n_orig + n_para}")

        if args.max_cumulative_tokens is not None and stats.cumulative_total_tokens >= args.max_cumulative_tokens:
            stats.stopped_on_limit = True
            stats.stop_reason = f"manual max cumulative tokens reached: {args.max_cumulative_tokens}"
            print(f"  -> stop: {stats.stop_reason}")
            break

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    # final materialization
    n_orig, n_para = materialize_outputs(tasks, run_root, args.copies_per_task)

    summary = {
        "input": str(input_path),
        "run_root": str(run_root),
        "started_at": started_at,
        "ended_at": utcnow(),
        "selected_tasks": stats.selected_tasks,
        "completed": stats.completed,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "stopped_on_limit": stats.stopped_on_limit,
        "stop_reason": stats.stop_reason,
        "cumulative_total_tokens_this_run": stats.cumulative_total_tokens,
        "materialized_original_records": n_orig,
        "materialized_paraphrase_records": n_para,
        "materialized_total_records": n_orig + n_para,
        "paraphrases_only_path": str(run_root / "paraphrases_only.partial.jsonl"),
        "expanded_with_originals_path": str(run_root / "expanded_with_originals.partial.jsonl"),
    }
    write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
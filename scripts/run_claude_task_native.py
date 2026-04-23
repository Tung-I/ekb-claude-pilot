#!/usr/bin/env python3
"""
Run GAIA shard tasks with native Claude Code tools and collect rich traces.

What it does
------------
1) Writes a small hook logger to scripts/_claude_trace_hook.py
2) Writes extra Claude settings to configs/claude/ekb_trace_settings.json
3) Runs `claude -p` once per task (non-interactive)
4) Saves:
   - raw CLI stdout/stderr
   - hook event JSONL
   - normalized trace JSON
   - a shard-level results.jsonl
   - a copy of the matching ~/.claude/projects/*.jsonl session file when found

This script intentionally does NOT use --bare because:
- you are authenticating with Claude Code OAuth / Pro token
- you want project hooks and CLAUDE.md loaded

Recommended first run:
  python scripts/run_claude_task_native.py --install-only

Dry run:
  python scripts/run_claude_task_native.py \
    --shard data/gaia/shards/gaia_level1_shard_00.jsonl \
    --limit 2

Full shard:
  python scripts/run_claude_task_native.py \
    --shard data/gaia/shards/gaia_level1_shard_00.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# -----------------------------
# Embedded hook script
# -----------------------------
HOOK_SCRIPT = r'''#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def main():
    trace_dir = os.environ.get("EKB_TRACE_DIR")
    if not trace_dir:
        return 0

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {"_raw_stdin": raw}

    record = {
        "hook_event": os.environ.get("EKB_HOOK_EVENT", "unknown"),
        "logged_at": utcnow(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "task_id": os.environ.get("EKB_TASK_ID"),
        "session_id_env": os.environ.get("EKB_SESSION_ID"),
        "payload": payload,
    }

    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hook_events.jsonl"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


# -----------------------------
# Small constants
# -----------------------------
FINAL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "brief_explanation": {"type": "string"},
    },
    "required": ["final_answer", "confidence", "brief_explanation"],
    "additionalProperties": False,
}

DEFAULT_ALLOWED_TOOLS = ",".join(
    [
        "Read",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "Bash(python *)",
        "Bash(python3 *)",
        "Bash(cat *)",
        "Bash(head *)",
        "Bash(tail *)",
        "Bash(ls *)",
        "Bash(find *)",
        "Bash(grep *)",
        "Bash(sed *)",
        "Bash(awk *)",
        "Bash(file *)",
        "Bash(unzip *)",
        "Bash(jq *)",
        "Bash(date *)",
    ]
)

APPEND_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are running a benchmark task. Work efficiently.

    Rules:
    - Solve exactly one GAIA task.
    - Prefer native WebSearch and WebFetch for web research.
    - Use Bash only for lightweight read-only inspection or simple calculations.
    - Do not modify repository files.
    - Do not create or edit files unless absolutely unavoidable.
    - Stop once you have enough evidence for the final answer.
    - The final answer should be concise and match the benchmark's expected answer format.
    - Return the final answer through the structured output only.
    """
).strip()


# -----------------------------
# Utility helpers
# -----------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_id)


def parse_cli_json(stdout_text: str) -> Optional[Dict[str, Any]]:
    stdout_text = stdout_text.strip()
    if not stdout_text:
        return None

    # Best case: whole stdout is one JSON object.
    try:
        obj = json.loads(stdout_text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: sometimes people wrap or prepend lines; try the last valid JSON line.
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


def build_prompt(task: Dict[str, Any]) -> str:
    question = task["question"].strip()
    parts = [
        "Solve the following GAIA validation task.",
        f"Task ID: {task['task_id']}",
        f"Question:\n{question}",
    ]

    file_path = task.get("file_path")
    file_name = task.get("file_name")
    if file_path:
        parts.append(f"A supporting file is available at this path:\n{file_path}")
    elif file_name:
        parts.append(
            f"The dataset record mentions a supporting file named '{file_name}', "
            "but no local absolute file_path is available in this prepared JSONL."
        )

    parts.append(
        textwrap.dedent(
            """
            Guidance:
            - Use tools only when needed.
            - Prefer WebSearch and WebFetch for factual lookup.
            - Use Read if a supporting file is available.
            - Use Bash only for lightweight read-only inspection or simple calculations.
            - Do not modify files.
            - When you are confident, provide the final answer through the structured output.
            """
        ).strip()
    )

    return "\n\n".join(parts)


def install_hook_assets(ekb_root: Path) -> Tuple[Path, Path]:
    """
    Writes:
      scripts/_claude_trace_hook.py
      configs/claude/ekb_trace_settings.json
    """
    hook_path = ekb_root / "scripts" / "_claude_trace_hook.py"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    os.chmod(hook_path, 0o755)

    hook_cmd = 'python3 "$CLAUDE_PROJECT_DIR"/scripts/_claude_trace_hook.py'
    settings_obj = {
        "hooks": {
            "SessionStart": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=SessionStart {hook_cmd}'}]}
            ],
            "UserPromptSubmit": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=UserPromptSubmit {hook_cmd}'}]}
            ],
            "PreToolUse": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=PreToolUse {hook_cmd}'}]}
            ],
            "PostToolUse": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=PostToolUse {hook_cmd}'}]}
            ],
            "PostToolUseFailure": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=PostToolUseFailure {hook_cmd}'}]}
            ],
            "Stop": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=Stop {hook_cmd}'}]}
            ],
            "StopFailure": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=StopFailure {hook_cmd}'}]}
            ],
            "SessionEnd": [
                {"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=SessionEnd {hook_cmd}'}]}
            ],
        }
    }

    settings_path = ekb_root / "configs" / "claude" / "ekb_trace_settings.json"
    write_json(settings_path, settings_obj)
    return hook_path, settings_path


def extract_usage(parsed_cli: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not parsed_cli:
        return {}

    usage = parsed_cli.get("usage")
    if isinstance(usage, dict):
        return usage

    # Fallback: keep common known metadata if present.
    out: Dict[str, Any] = {}
    for key in (
        "cost_usd",
        "cost",
        "duration_ms",
        "duration",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        if key in parsed_cli:
            out[key] = parsed_cli[key]
    return out


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


def load_hook_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events = load_jsonl(path)
    events.sort(key=lambda x: x.get("logged_at", ""))
    return events


def tool_kind(tool_name: str) -> str:
    mapping = {
        "WebSearch": "web_search",
        "WebFetch": "web_fetch",
        "Bash": "bash_command",
        "Read": "file_read",
        "Glob": "file_glob",
        "Grep": "file_grep",
        "Write": "file_write",
        "Edit": "file_edit",
    }
    return mapping.get(tool_name, tool_name.lower() if tool_name else "unknown")


def short_detail(tool_name: str, tool_input: Any) -> Any:
    if not isinstance(tool_input, dict):
        return tool_input

    preferred_keys = [
        "command",
        "query",
        "url",
        "file_path",
        "pattern",
        "path",
        "paths",
    ]
    detail: Dict[str, Any] = {}
    for key in preferred_keys:
        if key in tool_input:
            detail[key] = tool_input[key]

    if detail:
        return detail

    # Fallback: keep the full object if it is already small, else a truncated repr.
    text = json.dumps(tool_input, ensure_ascii=False)
    if len(text) <= 400:
        return tool_input
    return text[:400] + "...<truncated>"


def payload_tool_name(payload: Dict[str, Any]) -> str:
    return str(payload.get("tool_name") or payload.get("tool") or "")


def payload_tool_input(payload: Dict[str, Any]) -> Any:
    return payload.get("tool_input", {})


def pre_key(payload: Dict[str, Any]) -> str:
    name = payload_tool_name(payload)
    tool_input = payload_tool_input(payload)
    try:
        sig = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
    except TypeError:
        sig = repr(tool_input)
    return f"{name}::{sig}"


def pair_tool_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pair PreToolUse with PostToolUse / PostToolUseFailure as a simple FIFO matcher.
    This yields a robust first-pass normalized step list for analysis.
    """
    pres_by_exact: Dict[str, deque] = defaultdict(deque)
    pres_by_name: Dict[str, deque] = defaultdict(deque)
    steps: List[Dict[str, Any]] = []
    step_idx = 0

    for rec in events:
        hook_event = rec.get("hook_event")
        payload = rec.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"_payload": payload}

        if hook_event == "PreToolUse":
            key = pre_key(payload)
            name = payload_tool_name(payload)
            pres_by_exact[key].append(rec)
            pres_by_name[name].append(rec)
            continue

        if hook_event not in ("PostToolUse", "PostToolUseFailure"):
            continue

        name = payload_tool_name(payload)
        key = pre_key(payload)

        pre_rec = None
        if pres_by_exact[key]:
            pre_rec = pres_by_exact[key].popleft()
            # also remove one matching item from name queue
            if pres_by_name[name]:
                pres_by_name[name].popleft()
        elif pres_by_name[name]:
            pre_rec = pres_by_name[name].popleft()

        step_idx += 1
        start_ts = None
        latency_ms = None
        if pre_rec:
            start_str = pre_rec.get("logged_at")
            end_str = rec.get("logged_at")
            if start_str and end_str:
                try:
                    start_dt = datetime.fromisoformat(start_str)
                    end_dt = datetime.fromisoformat(end_str)
                    latency_ms = int((end_dt - start_dt).total_seconds() * 1000)
                    start_ts = start_str
                except Exception:
                    latency_ms = None

        tool_input = payload_tool_input(payload)
        steps.append(
            {
                "step": step_idx,
                "type": tool_kind(name),
                "tool": name,
                "action_detail": short_detail(name, tool_input),
                "status": "success" if hook_event == "PostToolUse" else "failure",
                "started_at": start_ts,
                "ended_at": rec.get("logged_at"),
                "latency_ms": latency_ms,
            }
        )

    return steps


def search_session_jsonl(session_id: str, start_unix: float) -> Optional[Path]:
    claude_config_root = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    projects_root = claude_config_root / "projects"
    if not projects_root.exists():
        return None

    candidates: List[Tuple[float, Path]] = []
    for path in projects_root.rglob("*.jsonl"):
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_mtime < start_unix - 3600:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if session_id in text:
            candidates.append((st.st_mtime, path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def exact_match(pred: Optional[str], gold: Optional[str]) -> Optional[bool]:
    if pred is None or gold is None:
        return None
    norm = lambda s: " ".join(str(s).strip().lower().split())
    return norm(pred) == norm(gold)


def run_one_task(
    task: Dict[str, Any],
    args: argparse.Namespace,
    ekb_root: Path,
    settings_path: Path,
    shard_name: str,
) -> Dict[str, Any]:
    trace_root = Path(os.environ["TRACE_ROOT"])
    result_root = Path(os.environ["RESULT_ROOT"])

    safe_id = safe_task_id(task["task_id"])
    task_trace_dir = trace_root / "claude_native" / shard_name / safe_id
    task_result_dir = result_root / "claude_native" / shard_name / safe_id
    task_trace_dir.mkdir(parents=True, exist_ok=True)
    task_result_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(task)
    write_text(task_trace_dir / "task_prompt.txt", prompt)
    write_json(task_trace_dir / "task_record.json", task)

    session_id = str(uuid.uuid4())
    started_unix = time.time()
    started_iso = utcnow()

    env = os.environ.copy()
    env["EKB_TRACE_DIR"] = str(task_trace_dir)
    env["EKB_TASK_ID"] = str(task["task_id"])
    env["EKB_SESSION_ID"] = session_id

    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(FINAL_OUTPUT_SCHEMA, separators=(",", ":")),
        "--settings",
        str(settings_path),
        "--setting-sources",
        "project,local",
        "--append-system-prompt",
        APPEND_SYSTEM_PROMPT,
        "--exclude-dynamic-system-prompt-sections",
        "--session-id",
        session_id,
        "--name",
        str(task["task_id"]),
        "--max-turns",
        str(args.max_turns),
        "--allowedTools",
        args.allowed_tools,
        "--model",
        args.model,
        "--no-chrome",
    ]
    if args.effort:
        cmd.extend(["--effort", args.effort])

    write_json(task_trace_dir / "cli_command.json", {"cmd": cmd})

    proc = subprocess.run(
        cmd,
        cwd=str(ekb_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout_sec,
    )

    ended_iso = utcnow()
    duration_ms = int((time.time() - started_unix) * 1000)

    write_text(task_trace_dir / "claude_stdout.txt", proc.stdout)
    write_text(task_trace_dir / "claude_stderr.txt", proc.stderr)

    parsed_cli = parse_cli_json(proc.stdout)
    if parsed_cli is not None:
        write_json(task_trace_dir / "claude_output.json", parsed_cli)

    usage = extract_usage(parsed_cli)
    total_tokens = extract_total_tokens(usage)
    structured_output = parsed_cli.get("structured_output", {}) if isinstance(parsed_cli, dict) else {}
    result_text = parsed_cli.get("result") if isinstance(parsed_cli, dict) else None
    returned_session_id = parsed_cli.get("session_id") if isinstance(parsed_cli, dict) else None
    if isinstance(returned_session_id, str) and returned_session_id:
        session_id = returned_session_id

    hook_events = load_hook_events(task_trace_dir / "hook_events.jsonl")
    steps = pair_tool_events(hook_events)

    session_jsonl_src = search_session_jsonl(session_id, started_unix)
    archived_session_jsonl = None
    if session_jsonl_src is not None:
        archived_path = task_trace_dir / "claude_session.jsonl"
        try:
            shutil.copy2(session_jsonl_src, archived_path)
            archived_session_jsonl = str(archived_path)
        except OSError:
            archived_session_jsonl = str(session_jsonl_src)

    final_answer_pred = None
    confidence = None
    brief_explanation = None
    if isinstance(structured_output, dict):
        final_answer_pred = structured_output.get("final_answer")
        confidence = structured_output.get("confidence")
        brief_explanation = structured_output.get("brief_explanation")

    success = proc.returncode == 0 and bool(final_answer_pred)

    normalized = {
        "query_id": task["task_id"],
        "query_text": task["question"],
        "benchmark": task.get("benchmark", "gaia"),
        "split": task.get("split"),
        "level": task.get("level"),
        "agent": "claude-code",
        "model_requested": args.model,
        "effort": args.effort,
        "session_id": session_id,
        "started_at": started_iso,
        "ended_at": ended_iso,
        "steps": steps,
        "total_steps": len(steps),
        "total_llm_calls": None,
        "total_tool_calls": len(steps),
        "total_latency_ms": duration_ms,
        "usage": usage,
        "total_tokens": total_tokens,
        "success": success,
        "tools_used": sorted({step["tool"] for step in steps if step.get("tool")}),
        "final_answer_pred": final_answer_pred,
        "confidence": confidence,
        "brief_explanation": brief_explanation,
        "result_text": result_text,
        "ground_truth_answer": task.get("final_answer"),
        "exact_match": exact_match(final_answer_pred, task.get("final_answer")),
        "raw_paths": {
            "task_prompt": str(task_trace_dir / "task_prompt.txt"),
            "hook_events": str(task_trace_dir / "hook_events.jsonl"),
            "claude_stdout": str(task_trace_dir / "claude_stdout.txt"),
            "claude_stderr": str(task_trace_dir / "claude_stderr.txt"),
            "claude_output_json": str(task_trace_dir / "claude_output.json") if parsed_cli else None,
            "claude_session_jsonl": archived_session_jsonl,
        },
        "failure": None if success else {
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
        },
    }

    write_json(task_result_dir / "normalized_trace.json", normalized)
    write_json(task_result_dir / "structured_output.json", structured_output if structured_output else {})
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, default=None, help="Path to shard JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N tasks.")
    parser.add_argument("--task-id", type=str, default=None, help="Run only one specific task_id.")
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model alias or full model name.")
    parser.add_argument("--effort", type=str, default="medium", help="low|medium|high|xhigh|max")
    parser.add_argument("--max-turns", type=int, default=12, help="Agentic turn cap per task.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Subprocess timeout per task.")
    parser.add_argument("--allowed-tools", type=str, default=DEFAULT_ALLOWED_TOOLS)
    parser.add_argument("--overwrite", action="store_true", help="Re-run tasks even if trace already exists.")
    parser.add_argument("--install-only", action="store_true", help="Only write hook assets and exit.")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    required_env = ["EKB_ROOT", "TRACE_ROOT", "RESULT_ROOT"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")

    ekb_root = Path(os.environ["EKB_ROOT"]).resolve()
    hook_path, settings_path = install_hook_assets(ekb_root)
    print(f"[setup] wrote hook: {hook_path}")
    print(f"[setup] wrote settings: {settings_path}")

    if args.install_only:
        print("[setup] install-only complete")
        return

    if not args.shard:
        raise ValueError("--shard is required unless --install-only is used.")

    shard_path = Path(args.shard).expanduser().resolve()
    tasks = load_jsonl(shard_path)
    shard_name = shard_path.stem

    if args.task_id:
        tasks = [t for t in tasks if str(t.get("task_id")) == args.task_id]

    if args.limit is not None:
        tasks = tasks[: args.limit]

    if not tasks:
        print("No tasks selected.")
        return

    result_root = Path(os.environ["RESULT_ROOT"])
    shard_results_jsonl = result_root / "claude_native" / shard_name / "results.jsonl"
    shard_summary_json = result_root / "claude_native" / shard_name / "summary.json"

    completed = 0
    skipped = 0
    failed = 0
    started_at = utcnow()

    for idx, task in enumerate(tasks, start=1):
        safe_id = safe_task_id(task["task_id"])
        normalized_trace_path = result_root / "claude_native" / shard_name / safe_id / "normalized_trace.json"

        if normalized_trace_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(tasks)}] skip {task['task_id']} (already exists)")
            skipped += 1
            continue

        print(f"[{idx}/{len(tasks)}] run {task['task_id']}")
        try:
            result = run_one_task(task, args, ekb_root, settings_path, shard_name)
            append_jsonl(shard_results_jsonl, result)
            if result.get("success"):
                completed += 1
                print(f"  -> success | pred={result.get('final_answer_pred')!r}")
            else:
                failed += 1
                print(f"  -> failed")
        except subprocess.TimeoutExpired as exc:
            failed += 1
            fail_obj = {
                "query_id": task["task_id"],
                "success": False,
                "failure": {
                    "type": "timeout",
                    "timeout_sec": args.timeout_sec,
                    "detail": str(exc),
                },
            }
            append_jsonl(shard_results_jsonl, fail_obj)
            print(f"  -> timeout")
        except Exception as exc:
            failed += 1
            fail_obj = {
                "query_id": task["task_id"],
                "success": False,
                "failure": {
                    "type": "exception",
                    "detail": repr(exc),
                },
            }
            append_jsonl(shard_results_jsonl, fail_obj)
            print(f"  -> exception: {exc!r}")

    summary = {
        "shard": str(shard_path),
        "started_at": started_at,
        "ended_at": utcnow(),
        "selected_tasks": len(tasks),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "settings_path": str(settings_path),
        "hook_path": str(hook_path),
    }
    write_json(shard_summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
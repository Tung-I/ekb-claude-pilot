#!/usr/bin/env python3
"""
Run GAIA benchmark tasks with native Claude Code tools and collect rich traces.

What it does
------------
1) Writes a small hook logger to tools/_claude_trace_hook.py
2) Writes extra Claude settings to configs/claude/ekb_trace_settings.json
3) Runs `claude -p` once per task (non-interactive)
4) Saves:
   - raw CLI stdout/stderr
   - hook event JSONL
   - normalized trace JSON
   - a run-level results.jsonl + summary.json
   - (optional) a copy of the matching ~/.claude/projects/*.jsonl session file

Resume-safe: re-running skips tasks whose normalized_trace.json already exists
(unless --overwrite is set, which resets the task directories cleanly first).

Graceful stop: detects Claude quota / rate-limit signals and halts without
corrupting already-written traces, so the next invocation can resume from
the next pending task.

Environment (set in .env or exported before running):
  EKB_ROOT    — repo root (absolute path)
  TRACE_ROOT  — where per-task trace dirs are written
  RESULT_ROOT — where normalized_trace.json / results.jsonl are written

Dry run:
  python runners/run_claude_task_native.py \\
    --input data/gaia/shards/gaia_level1_shard_00.jsonl \\
    --limit 2

Full run:
  python runners/run_claude_task_native.py \\
    --input data/gaia/shards/gaia_level1_shard_00.jsonl \\
    --run-name gaia_level1_shard_00
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
# Output schema
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
    You are running exactly one benchmark task. Work efficiently and stop early.

    Rules:
    - Solve exactly one task.
    - Prefer native WebSearch and WebFetch for factual lookup.
    - Prefer Read for directly provided local files.
    - Use Bash only for lightweight read-only inspection or simple calculations.
    - Avoid redundant searches and repeated fetches.
    - Use as few tool calls as needed to answer confidently.
    - Do not modify repository files.
    - Do not create or edit files unless absolutely unavoidable.
    - The final answer should be concise and match the benchmark's expected format.
    - Return the final answer through the structured output only.
    """
).strip()

# Patterns that indicate Claude hit a usage/rate/quota limit.
LIMIT_PATTERNS = [
    "you've hit your limit",
    "you have hit your limit",
    "hit your limit",
    "resets ",
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

LEADING_STRIP_CHARS  = "'\"\u2018\u2019\u201c\u201d`([{<"
TRAILING_STRIP_CHARS = "'\"\u2018\u2019\u201c\u201d`.,;:!?)]}>"
CURRENCY_CHARS = "$\u20ac\xa3\xa5\u20b9\u20a9\u20bd\u20aa\u0e3f\u20ab\u20b4\u20a6\u20b1\u20b2\u20b5\u20a1\u20ba\u20b8\u20bc\u20ad\u20ae\u20a8"

# Per-step tool result is stored up to this many characters; larger responses are truncated.
# Full responses remain available in hook_events.jsonl for L2 cache analysis.
TOOL_RESULT_MAX_CHARS = 10_000


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
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(task_id))


def reset_task_dir(task_trace_dir: Path) -> None:
    """Remove stale per-task trace dir before a rerun to prevent old hook events from leaking in."""
    if task_trace_dir.exists():
        shutil.rmtree(task_trace_dir)
    task_trace_dir.mkdir(parents=True, exist_ok=True)


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


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def transcribe_audio(file_path: str) -> Optional[str]:
    """Transcribe an audio file with openai-whisper. Returns transcript text or None on failure."""
    try:
        import whisper  # type: ignore
    except ImportError:
        print("[transcribe] openai-whisper not installed — skipping pre-transcription")
        return None
    try:
        print(f"[transcribe] loading whisper base model …")
        model = whisper.load_model("base")
        print(f"[transcribe] transcribing {file_path} …")
        result = model.transcribe(file_path)
        text = result.get("text", "").strip()
        print(f"[transcribe] done ({len(text)} chars)")
        return text if text else None
    except Exception as exc:
        print(f"[transcribe] failed: {exc}")
        return None


def build_prompt(task: Dict[str, Any], audio_transcript: Optional[str] = None) -> str:
    question = str(task["question"]).strip()
    parts = [
        "Solve the following benchmark task.",
        f"Task ID: {task['task_id']}",
        f"Question:\n{question}",
    ]

    file_path = task.get("file_path")
    file_name = task.get("file_name")
    if file_path:
        if audio_transcript is not None:
            parts.append(
                f"A supporting audio file was pre-transcribed for you. "
                f"Transcript of {Path(file_path).name}:\n{audio_transcript}"
            )
        else:
            parts.append(f"A supporting file is available at this path:\n{file_path}")
    elif file_name:
        parts.append(
            f"The dataset record mentions a supporting file named '{file_name}', "
            "but no local absolute file_path is available in this JSONL."
        )

    parts.append(
        textwrap.dedent(
            """
            Guidance:
            - Use tools only when needed.
            - Prefer one good WebSearch before trying many searches.
            - Prefer WebFetch / Read only for the most relevant source(s).
            - Use Bash only for lightweight read-only inspection or simple calculations.
            - Avoid redundant tool calls.
            - Do not modify files.
            - When confident, provide the final answer through the structured output.
            """
        ).strip()
    )

    return "\n\n".join(parts)


def install_hook_assets(ekb_root: Path) -> Tuple[Path, Path]:
    hook_path = ekb_root / "tools" / "_claude_trace_hook.py"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    os.chmod(hook_path, 0o755)

    hook_cmd = 'python3 "$CLAUDE_PROJECT_DIR"/tools/_claude_trace_hook.py'
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
    out: Dict[str, Any] = {}
    for key in (
        "cost_usd", "cost", "duration_ms", "duration",
        "input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens",
    ):
        if key in parsed_cli:
            out[key] = parsed_cli[key]
    return out


def extract_total_tokens(usage: Dict[str, Any]) -> Optional[int]:
    if not usage:
        return None
    total = 0
    found = False
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
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
    preferred_keys = ["command", "query", "url", "file_path", "pattern", "path", "paths"]
    detail: Dict[str, Any] = {k: tool_input[k] for k in preferred_keys if k in tool_input}
    if detail:
        return detail
    text = json.dumps(tool_input, ensure_ascii=False)
    return tool_input if len(text) <= 400 else text[:400] + "...<truncated>"


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


def truncate_tool_result(result: Any) -> Tuple[Any, bool]:
    """Return (possibly-truncated result, was_truncated). Keeps structure for small objects."""
    if result is None:
        return result, False
    if isinstance(result, str):
        if len(result) <= TOOL_RESULT_MAX_CHARS:
            return result, False
        return result[:TOOL_RESULT_MAX_CHARS] + "...<truncated>", True
    try:
        text = json.dumps(result, ensure_ascii=False)
    except TypeError:
        text = repr(result)
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return result, False
    return text[:TOOL_RESULT_MAX_CHARS] + "...<truncated>", True


def pair_tool_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pair PreToolUse with PostToolUse/PostToolUseFailure events and build step records.

    Matching strategy:
    1. By tool_use_id (reliable, Claude Code includes this in every hook payload).
    2. Fallback: FIFO by (tool_name + serialized tool_input), then by tool_name alone.

    Each step includes tool_result from tool_response (success) or error (failure),
    truncated to TOOL_RESULT_MAX_CHARS. Full results remain in hook_events.jsonl.
    """
    pres_by_id:    Dict[str, Any]   = {}
    pres_by_exact: Dict[str, deque] = defaultdict(deque)
    pres_by_name:  Dict[str, deque] = defaultdict(deque)
    steps: List[Dict[str, Any]] = []
    step_idx = 0

    for rec in events:
        hook_event = rec.get("hook_event")
        payload = rec.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"_payload": payload}

        if hook_event == "PreToolUse":
            key    = pre_key(payload)
            name   = payload_tool_name(payload)
            use_id = payload.get("tool_use_id")
            if use_id:
                pres_by_id[use_id] = rec
            pres_by_exact[key].append(rec)
            pres_by_name[name].append(rec)
            continue

        if hook_event not in ("PostToolUse", "PostToolUseFailure"):
            continue

        name   = payload_tool_name(payload)
        key    = pre_key(payload)
        use_id = payload.get("tool_use_id")

        # Primary: match by tool_use_id; keep FIFO queues tidy on hit.
        pre_rec = None
        if use_id and use_id in pres_by_id:
            pre_rec      = pres_by_id.pop(use_id)
            pre_payload  = pre_rec.get("payload", {})
            pre_key_val  = pre_key(pre_payload)
            pre_name     = payload_tool_name(pre_payload)
            if pres_by_exact[pre_key_val]:
                pres_by_exact[pre_key_val].popleft()
            if pres_by_name[pre_name]:
                pres_by_name[pre_name].popleft()
        elif pres_by_exact[key]:
            pre_rec = pres_by_exact[key].popleft()
            if pres_by_name[name]:
                pres_by_name[name].popleft()
        elif pres_by_name[name]:
            pre_rec = pres_by_name[name].popleft()

        step_idx += 1
        start_ts = latency_ms = None
        if pre_rec:
            start_str = pre_rec.get("logged_at")
            end_str   = rec.get("logged_at")
            if start_str and end_str:
                try:
                    start_dt   = datetime.fromisoformat(start_str)
                    end_dt     = datetime.fromisoformat(end_str)
                    latency_ms = int((end_dt - start_dt).total_seconds() * 1000)
                    start_ts   = start_str
                except Exception:
                    pass

        tool_input = payload_tool_input(payload)

        # tool_response on success; error (or tool_response) on failure.
        if hook_event == "PostToolUse":
            raw_result = payload.get("tool_response")
        else:
            raw_result = payload.get("error") or payload.get("tool_response")
        tool_result, was_truncated = truncate_tool_result(raw_result)

        steps.append({
            "step":                  step_idx,
            "type":                  tool_kind(name),
            "tool":                  name,
            "tool_use_id":           use_id,
            "action_detail":         short_detail(name, tool_input),
            "tool_result":           tool_result,
            "tool_result_truncated": was_truncated,
            "status":                "success" if hook_event == "PostToolUse" else "failure",
            "started_at":            start_ts,
            "ended_at":              rec.get("logged_at"),
            "latency_ms":            latency_ms,
        })

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


def _normalize_basic_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(LEADING_STRIP_CHARS + TRAILING_STRIP_CHARS)
    s = re.sub(rf"[{re.escape(TRAILING_STRIP_CHARS)}]+$", "", s).strip()
    return re.sub(r"\s+", " ", s).strip()


def _canonicalize_decimal_string(x: str) -> Optional[str]:
    s = unicodedata.normalize("NFKC", str(x)).strip()
    s = s.strip(LEADING_STRIP_CHARS + TRAILING_STRIP_CHARS).replace(" ", "")
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
    return "0" if out == "-0" else out


def exact_match(pred: Optional[str], gold: Optional[str]) -> Optional[bool]:
    if pred is None or gold is None:
        return None
    pred_num = _canonicalize_decimal_string(pred)
    gold_num = _canonicalize_decimal_string(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    return _normalize_basic_text(pred) == _normalize_basic_text(gold)


def detect_limit_signal(
    parsed_cli: Optional[Dict[str, Any]],
    stdout_text: str,
    stderr_text: str,
    result_text: Optional[str] = None,
) -> Optional[str]:
    """Return a description string if a Claude usage/rate/quota limit is detected, else None."""
    def find_pattern(text: Optional[str], prefix: str) -> Optional[str]:
        if not text:
            return None
        hay = str(text).casefold()
        for pat in LIMIT_PATTERNS:
            if pat in hay:
                return f"{prefix}:{pat}"
        return None

    hit = find_pattern(result_text, "result_text")
    if hit:
        return hit

    if isinstance(parsed_cli, dict):
        hit = find_pattern(parsed_cli.get("result"), "parsed_result")
        if hit:
            return hit

        errors = parsed_cli.get("errors")
        if isinstance(errors, list):
            for err in errors:
                hit = find_pattern(str(err), "cli_error")
                if hit:
                    return hit

        meta = " ".join(
            "" if x is None else str(x)
            for x in [parsed_cli.get("terminal_reason"), parsed_cli.get("subtype"), parsed_cli.get("stop_reason")]
        )
        hit = find_pattern(meta, "cli_meta")
        if hit:
            return hit

    # Scan stderr only — not raw stdout, which contains full conversation text
    # (tool results, message bodies) and produces false positives on words like
    # "quotation" matching the "quota" pattern.
    return find_pattern(stderr_text, "stderr")


# -----------------------------
# Per-task runner
# -----------------------------
def run_one_task(
    task: Dict[str, Any],
    args: argparse.Namespace,
    ekb_root: Path,
    settings_path: Path,
    run_name: str,
) -> Dict[str, Any]:
    trace_root = Path(os.environ["TRACE_ROOT"])

    safe_id        = safe_task_id(task["task_id"])
    task_trace_dir = trace_root / "claude_native" / run_name / safe_id

    # Always reset dir here; the skip-if-exists check is done in main().
    reset_task_dir(task_trace_dir)

    audio_transcript: Optional[str] = None
    file_path = task.get("file_path") or ""
    if Path(file_path).suffix.lower() in AUDIO_EXTENSIONS:
        audio_transcript = transcribe_audio(file_path)

    prompt = build_prompt(task, audio_transcript=audio_transcript)
    write_text(task_trace_dir / "task_prompt.txt", prompt)
    write_json(task_trace_dir / "task_record.json", task)

    session_id   = str(uuid.uuid4())
    started_unix = time.time()
    started_iso  = utcnow()

    # Build subprocess env: start from current process (which already has dotenv loaded),
    # then explicitly set the global path vars so they are always present.
    env = os.environ.copy()
    for key in ("EKB_ROOT", "TRACE_ROOT", "RESULT_ROOT", "GAIA_RAW_PATH"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    env["EKB_TRACE_DIR"]  = str(task_trace_dir)
    env["EKB_TASK_ID"]    = str(task["task_id"])
    env["EKB_SESSION_ID"] = session_id

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(FINAL_OUTPUT_SCHEMA, separators=(",", ":")),
        "--settings", str(settings_path),
        "--setting-sources", "project,local",
        "--append-system-prompt", APPEND_SYSTEM_PROMPT,
        "--exclude-dynamic-system-prompt-sections",
        "--session-id", session_id,
        "--name", str(task["task_id"]),
        "--max-turns", str(args.max_turns),
        "--allowedTools", args.allowed_tools,
        "--model", args.model,
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

    ended_iso   = utcnow()
    duration_ms = int((time.time() - started_unix) * 1000)

    write_text(task_trace_dir / "claude_stdout.txt", proc.stdout)
    write_text(task_trace_dir / "claude_stderr.txt", proc.stderr)

    parsed_cli = parse_cli_json(proc.stdout)
    if parsed_cli is not None:
        write_json(task_trace_dir / "claude_output.json", parsed_cli)

    result_text  = parsed_cli.get("result") if isinstance(parsed_cli, dict) else None
    usage        = extract_usage(parsed_cli)
    total_tokens = extract_total_tokens(usage)

    # --- Graceful stop on suspected quota/rate-limit ---
    limit_signal = detect_limit_signal(
        parsed_cli=parsed_cli,
        stdout_text=proc.stdout,
        stderr_text=proc.stderr,
        result_text=result_text,
    )
    if limit_signal is not None:
        write_json(task_trace_dir / "limit_stop.json", {
            "query_id":     task["task_id"],
            "limit_signal": limit_signal,
            "result_text":  result_text,
            "total_tokens": total_tokens,
            "started_at":   started_iso,
            "ended_at":     ended_iso,
        })
        return {
            "limit_stop":   True,
            "limit_signal": limit_signal,
            "query_id":     task["task_id"],
            "total_tokens": total_tokens,
            "trace_dir":    str(task_trace_dir),
            "result_text":  result_text,
        }

    structured_output   = parsed_cli.get("structured_output", {}) if isinstance(parsed_cli, dict) else {}
    returned_session_id = parsed_cli.get("session_id") if isinstance(parsed_cli, dict) else None
    if isinstance(returned_session_id, str) and returned_session_id:
        session_id = returned_session_id

    hook_events = load_hook_events(task_trace_dir / "hook_events.jsonl")
    steps       = pair_tool_events(hook_events)

    archived_session_jsonl = None
    if not args.disable_session_archive:
        session_jsonl_src = search_session_jsonl(session_id, started_unix)
        if session_jsonl_src is not None:
            archived_path = task_trace_dir / "claude_session.jsonl"
            try:
                shutil.copy2(session_jsonl_src, archived_path)
                archived_session_jsonl = str(archived_path)
            except OSError:
                archived_session_jsonl = str(session_jsonl_src)

    final_answer_pred = confidence = brief_explanation = None
    if isinstance(structured_output, dict):
        final_answer_pred = structured_output.get("final_answer")
        confidence        = structured_output.get("confidence")
        brief_explanation = structured_output.get("brief_explanation")

    success = proc.returncode == 0 and bool(final_answer_pred)

    normalized = {
        "query_id":           task["task_id"],
        "query_text":         task["question"],
        "benchmark":          task.get("benchmark", "gaia"),
        "split":              task.get("split"),
        "level":              task.get("level"),
        "agent":              "claude-code",
        "model_requested":    args.model,
        "effort":             args.effort,
        "session_id":         session_id,
        "started_at":         started_iso,
        "ended_at":           ended_iso,
        "steps":              steps,
        "total_steps":        len(steps),
        "total_llm_calls":    None,
        "total_tool_calls":   len(steps),
        "total_latency_ms":   duration_ms,
        "usage":              usage,
        "total_tokens":       total_tokens,
        "success":            success,
        "tools_used":         sorted({s["tool"] for s in steps if s.get("tool")}),
        "final_answer_pred":  final_answer_pred,
        "confidence":         confidence,
        "brief_explanation":  brief_explanation,
        "result_text":        result_text,
        "ground_truth_answer": task.get("final_answer"),
        "exact_match":        exact_match(final_answer_pred, task.get("final_answer")),
        "raw_paths": {
            "task_prompt":        str(task_trace_dir / "task_prompt.txt"),
            "hook_events":        str(task_trace_dir / "hook_events.jsonl"),
            "claude_stdout":      str(task_trace_dir / "claude_stdout.txt"),
            "claude_stderr":      str(task_trace_dir / "claude_stderr.txt"),
            "claude_output_json": str(task_trace_dir / "claude_output.json") if parsed_cli else None,
            "claude_session_jsonl": archived_session_jsonl,
        },
        "failure": None if success else {
            "returncode":  proc.returncode,
            "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
        },
    }

    write_json(task_trace_dir / "normalized_trace.json", normalized)
    write_json(task_trace_dir / "structured_output.json", structured_output or {})
    return normalized


# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GAIA benchmark tasks with Claude Code and collect traces."
    )
    parser.add_argument("--input",       type=str, default=None,
                        help="Path to benchmark JSONL.")
    parser.add_argument("--run-name",    type=str, default=None,
                        help="Run name used for output directory. Defaults to input stem.")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Process at most N tasks.")
    parser.add_argument("--task-id",     type=str, default=None,
                        help="Run only one specific task_id.")
    parser.add_argument("--model",       type=str, default="sonnet",
                        help="Claude model alias or full model name.")
    parser.add_argument("--effort",      type=str, default="medium",
                        help="low|medium|high|xhigh|max")
    parser.add_argument("--max-turns",   type=int, default=32,
                        help="Agentic turn cap per task.")
    parser.add_argument("--timeout-sec", type=int, default=300,
                        help="Subprocess timeout per task (seconds).")
    parser.add_argument("--allowed-tools", type=str, default=DEFAULT_ALLOWED_TOOLS)
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-run tasks even if trace already exists.")
    parser.add_argument("--install-only", action="store_true",
                        help="Only write hook assets and exit.")
    parser.add_argument("--sleep-sec",   type=float, default=1.0,
                        help="Sleep between tasks (seconds).")
    parser.add_argument("--max-cumulative-tokens", type=int, default=None,
                        help="Halt after this many cumulative tokens in one run.")
    parser.add_argument("--disable-session-archive", action="store_true",
                        help="Skip copying ~/.claude/projects/*.jsonl session files.")
    args = parser.parse_args()

    required_env = ["EKB_ROOT", "TRACE_ROOT", "RESULT_ROOT"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {missing}. "
            "Set them in .env or export them before running."
        )

    ekb_root = Path(os.environ["EKB_ROOT"]).resolve()
    hook_path, settings_path = install_hook_assets(ekb_root)
    print(f"[setup] hook:     {hook_path}")
    print(f"[setup] settings: {settings_path}")

    if args.install_only:
        print("[setup] install-only complete")
        return

    if not args.input:
        raise ValueError("--input is required unless --install-only is used.")

    input_path = Path(args.input).expanduser().resolve()
    tasks      = load_jsonl(input_path)
    run_name   = args.run_name or input_path.stem

    if args.task_id:
        tasks = [t for t in tasks if str(t.get("task_id")) == args.task_id]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("No tasks selected.")
        return

    trace_root_main    = Path(os.environ["TRACE_ROOT"])
    result_root        = Path(os.environ["RESULT_ROOT"])
    run_results_jsonl  = result_root / "claude_native" / run_name / "results.jsonl"
    run_summary_json   = result_root / "claude_native" / run_name / "summary.json"

    completed = skipped = failed = 0
    stopped_on_limit      = False
    stop_reason           = None
    cumulative_tokens     = 0
    started_at            = utcnow()

    for idx, task in enumerate(tasks, start=1):
        safe_id          = safe_task_id(task["task_id"])
        normalized_trace = trace_root_main / "claude_native" / run_name / safe_id / "normalized_trace.json"

        if normalized_trace.exists() and not args.overwrite:
            print(f"[{idx}/{len(tasks)}] skip {task['task_id']} (already exists)")
            skipped += 1
            continue

        print(f"[{idx}/{len(tasks)}] run {task['task_id']}")
        try:
            result = run_one_task(task, args, ekb_root, settings_path, run_name)

            if result.get("limit_stop"):
                stopped_on_limit = True
                stop_reason = f"suspected usage/rate limit: {result.get('limit_signal')}"
                if result.get("total_tokens") is not None:
                    cumulative_tokens += int(result["total_tokens"])
                print(f"  -> stop: {stop_reason}")
                break

            append_jsonl(run_results_jsonl, result)

            if result.get("total_tokens") is not None:
                cumulative_tokens += int(result["total_tokens"])

            if result.get("success"):
                completed += 1
                print(f"  -> success | pred={result.get('final_answer_pred')!r}")
            else:
                failed += 1
                print("  -> failed")

            if args.max_cumulative_tokens is not None and cumulative_tokens >= args.max_cumulative_tokens:
                stopped_on_limit = True
                stop_reason = f"manual token cap reached: {args.max_cumulative_tokens}"
                print(f"  -> stop: {stop_reason}")
                break

        except subprocess.TimeoutExpired as exc:
            failed += 1
            append_jsonl(run_results_jsonl, {
                "query_id": task["task_id"],
                "success": False,
                "failure": {"type": "timeout", "timeout_sec": args.timeout_sec, "detail": str(exc)},
            })
            print("  -> timeout")
        except Exception as exc:
            failed += 1
            append_jsonl(run_results_jsonl, {
                "query_id": task["task_id"],
                "success": False,
                "failure": {"type": "exception", "detail": repr(exc)},
            })
            print(f"  -> exception: {exc!r}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    summary = {
        "input":                          str(input_path),
        "run_name":                       run_name,
        "started_at":                     started_at,
        "ended_at":                       utcnow(),
        "selected_tasks":                 len(tasks),
        "completed":                      completed,
        "failed":                         failed,
        "skipped":                        skipped,
        "stopped_on_limit":               stopped_on_limit,
        "stop_reason":                    stop_reason,
        "cumulative_total_tokens_this_run": cumulative_tokens,
        "settings_path":                  str(settings_path),
        "hook_path":                      str(hook_path),
    }
    write_json(run_summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

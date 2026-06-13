#!/usr/bin/env python3
"""
Run GAIA test tasks with execution-plan reuse from the knowledge base.

For each test task the script looks up traces/plan-caching-study/test/<task_id>/3nn_meta.json,
selects the best knowledge-base neighbor that passes the similarity threshold, and
instructs the agent to follow that neighbor's tool sequence exactly (cache hit).
If no eligible neighbor exists the agent plans freely (cache miss).

The "best" neighbor is chosen by sorting eligible neighbors (similarity >= --min-similarity)
on the metric given by --plan-rank-by (lower is better, default: total_tokens).

The output traces are structurally identical to those produced by
run_claude_task_native_resume_fixed.py with three extra fields added to
normalized_trace.json:
    cache_hit              bool
    cache_source_task_id   str | null
    cache_source_similarity float | null
    cached_tool_sequence   list[str] | null
    plan_rank_by           str
    min_similarity_threshold float

Env vars required (same as the original runner):
    EKB_ROOT    – repo root
    TRACE_ROOT  – where raw per-task trace files are written
    RESULT_ROOT – where normalized_trace.json is written

Dry run:
python runners/run_claude_task_w_plan_reuse.py \\
  --input data/gaia/gaia_lv1.jsonl \\
  --run-name gaia_test_plan_reuse_dryrun \\
  --limit 2 \\
  --model sonnet \\
  --effort medium \\
  --max-turns 16 \\
  --disable-session-archive

Full run:
python runners/run_claude_task_w_plan_reuse.py \
  --input data/gaia/gaia_lv1.jsonl \
  --run-name gaia_test_plan_reuse_rank_token \
  --model sonnet \
  --effort medium \
  --max-turns 16 \
  --plan-rank-by total_tokens \
  --min-similarity 0.8 \
  --disable-session-archive
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
import time
import uuid

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

LEADING_STRIP_CHARS = "\"'`“”‘’([{<"
TRAILING_STRIP_CHARS = "\"'`“”‘’.,;:!?)]}>"
CURRENCY_CHARS = "$€\xa3\xa5₹₩₽₪฿₫₴₦₱₲₵₡₺₸₼₭₮₨"

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
        "Bash(ls *)",
        "Bash(find *)",
        "Bash(grep *)",
        "Bash(file *)",
        "Bash(unzip *)",
        "Bash(jq *)",
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

# Tools that are infrastructure overhead, not part of the semantic plan.
# Filtered out when extracting the plan template from a KB trace.
_INFRA_TOOLS = {"ToolSearch"}


# -----------------------------
# Utility helpers (unchanged from original runner)
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


def reset_task_dirs(task_trace_dir: Path, task_result_dir: Path) -> None:
    if task_trace_dir.exists():
        shutil.rmtree(task_trace_dir)
    if task_result_dir.exists():
        shutil.rmtree(task_result_dir)
    task_trace_dir.mkdir(parents=True, exist_ok=True)
    task_result_dir.mkdir(parents=True, exist_ok=True)


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


def install_hook_assets(ekb_root: Path) -> Tuple[Path, Path]:
    hook_path = ekb_root / "tools" / "_claude_trace_hook.py"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    os.chmod(hook_path, 0o755)

    hook_cmd = 'python3 "$CLAUDE_PROJECT_DIR"/tools/_claude_trace_hook.py'
    settings_obj = {
        "hooks": {
            "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=SessionStart {hook_cmd}'}]}],
            "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=UserPromptSubmit {hook_cmd}'}]}],
            "PreToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=PreToolUse {hook_cmd}'}]}],
            "PostToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=PostToolUse {hook_cmd}'}]}],
            "PostToolUseFailure": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=PostToolUseFailure {hook_cmd}'}]}],
            "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=Stop {hook_cmd}'}]}],
            "StopFailure": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=StopFailure {hook_cmd}'}]}],
            "SessionEnd": [{"matcher": "", "hooks": [{"type": "command", "command": f'EKB_HOOK_EVENT=SessionEnd {hook_cmd}'}]}],
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
    for key in ("cost_usd", "cost", "duration_ms", "duration", "input_tokens",
                "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
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
    detail: Dict[str, Any] = {}
    for key in preferred_keys:
        if key in tool_input:
            detail[key] = tool_input[key]
    if detail:
        return detail
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
        steps.append({
            "step": step_idx,
            "type": tool_kind(name),
            "tool": name,
            "action_detail": short_detail(name, tool_input),
            "status": "success" if hook_event == "PostToolUse" else "failure",
            "started_at": start_ts,
            "ended_at": rec.get("logged_at"),
            "latency_ms": latency_ms,
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
    return _normalize_basic_text(pred) == _normalize_basic_text(gold)


def detect_limit_signal(
    parsed_cli: Optional[Dict[str, Any]],
    stdout_text: str,
    stderr_text: str,
    result_text: Optional[str] = None,
) -> Optional[str]:
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
        meta_fields = [parsed_cli.get("terminal_reason"), parsed_cli.get("subtype"), parsed_cli.get("stop_reason")]
        hit = find_pattern(" ".join("" if x is None else str(x) for x in meta_fields), "cli_meta")
        if hit:
            return hit
    hit = find_pattern(stdout_text, "stdout")
    if hit:
        return hit
    return find_pattern(stderr_text, "stderr")


# -----------------------------
# Plan-cache logic (new)
# -----------------------------

def lookup_plan_cache(
    task_id: str,
    plan_cache_dir: Path,
    rank_by: str,
    min_similarity: float,
) -> Optional[Dict[str, Any]]:
    """
    Returns a plan-cache hit dict or None (cache miss).

    Hit dict keys:
        source_task_id     – KB task ID whose trace was selected
        similarity         – cosine similarity between test query and KB query
        tool_sequence      – ordered list of tool names (ToolSearch stripped)
        rank_by            – metric used for selection
        rank_by_value      – value of that metric for the selected KB task
    """
    nn_path = plan_cache_dir / "test" / task_id / "3nn_meta.json"
    if not nn_path.exists():
        return None

    nn_data = json.loads(nn_path.read_text(encoding="utf-8"))
    neighbors: List[Dict[str, Any]] = nn_data.get("neighbors", [])

    # Filter by similarity threshold
    eligible = [nb for nb in neighbors if (nb.get("similarity") or 0.0) >= min_similarity]
    if not eligible:
        return None

    # Sort ascending by rank_by metric (lower is better); push None values last
    eligible.sort(key=lambda nb: nb.get(rank_by) if nb.get(rank_by) is not None else float("inf"))
    best = eligible[0]

    # Load the KB trace to extract the ordered tool sequence
    kb_trace_path = plan_cache_dir / "knowledge-base" / best["task_id"] / "normalized_trace.json"
    if not kb_trace_path.exists():
        return None

    kb_trace = json.loads(kb_trace_path.read_text(encoding="utf-8"))
    steps: List[Dict[str, Any]] = kb_trace.get("steps", [])

    tool_sequence = [
        step["tool"]
        for step in steps
        if step.get("tool") and step["tool"] not in _INFRA_TOOLS
    ]
    if not tool_sequence:
        return None

    return {
        "source_task_id": best["task_id"],
        "similarity":     best.get("similarity"),
        "tool_sequence":  tool_sequence,
        "rank_by":        rank_by,
        "rank_by_value":  best.get(rank_by),
    }


def format_plan_context(tool_sequence: List[str]) -> str:
    steps_text = "\n".join(f"  {i + 1}. {tool}" for i, tool in enumerate(tool_sequence))
    return textwrap.dedent(f"""
        Execution Plan (retrieved from knowledge base — follow strictly):
        A semantically similar task was previously solved efficiently using this exact tool sequence:
        {steps_text}

        You MUST execute these tools in this exact order.
        Do not use any tools not listed above, do not skip steps, and do not add extra steps.
        Adhering to this plan avoids redundant planning and biases execution toward the most efficient known trajectory.
    """).strip()


# -----------------------------
# Prompt builder (extended)
# -----------------------------

def build_prompt(task: Dict[str, Any], plan_context: Optional[str] = None) -> str:
    question = str(task["question"]).strip()
    parts = [
        "Solve the following benchmark task.",
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

    if plan_context:
        parts.append(plan_context)

    return "\n\n".join(parts)


# -----------------------------
# Task runner (extended)
# -----------------------------

def run_one_task(
    task: Dict[str, Any],
    args: argparse.Namespace,
    ekb_root: Path,
    settings_path: Path,
    run_name: str,
    plan_cache_dir: Path,
) -> Dict[str, Any]:
    trace_root = Path(os.environ["TRACE_ROOT"])
    result_root = Path(os.environ["RESULT_ROOT"])

    safe_id = safe_task_id(task["task_id"])
    task_trace_dir  = trace_root  / "claude_native" / run_name / safe_id
    task_result_dir = result_root / "claude_native" / run_name / safe_id
    reset_task_dirs(task_trace_dir, task_result_dir)

    # ------------------------------------------------------------------
    # Plan-cache lookup
    # ------------------------------------------------------------------
    plan_hit = lookup_plan_cache(
        task_id       = task["task_id"],
        plan_cache_dir= plan_cache_dir,
        rank_by       = args.plan_rank_by,
        min_similarity= args.min_similarity,
    )

    if plan_hit:
        plan_context = format_plan_context(plan_hit["tool_sequence"])
        print(f"  -> cache HIT  src={plan_hit['source_task_id']}  "
              f"sim={plan_hit['similarity']:.3f}  "
              f"{args.plan_rank_by}={plan_hit['rank_by_value']}")
    else:
        plan_context = None
        print("  -> cache MISS (no eligible neighbor)")

    prompt = build_prompt(task, plan_context=plan_context)
    write_text(task_trace_dir / "task_prompt.txt", prompt)
    write_json(task_trace_dir / "task_record.json", task)
    if plan_hit:
        write_json(task_trace_dir / "plan_hit.json", plan_hit)

    session_id   = str(uuid.uuid4())
    started_unix = time.time()
    started_iso  = utcnow()

    env = os.environ.copy()
    env["EKB_TRACE_DIR"] = str(task_trace_dir)
    env["EKB_TASK_ID"]   = str(task["task_id"])
    env["EKB_SESSION_ID"] = session_id

    cmd = [
        "claude",
        "-p", prompt,
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

    limit_signal = detect_limit_signal(
        parsed_cli=parsed_cli,
        stdout_text=proc.stdout,
        stderr_text=proc.stderr,
        result_text=result_text,
    )

    if limit_signal is not None:
        write_json(task_result_dir / "limit_stop.json", {
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

    structured_output = parsed_cli.get("structured_output", {}) if isinstance(parsed_cli, dict) else {}

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

    final_answer_pred = None
    confidence        = None
    brief_explanation = None
    if isinstance(structured_output, dict):
        final_answer_pred = structured_output.get("final_answer")
        confidence        = structured_output.get("confidence")
        brief_explanation = structured_output.get("brief_explanation")

    success = proc.returncode == 0 and bool(final_answer_pred)

    normalized = {
        "query_id":        task["task_id"],
        "query_text":      task["question"],
        "benchmark":       task.get("benchmark", "gaia"),
        "split":           task.get("split"),
        "level":           task.get("level"),
        "agent":           "claude-code",
        "model_requested": args.model,
        "effort":          args.effort,
        "session_id":      session_id,
        "started_at":      started_iso,
        "ended_at":        ended_iso,
        "steps":           steps,
        "total_steps":     len(steps),
        "total_llm_calls": None,
        "total_tool_calls": len(steps),
        "total_latency_ms": duration_ms,
        "usage":           usage,
        "total_tokens":    total_tokens,
        "success":         success,
        "tools_used":      sorted({step["tool"] for step in steps if step.get("tool")}),
        "final_answer_pred":  final_answer_pred,
        "confidence":         confidence,
        "brief_explanation":  brief_explanation,
        "result_text":        result_text,
        "ground_truth_answer": task.get("final_answer"),
        "exact_match":        exact_match(final_answer_pred, task.get("final_answer")),
        # ------------------------------------------------------------------
        # Plan-cache metadata (new fields)
        # ------------------------------------------------------------------
        "cache_hit":                 plan_hit is not None,
        "cache_source_task_id":      plan_hit["source_task_id"]  if plan_hit else None,
        "cache_source_similarity":   plan_hit["similarity"]      if plan_hit else None,
        "cached_tool_sequence":      plan_hit["tool_sequence"]   if plan_hit else None,
        "plan_rank_by":              args.plan_rank_by,
        "min_similarity_threshold":  args.min_similarity,
        # ------------------------------------------------------------------
        "raw_paths": {
            "task_prompt":         str(task_trace_dir / "task_prompt.txt"),
            "hook_events":         str(task_trace_dir / "hook_events.jsonl"),
            "claude_stdout":       str(task_trace_dir / "claude_stdout.txt"),
            "claude_stderr":       str(task_trace_dir / "claude_stderr.txt"),
            "claude_output_json":  str(task_trace_dir / "claude_output.json") if parsed_cli else None,
            "claude_session_jsonl": archived_session_jsonl,
        },
        "failure": None if success else {
            "returncode":  proc.returncode,
            "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
        },
    }

    write_json(task_result_dir / "normalized_trace.json", normalized)
    write_json(task_result_dir / "structured_output.json", structured_output if structured_output else {})
    return normalized


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GAIA test tasks with execution-plan reuse from the knowledge base."
    )
    parser.add_argument("--input",       type=str, default=None,   help="Path to benchmark JSONL (e.g. data/gaia/gaia_lv1.jsonl).")
    parser.add_argument("--run-name",    type=str, default=None,   help="Run name for output directories. Defaults to input stem.")
    parser.add_argument("--limit",       type=int, default=None,   help="Process at most N tasks.")
    parser.add_argument("--task-id",     type=str, default=None,   help="Run only one specific task_id.")
    parser.add_argument("--model",       type=str, default="sonnet")
    parser.add_argument("--effort",      type=str, default="medium", help="low|medium|high|xhigh|max")
    parser.add_argument("--max-turns",   type=int, default=12)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--allowed-tools", type=str, default=DEFAULT_ALLOWED_TOOLS)
    parser.add_argument("--overwrite",   action="store_true",      help="Re-run tasks even if trace already exists.")
    parser.add_argument("--install-only", action="store_true",     help="Only write hook assets and exit.")
    parser.add_argument("--sleep-sec",   type=float, default=1.0)
    parser.add_argument("--max-cumulative-tokens", type=int, default=None)
    parser.add_argument("--disable-session-archive", action="store_true")
    # Plan-cache arguments
    parser.add_argument(
        "--plan-cache-dir",
        type=str,
        default=None,
        help="Path to plan-caching-study directory. Defaults to <EKB_ROOT>/traces/plan-caching-study.",
    )
    parser.add_argument(
        "--plan-rank-by",
        type=str,
        default="total_tokens",
        choices=["total_tokens", "total_latency_ms", "total_tool_calls"],
        help="Metric used to select the best KB neighbor (lower is better). Default: total_tokens.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.8,
        help="Minimum cosine similarity for a KB neighbor to be eligible. Default: 0.8.",
    )
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

    if not args.input:
        raise ValueError("--input is required unless --install-only is used.")

    plan_cache_dir = (
        Path(args.plan_cache_dir).expanduser().resolve()
        if args.plan_cache_dir
        else ekb_root / "traces" / "plan-caching-study"
    )
    print(f"[setup] plan-cache dir: {plan_cache_dir}")
    print(f"[setup] plan-rank-by:   {args.plan_rank_by}  min-similarity: {args.min_similarity}")

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

    result_root         = Path(os.environ["RESULT_ROOT"])
    shard_results_jsonl = result_root / "claude_native" / run_name / "results.jsonl"
    shard_summary_json  = result_root / "claude_native" / run_name / "summary.json"

    completed              = 0
    skipped                = 0
    failed                 = 0
    cache_hits             = 0
    cache_misses           = 0
    stopped_on_limit       = False
    stop_reason            = None
    cumulative_total_tokens = 0
    started_at             = utcnow()

    for idx, task in enumerate(tasks, start=1):
        safe_id = safe_task_id(task["task_id"])
        normalized_trace_path = result_root / "claude_native" / run_name / safe_id / "normalized_trace.json"

        if normalized_trace_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(tasks)}] skip {task['task_id']} (already exists)")
            skipped += 1
            continue

        print(f"[{idx}/{len(tasks)}] run {task['task_id']}")
        try:
            result = run_one_task(task, args, ekb_root, settings_path, run_name, plan_cache_dir)

            if result.get("limit_stop"):
                stopped_on_limit = True
                stop_reason = f"suspected usage/rate limit: {result.get('limit_signal')}"
                if result.get("total_tokens") is not None:
                    cumulative_total_tokens += int(result["total_tokens"])
                print(f"  -> stop: {stop_reason}")
                break

            append_jsonl(shard_results_jsonl, result)

            if result.get("total_tokens") is not None:
                cumulative_total_tokens += int(result["total_tokens"])

            if result.get("cache_hit"):
                cache_hits += 1
            else:
                cache_misses += 1

            if result.get("success"):
                completed += 1
                print(f"  -> success | pred={result.get('final_answer_pred')!r}")
            else:
                failed += 1
                print("  -> failed")

            if args.max_cumulative_tokens is not None and cumulative_total_tokens >= args.max_cumulative_tokens:
                stopped_on_limit = True
                stop_reason = f"manual token cap reached: {args.max_cumulative_tokens}"
                print(f"  -> stop: {stop_reason}")
                break

        except subprocess.TimeoutExpired as exc:
            failed += 1
            append_jsonl(shard_results_jsonl, {
                "query_id": task["task_id"],
                "success":  False,
                "failure":  {"type": "timeout", "timeout_sec": args.timeout_sec, "detail": str(exc)},
            })
            print("  -> timeout")
        except Exception as exc:
            failed += 1
            append_jsonl(shard_results_jsonl, {
                "query_id": task["task_id"],
                "success":  False,
                "failure":  {"type": "exception", "detail": repr(exc)},
            })
            print(f"  -> exception: {exc!r}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    summary = {
        "input":               str(input_path),
        "run_name":            run_name,
        "started_at":          started_at,
        "ended_at":            utcnow(),
        "selected_tasks":      len(tasks),
        "completed":           completed,
        "failed":              failed,
        "skipped":             skipped,
        "cache_hits":          cache_hits,
        "cache_misses":        cache_misses,
        "stopped_on_limit":    stopped_on_limit,
        "stop_reason":         stop_reason,
        "cumulative_total_tokens_this_run": cumulative_total_tokens,
        "plan_rank_by":        args.plan_rank_by,
        "min_similarity":      args.min_similarity,
        "settings_path":       str(settings_path),
        "hook_path":           str(hook_path),
    }
    write_json(shard_summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

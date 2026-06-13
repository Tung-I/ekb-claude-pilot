#!/usr/bin/env python3
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

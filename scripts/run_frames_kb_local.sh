#!/usr/bin/env bash
# Local (macOS) launcher for FRAMES KB trace collection (from scratch).
# Params aligned with run_frames_test_local.sh so latency/token stats are comparable.
# Resume-safe: skips tasks whose normalized_trace.json already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

python3 runners/run_claude_task_frames.py \
  --input data/frames/frames_kb.jsonl \
  --run-name frames_kb \
  --model sonnet \
  --effort medium \
  --max-turns 48 \
  --timeout-sec 900 \
  --disable-session-archive \
  --sleep-sec 1

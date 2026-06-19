#!/usr/bin/env bash
# Local (macOS) launcher for FRAMES test trace collection.
# Resume-safe: skips tasks whose normalized_trace.json already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

python3 runners/run_claude_task_frames.py \
  --input data/frames/frames_test.jsonl \
  --run-name frames_test \
  --model sonnet \
  --effort medium \
  --max-turns 48 \
  --timeout-sec 900 \
  --disable-session-archive \
  --sleep-sec 1

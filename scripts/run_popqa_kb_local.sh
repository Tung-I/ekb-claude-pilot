#!/usr/bin/env bash
# Local (macOS) launcher for PopQA KB trace collection.
# Resume-safe: skips tasks whose normalized_trace.json already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

python3 runners/run_claude_popqa.py \
  --input data/popqa/popqa_filtered_kb.jsonl \
  --run-name popqa_kb \
  --model sonnet \
  --effort low \
  --max-turns 32 \
  --timeout-sec 300 \
  --disable-session-archive \
  --sleep-sec 1

#!/usr/bin/env bash
# Local (macOS) launcher for GAIA L2 x3 trace collection.
# Higher budget than lv1 (lv2 is harder): --max-turns 72 --timeout-sec 1200.
# Resume-safe. Run serially (do not run alongside another collector — avoids API
# contention that would bias latency).
set -euo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

python3 runners/run_claude_task.py \
  --input data/gaia_paraphrased/gaia_lv2_x3.jsonl \
  --run-name gaia_lv2_x3 \
  --model sonnet --effort medium \
  --max-turns 72 --timeout-sec 1200 \
  --disable-session-archive --sleep-sec 1

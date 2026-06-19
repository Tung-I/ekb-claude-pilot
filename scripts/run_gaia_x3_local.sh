#!/usr/bin/env bash
# Local (macOS) SERIAL launcher for GAIA L1/L2 x3 paraphrase trace collection.
# Runs lv1_x3 then lv2_x3 one after another (never concurrent) so the Claude API
# is never contended -> latency is unbiased and comparable across levels.
# Same params for both levels for a fair difficulty comparison. Resume-safe.
set -euo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

# args: input, run_name, max_turns, timeout_sec
run() {
  python3 runners/run_claude_task.py \
    --input "$1" --run-name "$2" \
    --model sonnet --effort medium \
    --max-turns "$3" --timeout-sec "$4" \
    --disable-session-archive --sleep-sec 1
}

# lv1: easy -> 48 turns / 900s ; lv2: harder -> 72 turns / 1200s
echo "===== START gaia_lv1_x3 ($(date)) ====="
run data/gaia_paraphrased/gaia_lv1_x3.jsonl gaia_lv1_x3 48 900
echo "===== START gaia_lv2_x3 ($(date)) ====="
run data/gaia_paraphrased/gaia_lv2_x3.jsonl gaia_lv2_x3 72 1200
echo "===== ALL DONE ($(date)) ====="

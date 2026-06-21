#!/usr/bin/env bash
# SERIAL plan-reuse experiments for GAIA L1 (3 ranking signals, one after another
# so the Claude API is never contended -> unbiased latency).
# Test set = gaia_lv1_x3 originals; KB = gaia_lv1_x3 paraphrases; embeddings = gaia_lv1_x3 (mpnet 768d).
# Hyperparams match the gaia_lv1 trace collection (sonnet/medium/48 turns/900s).
set -euo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

run() {  # $1 = plan-rank-by, $2 = run-name
  python3 runners/run_claude_task_w_plan_reuse.py \
    --input data/gaia_paraphrased/gaia_lv1_x3_test.jsonl \
    --run-name "$2" \
    --kb-jsonl data/gaia_paraphrased/gaia_lv1_x3_kb.jsonl \
    --kb-trace-run gaia_lv1_x3 \
    --kb-embedding-run gaia_lv1_x3 \
    --plan-rank-by "$1" \
    --model sonnet --effort medium --max-turns 48 --timeout-sec 900 \
    --top-k 5 --min-similarity 0.8 \
    --disable-session-archive --sleep-sec 1
}

echo "===== START rank_by_step (total_tool_calls) $(date) ====="
run total_tool_calls  gaia_lv1_plan_reuse_rank_by_step
echo "===== START rank_by_token (total_tokens) $(date) ====="
run total_tokens       gaia_lv1_plan_reuse_rank_by_token
echo "===== START rank_by_latency (total_latency_ms) $(date) ====="
run total_latency_ms   gaia_lv1_plan_reuse_rank_by_latency
echo "===== ALL DONE $(date) ====="

#!/usr/bin/env bash
# SERIAL plan-reuse experiments for GAIA L2 (3 ranking signals), AUTO-CHAINED after
# the lv1 plan-reuse run. Starts only once lv1's final experiment (rank_by_latency)
# has written its summary AND no lv1 process is running -> never concurrent (unbiased latency).
# Test=gaia_lv2_x3 originals; KB=gaia_lv2_x3 paraphrases; embeddings=gaia_lv2_x3 (mpnet 768d).
# Hyperparams match the gaia_lv2 trace collection (sonnet/medium/72 turns/1200s).
set -uo pipefail
cd "$(dirname "$0")/.."
export EKB_ROOT="$PWD"
export TRACE_ROOT="$PWD/traces"
export RESULT_ROOT="$PWD/results"

LV1_DONE="results/claude_native/gaia_lv1_plan_reuse_rank_by_latency/summary.json"
echo "[chain] waiting for lv1 to finish ($LV1_DONE) ..."
while [ ! -f "$LV1_DONE" ]; do sleep 30; done
while pgrep -f run_gaia_lv1_plan_reuse_local.sh >/dev/null 2>&1; do sleep 10; done
while pgrep -f run_claude_task_w_plan_reuse.py   >/dev/null 2>&1; do sleep 10; done
echo "[chain] lv1 complete — starting lv2 at $(date)"

run() {  # $1 = plan-rank-by, $2 = run-name
  python3 runners/run_claude_task_w_plan_reuse.py \
    --input data/gaia_paraphrased/gaia_lv2_x3_test.jsonl \
    --run-name "$2" \
    --kb-jsonl data/gaia_paraphrased/gaia_lv2_x3_kb.jsonl \
    --kb-trace-run gaia_lv2_x3 \
    --kb-embedding-run gaia_lv2_x3 \
    --plan-rank-by "$1" \
    --model sonnet --effort medium --max-turns 72 --timeout-sec 1200 \
    --top-k 5 --min-similarity 0.8 \
    --disable-session-archive --sleep-sec 1
}

echo "===== START lv2 rank_by_step (total_tool_calls) $(date) ====="
run total_tool_calls  gaia_lv2_plan_reuse_rank_by_step
echo "===== START lv2 rank_by_token (total_tokens) $(date) ====="
run total_tokens       gaia_lv2_plan_reuse_rank_by_token
echo "===== START lv2 rank_by_latency (total_latency_ms) $(date) ====="
run total_latency_ms   gaia_lv2_plan_reuse_rank_by_latency
echo "===== ALL LV2 DONE $(date) ====="

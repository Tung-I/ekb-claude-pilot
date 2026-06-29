#!/usr/bin/env bash
# Stronger-origin (Opus) tier for the cascade quality ceiling.
# Opus is ~5x Sonnet cost -> run on a representative sample, not full sets.
# Launch ONLY after the Haiku edge chain completes (avoid API contention).
# Resume-safe.
set -uo pipefail
cd /Users/tungi/ekb-claude-pilot

echo "[$(date)] === Opus origin: PopQA test (first 200) ==="
python3 runners/run_claude_popqa.py \
  --input data/popqa/popqa_filtered_test.jsonl --run-name popqa_origin_opus \
  --model opus --effort low --max-turns 32 --timeout-sec 300 \
  --disable-session-archive --sleep-sec 1 --limit 200

echo "[$(date)] === Opus origin: FRAMES test (197) ==="
python3 runners/run_claude_task_frames.py \
  --input data/frames/frames_test.jsonl --run-name frames_origin_opus \
  --model opus --effort medium --max-turns 40 --timeout-sec 480 \
  --disable-session-archive --sleep-sec 1

echo "[$(date)] === Opus origin: GAIA L1 (51) ==="
python3 runners/run_claude_task.py \
  --input data/gaia/gaia_lv1.jsonl --run-name gaia_lv1_origin_opus \
  --model opus --effort medium --max-turns 32 --timeout-sec 600 \
  --disable-session-archive --sleep-sec 1

echo "[$(date)] === Opus origin COMPLETE ==="

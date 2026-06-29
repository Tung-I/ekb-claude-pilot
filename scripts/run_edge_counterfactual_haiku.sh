#!/usr/bin/env bash
# Edge-tier (Haiku) counterfactual collection across the difficulty spectrum.
# Sequential (one model at a time) to limit API contention with other sessions.
# Resume-safe: each runner skips tasks whose normalized_trace.json already exists.
set -uo pipefail
cd /Users/tungi/ekb-claude-pilot

echo "[$(date)] === Haiku edge: PopQA test (1202) ==="
python3 runners/run_claude_popqa.py \
  --input data/popqa/popqa_filtered_test.jsonl --run-name popqa_edge_haiku \
  --model haiku --effort low --max-turns 32 --timeout-sec 300 \
  --disable-session-archive --sleep-sec 1

echo "[$(date)] === Haiku edge: FRAMES test (197) ==="
python3 runners/run_claude_task_frames.py \
  --input data/frames/frames_test.jsonl --run-name frames_edge_haiku \
  --model haiku --effort medium --max-turns 40 --timeout-sec 480 \
  --disable-session-archive --sleep-sec 1

echo "[$(date)] === Haiku edge: GAIA L1 (51 originals) ==="
python3 runners/run_claude_task.py \
  --input data/gaia/gaia_lv1.jsonl --run-name gaia_lv1_edge_haiku \
  --model haiku --effort medium --max-turns 32 --timeout-sec 600 \
  --disable-session-archive --sleep-sec 1

echo "[$(date)] === Haiku edge counterfactual COMPLETE ==="

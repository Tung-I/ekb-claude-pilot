  module load conda/latest && conda activate /work/pi_rsitaram_umass_edu/tungi/conda/envs/ekb

  export EKB_ROOT=/work/pi_rsitaram_umass_edu/tungi/ekb-claude-pilot
  export TRACE_ROOT=$EKB_ROOT/traces
  export RESULT_ROOT=$EKB_ROOT/results

  python runners/run_claude_task_frames.py \
    --input data/frames/frames_kb.jsonl \
    --run-name frames_kb \
    --model sonnet \
    --effort medium \
    --max-turns 48 \
    --timeout-sec 600 \
    --disable-session-archive \
    --sleep-sec 1
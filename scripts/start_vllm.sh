#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/QwQ-32B-AWQ"

vllm serve "$MODEL" \
  --served-model-name qwq-32b-awq \
  --api-key local-key \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
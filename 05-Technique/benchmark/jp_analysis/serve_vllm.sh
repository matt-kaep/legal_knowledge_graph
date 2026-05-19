#!/usr/bin/env bash
set -euo pipefail
# gemma4-31B-AWQ via vLLM OpenAI-compatible server.
# max-model-len MUST be explicit (adversarial finding #4) — never inherit a default.
MODEL_ID="${MODEL_ID:-QuantTrio/gemma-4-31B-it-AWQ}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --guided-decoding-backend xgrammar \
  --gpu-memory-utilization 0.92

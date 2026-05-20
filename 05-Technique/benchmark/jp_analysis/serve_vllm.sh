#!/usr/bin/env bash
set -euo pipefail
# gemma4-31B-AWQ via vLLM OpenAI-compatible server.
# max-model-len MUST be explicit (adversarial finding #4) — never inherit a default.
MODEL_ID="${MODEL_ID:-QuantTrio/gemma-4-31B-it-AWQ}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"

MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
# vLLM 0.21 : --guided-decoding-backend supprimé (xgrammar par défaut) ; pour
# gemma-4 (multimodal) il FAUT --max-num-batched-tokens ≥ max_tokens_per_mm_item
# (~2496) sinon refus au démarrage, même en usage text-only.

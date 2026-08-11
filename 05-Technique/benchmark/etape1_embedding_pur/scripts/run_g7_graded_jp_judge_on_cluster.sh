#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
export LKG_REPO="${LKG_REPO:-$HOME/legal_knowledge_graph}"
SCRIPT_DIR="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts"
OUT_DIR="${OUT_DIR:-$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/G7-citation-JJ-cit1-sem025-knn5/eval_rich_retrievable_strict/E016-g7-graded-jp-v1}"
MODEL_ID="${MODEL_ID:-cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-18000}"
GPU_UTIL="${GPU_UTIL:-0.92}"
WORKERS="${WORKERS:-32}"

mkdir -p "$OUT_DIR/logs"
VLLM_LOG="$OUT_DIR/logs/vllm_g7_graded_${SLURM_JOB_ID:-local}.log"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM stopped during startup; see $VLLM_LOG" >&2
    exit 2
  fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "vLLM did not become ready; see $VLLM_LOG" >&2
  exit 2
fi

ARGS=(--jobs "$OUT_DIR/judge_jobs.jsonl" --out-dir "$OUT_DIR" --model-id "$MODEL_ID" --port "$PORT" --workers "$WORKERS")
if [ "$MODE" = "pilot" ]; then
  ARGS+=(--limit "${PILOT_LIMIT:-30}")
elif [ "$MODE" != "full" ]; then
  echo "mode must be pilot or full" >&2
  exit 1
fi
if [ "${RETRY_NON_OK:-0}" = "1" ]; then
  ARGS+=(--retry-non-ok)
fi

python3 "$SCRIPT_DIR/76_run_g7_graded_jp_judge.py" "${ARGS[@]}"

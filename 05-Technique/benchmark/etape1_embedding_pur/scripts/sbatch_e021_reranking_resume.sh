#!/usr/bin/env bash
#SBATCH --job-name=lkg-e021-resume
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Continues only missing or invalid E021 family/question units. The original
# response JSONL remains append-only; metrics and the receipt have new paths.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the code checkout before submitting.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout before submitting.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
DATA_BENCH="$LKG_DATA_ROOT/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
RESUME_MANIFEST="$LKG_REPO/experiments/reranking-comparable/manifest_cluster_gpu_runtime_v5_resume_v2.json"
JOBS="$DATA_BENCH/_e021_jobs/E021-cluster-gpu-runtime-v5/jobs.jsonl"
RESPONSES="$DATA_BENCH/_e021_jobs/E021-cluster-gpu-runtime-v5/responses.jsonl"
QUESTIONS="$DATA_BENCH/eval_rich_retrievable_strict/bench_global.json"
PROMPT="$ROOT/prompts/reranking_comparable_v1.txt"
RUN_ROOT="$DATA_BENCH/_e021_jobs/E021-cluster-gpu-runtime-v5/resume_v2"
PORT="${E021_VLLM_PORT:-8000}"
MODEL="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
REVISION="4033b16200f4152e55e100ea12dc388c537df622"
MODEL_SNAPSHOT="${E021_MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/$REVISION}"
EXPECTED_JOBS_SHA="36f03198d39ec764095d3340ea1f8dc006b941e585245e30bd8e4c14a0a5afdf"
EXPECTED_RESPONSES_SHA="780e53c1d69481660869d4c0f9e68b377be7d4ab2f0b5c869eae1522b9a3a9fb"
SERVER_PID=""

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1

stop_server() {
  status=$?
  trap - EXIT
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap stop_server EXIT

for path in "$PYTHON_BIN" "$VLLM_BIN" "$RESUME_MANIFEST" "$JOBS" "$RESPONSES" "$QUESTIONS" "$PROMPT" "$MODEL_SNAPSHOT"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

actual_jobs_sha="$(sha256sum "$JOBS" | awk '{print $1}')"
actual_responses_sha="$(sha256sum "$RESPONSES" | awk '{print $1}')"
[[ "$actual_jobs_sha" == "$EXPECTED_JOBS_SHA" ]] || { echo "jobs SHA-256 mismatch" >&2; exit 2; }
[[ "$actual_responses_sha" == "$EXPECTED_RESPONSES_SHA" ]] || { echo "response history changed before resume" >&2; exit 2; }

export LKG_REPO LKG_DATA_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON_BIN" "$ROOT/scripts/70_aggregate_e021_metrics.py" \
  --questions "$QUESTIONS" --jobs "$JOBS" --responses "$RESPONSES" \
  --output "$RUN_ROOT/pre_resume_metrics.json"

missing_units="$("$PYTHON_BIN" - "$RUN_ROOT/pre_resume_metrics.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(int(row["missing_questions"]) for row in payload["families"].values()))
PY
)"

if [[ "$missing_units" -gt 0 ]]; then
  "$VLLM_BIN" serve "$MODEL_SNAPSHOT" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len 16384 --max-num-seqs 2 \
    --gpu-memory-utilization 0.9 > "$RUN_ROOT/vllm.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited before becoming healthy" >&2
      tail -n 80 "$RUN_ROOT/vllm.log" >&2 || true
      exit 1
    fi
    sleep 5
  done
  curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null
  "$PYTHON_BIN" "$ROOT/scripts/68_run_e021_reranking.py" run \
    --jobs "$JOBS" --responses "$RESPONSES" \
    --endpoint "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL" --prompt "$PROMPT"
fi

"$PYTHON_BIN" "$ROOT/scripts/70_aggregate_e021_metrics.py" \
  --questions "$QUESTIONS" --jobs "$JOBS" --responses "$RESPONSES" \
  --output "$RUN_ROOT/metrics.json"
"$PYTHON_BIN" "$ROOT/scripts/72_finalize_e021_resume.py" \
  --jobs "$JOBS" --responses "$RESPONSES" --metrics "$RUN_ROOT/metrics.json" \
  --output "$RUN_ROOT/completion_receipt.json" \
  --resume-manifest "$RESUME_MANIFEST" \
  --expected-family cosine_bge_m3 --expected-family ppr --expected-family lightgcn \
  --expected-questions-per-family 754

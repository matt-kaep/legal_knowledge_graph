#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e027-direct
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# E027: direct LLM baseline.  Submit only after the G6 LightGCN queue is empty:
# it consumes the same per-user GPU quota even though it targets another partition.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the reproducibility checkout.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E027_MODALITY:?Set E027_MODALITY to article or jp.}"
: "${E027_MANIFEST_SHA:?Pass the SHA-256 of the frozen B2 manifest at submission.}"

case "$E027_MODALITY" in article|jp) ;; *) echo "invalid E027_MODALITY: $E027_MODALITY" >&2; exit 2 ;; esac

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
MANIFEST="$ROOT/configs/b2_direct_llm_a3_v1.json"
RUNNER="$ROOT/scripts/100_run_b2_direct_llm.py"
PORT="${E027_VLLM_PORT:-8017}"
SERVER_PID=""

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

for path in "$PYTHON_BIN" "$VLLM_BIN" "$MANIFEST" "$RUNNER"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$E027_MANIFEST_SHA" ]] || {
  echo "B2 manifest SHA-256 mismatch" >&2
  exit 2
}

eval "$("$PYTHON_BIN" - "$MANIFEST" "$E027_MODALITY" <<'PY'
import json
import shlex
import sys

manifest_path, modality = sys.argv[1:]
payload = json.load(open(manifest_path, encoding="utf-8"))
if payload["experiment_id"] != "E027" or payload["status"] != "frozen_authorized_waiting_for_g6_queue":
    raise SystemExit("unexpected B2 manifest identity or status")
if payload["model"]["temperature"] != 0:
    raise SystemExit("E027 temperature must be exactly zero")
task = payload["tasks"].get(modality)
if not task:
    raise SystemExit(f"missing frozen task for {modality}")
for key, value in {
    "MODEL": payload["model"]["id"],
    "REVISION": payload["model"]["revision"],
    "PROMPT_REL": task["prompt"]["path"],
    "PROMPT_SHA": task["prompt"]["sha256"],
    "JOBS_REL": task["jobs"]["path"],
    "JOBS_SHA": task["jobs"]["sha256"],
    "QUESTIONS_REL": payload["datasets"]["evaluation"]["path"],
    "QUESTIONS_SHA": payload["datasets"]["evaluation"]["sha256"],
    "A3_REL": payload["a3"]["manifest_path"],
    "A3_SHA": payload["a3"]["sha256"],
    "OUT_REL": task["outputs"]["root"],
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

PROMPT="$LKG_REPO/$PROMPT_REL"
JOBS="$LKG_DATA_ROOT/$JOBS_REL"
QUESTIONS="$LKG_DATA_ROOT/$QUESTIONS_REL"
A3="$LKG_REPO/$A3_REL"
RUN_ROOT="$LKG_DATA_ROOT/$OUT_REL"
RESPONSES="$RUN_ROOT/responses.jsonl"
MATERIALIZED="$RUN_ROOT/materialized"
MODEL_SNAPSHOT="${E027_MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/$REVISION}"

for path in "$PROMPT" "$QUESTIONS" "$A3" "$MODEL_SNAPSHOT"; do
  [[ -e "$path" ]] || { echo "missing frozen E027 input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PROMPT" | awk '{print $1}')" == "$PROMPT_SHA" ]] || { echo "prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$QUESTIONS" | awk '{print $1}')" == "$QUESTIONS_SHA" ]] || { echo "evaluation SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$A3" | awk '{print $1}')" == "$A3_SHA" ]] || { echo "A3 SHA mismatch" >&2; exit 2; }

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1

export LKG_REPO LKG_DATA_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -e "$JOBS" ]]; then
  mkdir -p "$(dirname "$JOBS")"
  "$PYTHON_BIN" "$RUNNER" prepare \
    --questions "$QUESTIONS" --output "$JOBS" --modality "$E027_MODALITY" --prompt "$PROMPT" \
    --experiment-id E027 --model-id "$MODEL" --model-revision "$REVISION"
fi
[[ "$(sha256sum "$JOBS" | awk '{print $1}')" == "$JOBS_SHA" ]] || { echo "frozen jobs SHA mismatch" >&2; exit 2; }

if [[ -f "$MATERIALIZED/materialization_receipt.json" ]]; then
  echo "E027 $E027_MODALITY already materialized; leaving immutable output untouched."
  exit 0
fi

"$VLLM_BIN" serve "$MODEL_SNAPSHOT" \
  --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 --max-num-seqs 2 --gpu-memory-utilization 0.90 \
  > "$RUN_ROOT/vllm.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 100 "$RUN_ROOT/vllm.log" >&2 || true
    exit 1
  fi
  sleep 5
done
curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null

"$PYTHON_BIN" "$RUNNER" run \
  --jobs "$JOBS" --responses "$RESPONSES" --a3-manifest "$A3" \
  --endpoint "http://127.0.0.1:${PORT}/v1" --model-id "$MODEL" --prompt "$PROMPT"

"$PYTHON_BIN" - "$JOBS" "$RESPONSES" <<'PY'
import json
import sys

jobs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        jobs[str(row["qid"])] = row["input_sha256"]
terminal = {}
for line in open(sys.argv[2], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        if row.get("status") in {"ok", "invalid"}:
            terminal[str(row["qid"])] = row
complete = {
    qid for qid, input_sha in jobs.items()
    if qid in terminal and terminal[qid].get("input_sha256") == input_sha
}
if complete != set(jobs):
    raise SystemExit(f"incomplete E027 response coverage: {len(complete)}/{len(jobs)}")
PY

"$PYTHON_BIN" "$RUNNER" materialize \
  --questions "$QUESTIONS" --responses "$RESPONSES" --a3-manifest "$A3" \
  --modality "$E027_MODALITY" --out-dir "$MATERIALIZED"

"$PYTHON_BIN" - "$RUN_ROOT" "$E027_MODALITY" "$actual_manifest_sha" "$RUNNER" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = root / "materialized" / "materialization_receipt.json"
payload = {
    "experiment_id": "E027",
    "modality": sys.argv[2],
    "b2_manifest_sha256": sys.argv[3],
    "runner_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "materialization_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip(),
}
(root / "run_receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

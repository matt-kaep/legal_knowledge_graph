#!/usr/bin/env bash
# Execute one immutable E030 LLM-as-a-Judge shard after its context gate.
# The execution manifest is intentionally created only after final rankings exist.
#SBATCH --job-name=lkg-b2-e030
#SBATCH --partition=A100,L40S,H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the isolated reproducibility staging.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E030_EXECUTION_MANIFEST:?Set the absolute frozen E030 execution manifest path.}"
: "${E030_EXECUTION_MANIFEST_SHA:?Pass the frozen E030 execution manifest SHA-256.}"
: "${E030_SHARD:?Set the immutable E030 shard identifier.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
RUNNER="$ROOT/scripts/110_run_b2_e030_llm_judge.py"
AUDITOR="$ROOT/scripts/114_audit_b2_e030_context.py"
PORT="${E030_VLLM_PORT:-8030}"
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

for path in "$PYTHON_BIN" "$VLLM_BIN" "$E030_EXECUTION_MANIFEST" "$RUNNER" "$AUDITOR"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
actual_manifest_sha="$(sha256sum "$E030_EXECUTION_MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$E030_EXECUTION_MANIFEST_SHA" ]] || { echo "E030 execution manifest SHA mismatch" >&2; exit 2; }

eval "$("$PYTHON_BIN" - "$E030_EXECUTION_MANIFEST" "$E030_SHARD" <<'PY'
import json
import shlex
import sys

manifest_path, shard_id = sys.argv[1:]
payload = json.load(open(manifest_path, encoding="utf-8"))
if payload.get("experiment_id") != "E030" or payload.get("status") != "frozen_authorized_model_execution_v1":
    raise SystemExit("unexpected E030 execution manifest")
model = payload.get("model", {})
if model.get("temperature") != 0:
    raise SystemExit("E030 temperature must be exactly zero")
if not isinstance(model.get("max_output_tokens"), int) or model["max_output_tokens"] <= 0:
    raise SystemExit("E030 requires a positive frozen output budget")
shard = next((item for item in payload.get("shards", []) if item.get("id") == shard_id), None)
if not isinstance(shard, dict) or shard.get("modality") not in {"article", "jp"}:
    raise SystemExit("unknown E030 shard or modality")
for key in ("jobs", "context_audit"):
    if not isinstance(shard.get(key), dict):
        raise SystemExit(f"E030 shard lacks {key}")
if int(shard["jobs"].get("count", 0)) <= 0:
    raise SystemExit("E030 shard requires a non-empty frozen job list")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

prompt = payload["prompts"][shard["modality"]]
for name, value in {
    "MODEL": model["id"],
    "MODEL_SNAPSHOT": model["snapshot"],
    "MAX_MODEL_TOKENS": model["max_model_tokens"],
    "MAX_OUTPUT_TOKENS": model["max_output_tokens"],
    "MAX_NUM_SEQS": model.get("max_num_seqs", 2),
    "MAX_WORKERS": model.get("max_workers", 2),
    "PROMPT_REL": prompt["path"],
    "PROMPT_SHA": prompt["sha256"],
    "OUT_REL": payload["outputs"]["root"],
    "JOBS_REL": shard["jobs"]["path"],
    "JOBS_SHA": shard["jobs"]["sha256"],
    "EXPECTED_JOBS": shard["jobs"]["count"],
    "AUDIT_REL": shard["context_audit"]["path"],
    "AUDIT_SHA": shard["context_audit"]["sha256"],
    "RUNNER_SHA": payload["code_bundle"]["judge_runner"]["sha256"],
    "AUDITOR_SHA": payload["code_bundle"]["context_auditor"]["sha256"],
}.items():
    emit(name, value)
PY
)"

PROMPT="$LKG_REPO/$PROMPT_REL"
JOBS="$LKG_DATA_ROOT/$JOBS_REL"
AUDIT="$LKG_DATA_ROOT/$AUDIT_REL"
OUT_ROOT="$LKG_DATA_ROOT/$OUT_REL"
RUN_ROOT="$OUT_ROOT/shards/$E030_SHARD"
RESPONSES="$RUN_ROOT/responses.jsonl"
MATERIALIZED="$RUN_ROOT/materialized"

for path in "$MODEL_SNAPSHOT" "$PROMPT" "$JOBS" "$AUDIT"; do
  [[ -e "$path" ]] || { echo "missing frozen E030 input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PROMPT" | awk '{print $1}')" == "$PROMPT_SHA" ]] || { echo "E030 prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JOBS" | awk '{print $1}')" == "$JOBS_SHA" ]] || { echo "E030 jobs SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AUDIT" | awk '{print $1}')" == "$AUDIT_SHA" ]] || { echo "E030 context audit SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA" ]] || { echo "E030 runner SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AUDITOR" | awk '{print $1}')" == "$AUDITOR_SHA" ]] || { echo "E030 auditor SHA mismatch" >&2; exit 2; }

"$PYTHON_BIN" - "$JOBS" "$AUDIT" "$EXPECTED_JOBS" <<'PY'
import json
import sys

jobs = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
audit = json.load(open(sys.argv[2], encoding="utf-8"))
if len(jobs) != int(sys.argv[3]):
    raise SystemExit(f"frozen E030 job count differs: {len(jobs)}/{sys.argv[3]}")
if audit.get("model_calls") != 0 or audit.get("compatible") is not True:
    raise SystemExit("E030 context audit does not authorize model execution")
job_ids = [row.get("job_id") for row in jobs]
if len(job_ids) != len(set(job_ids)):
    raise SystemExit("duplicate immutable E030 job id")
PY

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
if [[ -f "$MATERIALIZED/materialization_receipt.json" ]]; then
  echo "E030 shard $E030_SHARD is already materialized; immutable output preserved."
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
gpu_compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)"
if awk 'BEGIN { rejected = 0 } { if ($1 + 0 < 7.0) rejected = 1 } END { exit rejected ? 0 : 1 }' <<< "$gpu_compute_capability"; then
  echo "GPU compute_capability $gpu_compute_capability is below the minimum required capability 7.0 for compressed-tensors Gemma" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
"$VLLM_BIN" serve "$MODEL_SNAPSHOT" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$MAX_MODEL_TOKENS" --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.90 > "$RUN_ROOT/vllm.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
  curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 100 "$RUN_ROOT/vllm.log" >&2; exit 1; }
  sleep 5
done
curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null

"$PYTHON_BIN" "$RUNNER" run --jobs "$JOBS" --responses "$RESPONSES" --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model-id "$MODEL" --prompt "$PROMPT" --max-workers "$MAX_WORKERS" --max-tokens "$MAX_OUTPUT_TOKENS"
"$PYTHON_BIN" "$RUNNER" materialize --jobs "$JOBS" --responses "$RESPONSES" --out-dir "$MATERIALIZED"

"$PYTHON_BIN" - "$RUN_ROOT" "$E030_SHARD" "$actual_manifest_sha" "$RUNNER" "$AUDITOR" "$JOBS" "$AUDIT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = root / "materialized" / "materialization_receipt.json"
payload = {
    "experiment_id": "E030",
    "shard": sys.argv[2],
    "execution_manifest_sha256": sys.argv[3],
    "judge_runner_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "context_auditor_sha256": hashlib.sha256(Path(sys.argv[5]).read_bytes()).hexdigest(),
    "jobs_sha256": hashlib.sha256(Path(sys.argv[6]).read_bytes()).hexdigest(),
    "context_audit_sha256": hashlib.sha256(Path(sys.argv[7]).read_bytes()).hexdigest(),
    "materialization_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip(),
}
(root / "run_receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

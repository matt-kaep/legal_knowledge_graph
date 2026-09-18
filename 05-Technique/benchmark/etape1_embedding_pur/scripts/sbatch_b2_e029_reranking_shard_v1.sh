#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e029-reranking
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Run exactly one frozen E029 reranking shard.  The immutable execution
# manifest supplies a prebuilt, hash-checked jobs file: this script must never
# materialize pools, select a configuration, or alter full candidate texts.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the reproducibility checkout.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E029_EXECUTION_MANIFEST_SHA:?Pass the SHA-256 of the frozen execution manifest.}"
: "${E029_SHARD:?Set the immutable shard identifier.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
MANIFEST="$ROOT/configs/b2_reranking_comparable_a3_execution_v1.json"
RUNNER="$ROOT/scripts/102_run_b2_comparable_reranking.py"
PORT="${E029_VLLM_PORT:-8029}"
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
[[ "$actual_manifest_sha" == "$E029_EXECUTION_MANIFEST_SHA" ]] || {
  echo "E029 execution manifest SHA-256 mismatch" >&2
  exit 2
}

eval "$("$PYTHON_BIN" - "$MANIFEST" "$E029_SHARD" <<'PY'
import json
import shlex
import sys

manifest_path, shard_id = sys.argv[1:]
payload = json.load(open(manifest_path, encoding="utf-8"))
if payload["experiment_id"] != "E029" or payload["status"] != "frozen_authorized_model_execution":
    raise SystemExit("unexpected E029 execution manifest identity or status")
if payload["model"]["temperature"] != 0:
    raise SystemExit("E029 temperature must be exactly zero")
if payload["reranking_contract"]["candidate_text_policy"] != "full_text_unmodified":
    raise SystemExit("candidate text policy must remain full_text_unmodified")
if payload["reranking_contract"]["truncation"] != "forbidden":
    raise SystemExit("truncation must remain forbidden")
if payload["reranking_contract"]["k_out"] != 10:
    raise SystemExit("K_out must be exactly 10")
shard = next((row for row in payload["shards"] if row["id"] == shard_id), None)
if shard is None:
    raise SystemExit(f"unknown E029 shard: {shard_id}")
if not shard["conditions"]:
    raise SystemExit("E029 shard has no frozen conditions")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

for name, value in {
    "MODEL": payload["model"]["id"],
    "REVISION": payload["model"]["revision"],
    "A3_REL": payload["a3"]["manifest_path"],
    "A3_SHA": payload["a3"]["sha256"],
    "QUESTIONS_REL": payload["a3"]["evaluation_path"],
    "QUESTIONS_SHA": payload["a3"]["evaluation_sha256"],
    "ARTICLE_PROMPT_REL": payload["prompts"]["article"]["path"],
    "ARTICLE_PROMPT_SHA": payload["prompts"]["article"]["sha256"],
    "JP_PROMPT_REL": payload["prompts"]["jp"]["path"],
    "JP_PROMPT_SHA": payload["prompts"]["jp"]["sha256"],
    "OUT_REL": payload["outputs"]["root"],
    "JOBS_REL": shard["jobs"]["path"],
    "JOBS_SHA": shard["jobs"]["sha256"],
    "EXPECTED_JOBS": shard["jobs"]["count"],
    "RUNNER_SHA": payload["code_bundle"]["reranking_runner"]["sha256"],
}.items():
    emit(name, value)
PY
)"

A3="$LKG_REPO/$A3_REL"
QUESTIONS="$LKG_DATA_ROOT/$QUESTIONS_REL"
ARTICLE_PROMPT="$LKG_REPO/$ARTICLE_PROMPT_REL"
JP_PROMPT="$LKG_REPO/$JP_PROMPT_REL"
JOBS="$LKG_DATA_ROOT/$JOBS_REL"
OUT_ROOT="$LKG_DATA_ROOT/$OUT_REL"
RUN_ROOT="$OUT_ROOT/shards/$E029_SHARD"
RESPONSES="$RUN_ROOT/responses.jsonl"
MATERIALIZED="$RUN_ROOT/materialized"
MODEL_SNAPSHOT="${E029_MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/$REVISION}"

for path in "$A3" "$QUESTIONS" "$ARTICLE_PROMPT" "$JP_PROMPT" "$JOBS" "$MODEL_SNAPSHOT"; do
  [[ -e "$path" ]] || { echo "missing frozen E029 input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$A3" | awk '{print $1}')" == "$A3_SHA" ]] || { echo "A3 SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$QUESTIONS" | awk '{print $1}')" == "$QUESTIONS_SHA" ]] || { echo "evaluation SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$ARTICLE_PROMPT" | awk '{print $1}')" == "$ARTICLE_PROMPT_SHA" ]] || { echo "article prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_PROMPT" | awk '{print $1}')" == "$JP_PROMPT_SHA" ]] || { echo "JP prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JOBS" | awk '{print $1}')" == "$JOBS_SHA" ]] || { echo "frozen jobs SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA" ]] || { echo "reranking runner SHA mismatch" >&2; exit 2; }

"$PYTHON_BIN" - "$JOBS" "$EXPECTED_JOBS" <<'PY'
import json
import sys

jobs = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if len(jobs) != int(sys.argv[2]):
    raise SystemExit(f"frozen job count differs: {len(jobs)}/{sys.argv[2]}")
keys = {
    (row["family"], row["modality"], int(row["k_in"]), str(row.get("replay_seed") or ""), row["qid"])
    for row in jobs
}
if len(keys) != len(jobs):
    raise SystemExit("duplicate immutable E029 job key")
if any(len(row["candidates"]) != int(row["k_in"]) for row in jobs):
    raise SystemExit("frozen job has an invalid pool cardinality")
if any(int(row["k_out"]) != 10 for row in jobs):
    raise SystemExit("frozen job violates K_out=10")
PY

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
export LKG_REPO LKG_DATA_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ -f "$MATERIALIZED/materialization_receipt.json" ]]; then
  echo "E029 shard $E029_SHARD already materialized; leaving immutable output untouched."
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
  --jobs "$JOBS" --responses "$RESPONSES" --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model-id "$MODEL" --prompt-article "$ARTICLE_PROMPT" --prompt-jp "$JP_PROMPT"

"$PYTHON_BIN" - "$JOBS" "$RESPONSES" <<'PY'
import json
import sys

def key(row):
    return row["family"], row["modality"], int(row["k_in"]), str(row.get("replay_seed") or ""), row["qid"]

jobs = {key(row): row for line in open(sys.argv[1], encoding="utf-8") if line.strip() for row in [json.loads(line)]}
terminal = {}
for line in open(sys.argv[2], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        if row.get("status") in {"ok", "invalid"}:
            terminal[key(row)] = row
complete = {
    item for item, job in jobs.items()
    if item in terminal and terminal[item].get("input_sha256") == job.get("input_sha256")
}
if complete != set(jobs):
    raise SystemExit(f"incomplete E029 shard response coverage: {len(complete)}/{len(jobs)}")
PY

"$PYTHON_BIN" "$RUNNER" materialize \
  --questions "$QUESTIONS" --jobs "$JOBS" --responses "$RESPONSES" --out-dir "$MATERIALIZED"

"$PYTHON_BIN" - "$RUN_ROOT" "$E029_SHARD" "$actual_manifest_sha" "$RUNNER" "$JOBS" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = root / "materialized" / "materialization_receipt.json"
payload = {
    "experiment_id": "E029",
    "shard": sys.argv[2],
    "execution_manifest_sha256": sys.argv[3],
    "runner_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "jobs_sha256": hashlib.sha256(Path(sys.argv[5]).read_bytes()).hexdigest(),
    "materialization_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip(),
}
(root / "run_receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e029-k70
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Run exactly one successor E029 shard.  The caller must supply the absolute
# frozen execution manifest; this wrapper never regenerates pools or jobs.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the isolated reproducibility staging.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E029_EXECUTION_MANIFEST:?Set the absolute frozen execution manifest path.}"
: "${E029_EXECUTION_MANIFEST_SHA:?Pass the frozen execution manifest SHA-256.}"
: "${E029_SHARD:?Set the immutable shard identifier.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
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

for path in "$PYTHON_BIN" "$VLLM_BIN" "$E029_EXECUTION_MANIFEST" "$RUNNER"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
actual_manifest_sha="$(sha256sum "$E029_EXECUTION_MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$E029_EXECUTION_MANIFEST_SHA" ]] || { echo "execution manifest SHA mismatch" >&2; exit 2; }

eval "$("$PYTHON_BIN" - "$E029_EXECUTION_MANIFEST" "$E029_SHARD" <<'PY'
import json
import shlex
import sys

manifest_path, shard_id = sys.argv[1:]
payload = json.load(open(manifest_path, encoding="utf-8"))
if payload.get("experiment_id") != "E029" or payload.get("status") != "frozen_authorized_model_execution":
    raise SystemExit("unexpected E029 execution manifest")
contract = payload["reranking_contract"]
if contract.get("depth_grid") != [10, 20, 30, 40, 50, 60, 70] or contract.get("k_out") != 10:
    raise SystemExit("unexpected E029 depth grid or K_out")
if contract.get("representation") != {
    "article": {"source_field": "texte", "projection": "token_prefix", "token_cap": 192},
    "jp": {"source_field": "synthese", "projection": "complete_unmodified"},
}:
    raise SystemExit("unexpected E029 text representation")
if payload["model"].get("temperature") != 0:
    raise SystemExit("E029 temperature must be exactly zero")
shard = next((item for item in payload["shards"] if item["id"] == shard_id), None)
if shard is None or not shard.get("conditions"):
    raise SystemExit(f"unknown or empty E029 shard: {shard_id}")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

for name, value in {
    "MODEL": payload["model"]["id"],
    "REVISION": payload["model"]["revision"],
    "MODEL_SNAPSHOT": payload["model"]["snapshot"],
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

for path in "$A3" "$QUESTIONS" "$ARTICLE_PROMPT" "$JP_PROMPT" "$JOBS" "$MODEL_SNAPSHOT"; do
  [[ -e "$path" ]] || { echo "missing frozen E029 input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$A3" | awk '{print $1}')" == "$A3_SHA" ]] || { echo "A3 SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$QUESTIONS" | awk '{print $1}')" == "$QUESTIONS_SHA" ]] || { echo "evaluation SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$ARTICLE_PROMPT" | awk '{print $1}')" == "$ARTICLE_PROMPT_SHA" ]] || { echo "Article prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_PROMPT" | awk '{print $1}')" == "$JP_PROMPT_SHA" ]] || { echo "JP prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JOBS" | awk '{print $1}')" == "$JOBS_SHA" ]] || { echo "job SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA" ]] || { echo "runner SHA mismatch" >&2; exit 2; }

"$PYTHON_BIN" - "$JOBS" "$EXPECTED_JOBS" <<'PY'
import json
import sys

jobs = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if len(jobs) != int(sys.argv[2]):
    raise SystemExit(f"frozen job count differs: {len(jobs)}/{sys.argv[2]}")
keys = {(row["family"], row["modality"], int(row["k_in"]), str(row.get("replay_seed") or ""), row["qid"]) for row in jobs}
if len(keys) != len(jobs):
    raise SystemExit("duplicate immutable E029 job key")
for row in jobs:
    if len(row["candidates"]) != int(row["k_in"]) or int(row["k_out"]) != 10:
        raise SystemExit("invalid frozen E029 candidate cardinality or K_out")
    representation = row.get("candidate_text_representation")
    expected = ({"source_field": "texte", "projection": "token_prefix", "tokenizer_id": row["model_id"], "tokenizer_revision": row["model_revision"], "token_cap": 192}
                if row["modality"] == "article" else {"source_field": "synthese", "projection": "complete_unmodified"})
    if representation != expected:
        raise SystemExit("job text representation differs from the frozen contract")
PY

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
if [[ -f "$MATERIALIZED/materialization_receipt.json" ]]; then
  echo "E029 shard $E029_SHARD is already materialized; immutable output preserved."
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$VLLM_BIN" serve "$MODEL_SNAPSHOT" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 --max-num-seqs 2 --gpu-memory-utilization 0.90 > "$RUN_ROOT/vllm.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
  curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 100 "$RUN_ROOT/vllm.log" >&2; exit 1; }
  sleep 5
done
curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null

"$PYTHON_BIN" "$RUNNER" run --jobs "$JOBS" --responses "$RESPONSES" --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model-id "$MODEL" --prompt-article "$ARTICLE_PROMPT" --prompt-jp "$JP_PROMPT" --max-workers 2
"$PYTHON_BIN" "$RUNNER" materialize --questions "$QUESTIONS" --jobs "$JOBS" --responses "$RESPONSES" --out-dir "$MATERIALIZED"

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
    "experiment_id": "E029", "shard": sys.argv[2], "execution_manifest_sha256": sys.argv[3],
    "runner_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "jobs_sha256": hashlib.sha256(Path(sys.argv[5]).read_bytes()).hexdigest(),
    "materialization_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip(),
}
(root / "run_receipt.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

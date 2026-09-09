#!/usr/bin/env bash
# Start the frozen E030 model once, prove local service identity, and stop.
# This script never loads judge jobs and never emits a judgment.
#SBATCH --job-name=lkg-e030-v3-smoke
#SBATCH --partition=A100,L40S,H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -eEuo pipefail
: "${LKG_REPO:?Set LKG_REPO.}"
: "${LKG_DATA_ROOT:?Set the data root containing doctrine_v3plus_bench.}"
: "${E030_EXECUTION_MANIFEST:?Set absolute v3 manifest path.}"
: "${E030_EXECUTION_MANIFEST_SHA:?Set v3 manifest SHA-256.}"
: "${E030_SHARD:?Set one v3 retry shard id.}"
PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"
RUN_ROOT=""
SERVER_PID=""

write_failure() {
  local code="$1"
  [[ -n "$RUN_ROOT" ]] || return 0
  "$PYTHON_BIN" - "$RUN_ROOT/technical_failure_receipt.json" "$code" "${SLURM_JOB_ID:-manual}" "${HOSTNAME:-unknown}" "${PORT:-unknown}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"status":"fail","exit_code":int(sys.argv[2]),"slurm_job_id":sys.argv[3],"node":sys.argv[4],"port":sys.argv[5]}, indent=2)+"\n")
PY
}
cleanup() { code=$?; if [[ "$code" -ne 0 ]]; then write_failure "$code"; fi; [[ -z "$SERVER_PID" ]] || kill "$SERVER_PID" 2>/dev/null || true; exit "$code"; }
trap cleanup EXIT

[[ "$(sha256sum "$E030_EXECUTION_MANIFEST" | awk '{print $1}')" == "$E030_EXECUTION_MANIFEST_SHA" ]] || { echo "manifest hash mismatch" >&2; exit 2; }
eval "$("$PYTHON_BIN" - "$E030_EXECUTION_MANIFEST" "$E030_SHARD" <<'PY'
import json, shlex, sys
p=json.load(open(sys.argv[1])); s=next((x for x in p["shards"] if x["id"]==sys.argv[2]),None)
if p.get("status") != "frozen_preflight_only" or not s: raise SystemExit("unexpected E030 v3 manifest/shard")
m=p["model"]
for k,v in {"MODEL":m["id"],"MODEL_REVISION":m["revision"],"MODEL_SNAPSHOT":m["snapshot"],"PORT":s["vllm_port"],"OUT_REL":p["outputs"]["root"],"MAX_MODEL_TOKENS":m["max_model_tokens"],"MAX_NUM_SEQS":m["max_num_seqs"]}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
RUN_ROOT="$LKG_DATA_ROOT/$OUT_REL/smoke/$E030_SHARD"
mkdir -p "$RUN_ROOT"
[[ "$(basename "$(readlink -f "$MODEL_SNAPSHOT")")" == "$MODEL_REVISION" ]] || { echo "snapshot revision mismatch" >&2; exit 2; }
port_must_be_unbound() { ! ss -ltnH "sport = :$PORT" | grep -q .; }
port_must_be_unbound || { echo "reserved port already has a listener: $PORT" >&2; exit 2; }
SERVICE_MODEL="${MODEL}@${MODEL_REVISION}"
"$VLLM_BIN" serve "$MODEL_SNAPSHOT" --served-model-name "$SERVICE_MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_TOKENS" --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.90 > "$RUN_ROOT/vllm.log" 2>&1 & SERVER_PID=$!
for _ in $(seq 1 120); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$RUN_ROOT/vllm.log" >&2; exit 1; }; sleep 5; done
curl --fail --silent "http://127.0.0.1:$PORT/v1/models" > "$RUN_ROOT/v1_models.json"
"$PYTHON_BIN" - "$RUN_ROOT" "$SERVICE_MODEL" "$MODEL" "$MODEL_REVISION" "$PORT" <<'PY'
import json, os, subprocess, sys
root, served, model, revision, port = sys.argv[1:]
models=json.load(open(os.path.join(root,"v1_models.json")))
if served not in [x.get("id") for x in models.get("data",[])]: raise SystemExit("/v1/models identity mismatch")
payload={"status":"pass","node":os.environ.get("HOSTNAME"),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"port":int(port),"listener":"127.0.0.1","gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],text=True).strip(),"model_id":model,"model_revision":revision,"served_model_id":served,"v1_models":models}
open(os.path.join(root,"smoke_receipt.json"),"w").write(json.dumps(payload,indent=2)+"\n")
PY

#!/usr/bin/env bash
# Runs one v3 retry shard. Any HTTP failure exits before a score is materialized.
#SBATCH --job-name=lkg-e030-v3
#SBATCH --partition=A100,L40S,H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -eEuo pipefail
: "${LKG_REPO:?}"; : "${LKG_DATA_ROOT:?}"; : "${E030_EXECUTION_MANIFEST:?}"; : "${E030_EXECUTION_MANIFEST_SHA:?}"; : "${E030_SHARD:?}"; : "${E030_SMOKE_RECEIPT:?}"; : "${E030_ALLOW_FULL_RETRY:?}"
[[ "$E030_ALLOW_FULL_RETRY" == 1 ]] || { echo "E030_ALLOW_FULL_RETRY=1 required" >&2; exit 2; }
PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"; VLLM_BIN="${VLLM_BIN:-${PYTHON_BIN%/python}/vllm}"; ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"; RUNNER="$ROOT/scripts/121_run_b2_e030_judge_fail_closed_v3.py"; PORT=""; RUN_ROOT=""; SERVER_PID=""
failure() { code=$?; if [[ -n "$RUN_ROOT" && "$code" -ne 0 ]]; then printf '{"status":"fail","exit_code":%s,"node":"%s","slurm_job_id":"%s","port":"%s"}\n' "$code" "${HOSTNAME:-unknown}" "${SLURM_JOB_ID:-manual}" "${PORT:-unknown}" > "$RUN_ROOT/technical_failure_receipt.json"; fi; [[ -z "$SERVER_PID" ]] || kill "$SERVER_PID" 2>/dev/null || true; exit "$code"; }; trap failure EXIT
[[ "$(sha256sum "$E030_EXECUTION_MANIFEST"|awk '{print $1}')" == "$E030_EXECUTION_MANIFEST_SHA" ]] || exit 2
eval "$("$PYTHON_BIN" - "$E030_EXECUTION_MANIFEST" "$E030_SHARD" "$E030_SMOKE_RECEIPT" <<'PY'
import hashlib,json,shlex,sys
p=json.load(open(sys.argv[1])); s=next((x for x in p['shards'] if x['id']==sys.argv[2]),None); smoke=json.load(open(sys.argv[3]))
if not s or smoke.get('manifest_sha256')!=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest() or smoke.get('model_calls')!=0: raise SystemExit('smoke receipt mismatch')
m=p['model']; prompt=p['prompts'][s['modality']]
for k,v in {'MODEL':m['id'],'MODEL_REVISION':m['revision'],'SNAPSHOT':m['snapshot'],'PORT':s['vllm_port'],'OUT':p['outputs']['root'],'JOBS':s['jobs']['path'],'JOBS_SHA':s['jobs']['sha256'],'PROMPT':prompt['path'],'PROMPT_SHA':prompt['sha256'],'MAXLEN':m['max_model_tokens'],'MAXTOK':m['max_output_tokens'],'WORKERS':m['max_workers']}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
RUN_ROOT="$LKG_DATA_ROOT/$OUT/shards/$E030_SHARD"; mkdir -p "$RUN_ROOT"; JOBS="$LKG_DATA_ROOT/$JOBS"; PROMPT="$LKG_REPO/$PROMPT"
[[ -f "$JOBS" && -f "$PROMPT" && -x "$PYTHON_BIN" && -x "$VLLM_BIN" ]] || exit 2; [[ "$(sha256sum "$JOBS"|awk '{print $1}')" == "$JOBS_SHA" && "$(sha256sum "$PROMPT"|awk '{print $1}')" == "$PROMPT_SHA" ]] || exit 2; [[ "$(basename "$(readlink -f "$SNAPSHOT")")" == "$MODEL_REVISION" ]] || exit 2
port_must_be_unbound() { ! ss -ltnH "sport = :$PORT" | grep -q .; }; port_must_be_unbound || exit 2
SERVICE_MODEL="$MODEL@$MODEL_REVISION"; "$VLLM_BIN" serve "$SNAPSHOT" --served-model-name "$SERVICE_MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len "$MAXLEN" --max-num-seqs 2 --gpu-memory-utilization .90 > "$RUN_ROOT/vllm.log" 2>&1 & SERVER_PID=$!
for _ in $(seq 1 120); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 "$SERVER_PID" 2>/dev/null || exit 1; sleep 5; done
curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$RUN_ROOT/v1_models.json"; "$PYTHON_BIN" - "$RUN_ROOT/v1_models.json" "$SERVICE_MODEL" <<'PY'
import json,sys
if sys.argv[2] not in [x.get('id') for x in json.load(open(sys.argv[1])).get('data',[])]: raise SystemExit('/v1/models identity mismatch')
PY
"$PYTHON_BIN" "$RUNNER" --jobs "$JOBS" --responses "$RUN_ROOT/responses.jsonl" --endpoint "http://127.0.0.1:$PORT/v1" --frozen-model "$MODEL" --served-model "$SERVICE_MODEL" --prompt "$PROMPT" --max-tokens "$MAXTOK" --max-workers "$WORKERS"
"$PYTHON_BIN" - "$JOBS" "$RUN_ROOT/responses.jsonl" <<'PY'
import json,sys
jobs=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
if len(jobs)!=len(rows): raise SystemExit('response coverage incomplete; no materialization')
PY
printf '{"status":"complete_unaggregated","node":"%s","slurm_job_id":"%s","port":%s,"listener":"127.0.0.1","model_id":"%s","model_revision":"%s","served_model_id":"%s"}\n' "${HOSTNAME:-unknown}" "${SLURM_JOB_ID:-manual}" "$PORT" "$MODEL" "$MODEL_REVISION" "$SERVICE_MODEL" > "$RUN_ROOT/run_receipt.json"

#!/usr/bin/env bash
# CPU-only preflight for the immutable E029 output-budget successor.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the isolated reproducibility checkout.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E029_PREFLIGHT_MANIFEST:?Set the absolute successor preflight manifest path.}"
: "${E029_PREFLIGHT_MANIFEST_SHA:?Pass the successor preflight manifest SHA-256.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
CONTEXT_AUDITOR="$ROOT/scripts/103_audit_b2_fulltext_context.py"
COMPLETION_AUDITOR="$ROOT/scripts/112_audit_b2_e029_completion_budget.py"
AGGREGATOR="$ROOT/scripts/105_aggregate_b2_fulltext_context_audits.py"

for path in "$PYTHON_BIN" "$E029_PREFLIGHT_MANIFEST" "$CONTEXT_AUDITOR" "$COMPLETION_AUDITOR" "$AGGREGATOR"; do
  [[ -e "$path" ]] || { echo "missing required preflight path: $path" >&2; exit 2; }
done
actual_manifest_sha="$(sha256sum "$E029_PREFLIGHT_MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$E029_PREFLIGHT_MANIFEST_SHA" ]] || { echo "successor preflight manifest SHA mismatch" >&2; exit 2; }

eval "$("$PYTHON_BIN" - "$E029_PREFLIGHT_MANIFEST" <<'PY'
import json
import shlex
import sys

p=json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("experiment_id") != "E029" or p.get("status") != "authorized_cpu_preflight_no_model_call_v2":
    raise SystemExit("unexpected E029 successor preflight manifest")
if p["model"].get("temperature") != 0 or int(p["model"].get("max_output_tokens", 0)) != 512:
    raise SystemExit("unexpected successor generation budget")
if p["reranking_contract"].get("depth_grid") != [10,20,30,40,50,60,70]:
    raise SystemExit("unexpected successor depth grid")

def emit(name, value): print(f"{name}={shlex.quote(str(value))}")
emit("MODEL", p["model"]["id"])
emit("REVISION", p["model"]["revision"])
emit("SNAPSHOT", p["model"]["snapshot"])
emit("MAX_OUTPUT", p["model"]["max_output_tokens"])
emit("ARTICLE_PROMPT", p["prompts"]["article"]["path"])
emit("ARTICLE_PROMPT_SHA", p["prompts"]["article"]["sha256"])
emit("JP_PROMPT", p["prompts"]["jp"]["path"])
emit("JP_PROMPT_SHA", p["prompts"]["jp"]["sha256"])
emit("OUT_REL", p["outputs"]["root"])
emit("EXPECTED_CONDITIONS", p["totals"]["conditions"])
emit("EXPECTED_JOBS", p["totals"]["model_calls"])
emit("CONTEXT_AUDITOR_SHA", p["code_bundle"]["context_auditor"]["sha256"])
emit("COMPLETION_AUDITOR_SHA", p["code_bundle"]["completion_auditor"]["sha256"])
emit("AGGREGATOR_SHA", p["code_bundle"]["context_aggregator"]["sha256"])
for shard in p["shards"]:
    emit(f"JOBS_{shard['id'].replace('-', '_').upper()}", shard["jobs"]["path"])
    emit(f"JOBS_SHA_{shard['id'].replace('-', '_').upper()}", shard["jobs"]["sha256"])
    emit(f"JOBS_COUNT_{shard['id'].replace('-', '_').upper()}", shard["jobs"]["count"])
PY
)"

ARTICLE_PROMPT="$LKG_REPO/$ARTICLE_PROMPT"
JP_PROMPT="$LKG_REPO/$JP_PROMPT"
OUT_ROOT="$LKG_DATA_ROOT/$OUT_REL"
for path in "$SNAPSHOT" "$ARTICLE_PROMPT" "$JP_PROMPT"; do
  [[ -e "$path" ]] || { echo "missing frozen successor input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$ARTICLE_PROMPT" | awk '{print $1}')" == "$ARTICLE_PROMPT_SHA" ]] || { echo "Article prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_PROMPT" | awk '{print $1}')" == "$JP_PROMPT_SHA" ]] || { echo "JP prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$CONTEXT_AUDITOR" | awk '{print $1}')" == "$CONTEXT_AUDITOR_SHA" ]] || { echo "context auditor SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$COMPLETION_AUDITOR" | awk '{print $1}')" == "$COMPLETION_AUDITOR_SHA" ]] || { echo "completion auditor SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AGGREGATOR" | awk '{print $1}')" == "$AGGREGATOR_SHA" ]] || { echo "context aggregator SHA mismatch" >&2; exit 2; }
[[ ! -e "$OUT_ROOT" ]] || { echo "refusing to overwrite immutable successor preflight: $OUT_ROOT" >&2; exit 2; }

declare -a SHARDS=(cosine ppr lightgcn-seed42 lightgcn-seed43 lightgcn-seed44)
declare -a JOBS=()
for shard in "${SHARDS[@]}"; do
  variable="${shard//-/_}"; variable="${variable^^}"
  jobs_rel="$(eval "printf '%s' \"\${JOBS_${variable}}\"")"
  jobs_sha="$(eval "printf '%s' \"\${JOBS_SHA_${variable}}\"")"
  jobs_count="$(eval "printf '%s' \"\${JOBS_COUNT_${variable}}\"")"
  jobs="$LKG_DATA_ROOT/$jobs_rel"
  [[ -e "$jobs" ]] || { echo "missing frozen jobs: $jobs" >&2; exit 2; }
  [[ "$(sha256sum "$jobs" | awk '{print $1}')" == "$jobs_sha" ]] || { echo "jobs SHA mismatch: $shard" >&2; exit 2; }
  [[ "$(grep -cve '^[[:space:]]*$' "$jobs")" == "$jobs_count" ]] || { echo "jobs count mismatch: $shard" >&2; exit 2; }
  JOBS+=("$jobs")
done

mkdir -p "$OUT_ROOT/audits"
for index in "${!SHARDS[@]}"; do
  shard="${SHARDS[$index]}"
  "$PYTHON_BIN" "$CONTEXT_AUDITOR" --jobs "${JOBS[$index]}" --prompt-article "$ARTICLE_PROMPT" --prompt-jp "$JP_PROMPT" \
    --model-id "$MODEL" --model-revision "$REVISION" --model-snapshot "$SNAPSHOT" --context-limit-tokens 16384 \
    --max-output-tokens "$MAX_OUTPUT" --expected-questions 754 --output "$OUT_ROOT/audits/${shard}.json"
done

aggregate_args=()
for shard in "${SHARDS[@]}"; do aggregate_args+=(--report "$OUT_ROOT/audits/${shard}.json"); done
"$PYTHON_BIN" "$AGGREGATOR" "${aggregate_args[@]}" --expected-conditions "$EXPECTED_CONDITIONS" --output "$OUT_ROOT/context_audit.json"

completion_args=()
for jobs in "${JOBS[@]}"; do completion_args+=(--jobs "$jobs"); done
"$PYTHON_BIN" "$COMPLETION_AUDITOR" "${completion_args[@]}" --model-snapshot "$SNAPSHOT" --expected-questions 754 \
  --max-output-tokens "$MAX_OUTPUT" --output "$OUT_ROOT/completion_budget_audit.json"

"$PYTHON_BIN" - "$OUT_ROOT" "$actual_manifest_sha" "$EXPECTED_CONDITIONS" "$EXPECTED_JOBS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root=Path(sys.argv[1])
manifest_sha, expected_conditions, expected_jobs = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
context=json.load((root / "context_audit.json").open(encoding="utf-8"))
completion=json.load((root / "completion_budget_audit.json").open(encoding="utf-8"))
if context["compatible_conditions"] != expected_conditions or context["overflow_questions"] != 0:
    raise SystemExit("context audit did not clear the successor matrix")
if completion["compatible_conditions"] != expected_conditions or completion["incompatible_conditions"] != 0:
    raise SystemExit("completion audit did not clear the successor matrix")
receipt={
  "schema_version":"b2-e029-output512-preflight-receipt.v1", "status":"complete_no_model_call", "model_calls":0,
  "preflight_manifest_sha256":manifest_sha, "expected_conditions":expected_conditions, "expected_jobs":expected_jobs,
  "context_audit_sha256":hashlib.sha256((root / "context_audit.json").read_bytes()).hexdigest(),
  "completion_budget_audit_sha256":hashlib.sha256((root / "completion_budget_audit.json").read_bytes()).hexdigest(),
}
(root / "preflight_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
PY

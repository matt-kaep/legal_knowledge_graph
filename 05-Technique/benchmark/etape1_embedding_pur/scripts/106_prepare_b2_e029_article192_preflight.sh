#!/usr/bin/env bash
# Materialize and audit the approved E029 Article-192 / JP-synthese jobs.
# This CPU-only gate never contacts a model endpoint.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the reproducibility checkout.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E029_PREFLIGHT_MANIFEST_SHA:?Pass the SHA-256 of the frozen preflight manifest.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
MANIFEST="$ROOT/configs/b2_reranking_comparable_a3_k70_article192_preflight_v1.json"
RUNNER="$ROOT/scripts/102_run_b2_comparable_reranking.py"
AUDITOR="$ROOT/scripts/103_audit_b2_fulltext_context.py"
AGGREGATOR="$ROOT/scripts/105_aggregate_b2_fulltext_context_audits.py"

for path in "$PYTHON_BIN" "$MANIFEST" "$RUNNER" "$AUDITOR" "$AGGREGATOR"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$E029_PREFLIGHT_MANIFEST_SHA" ]] || {
  echo "E029 preflight manifest SHA-256 mismatch" >&2; exit 2;
}

eval "$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import shlex
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest["experiment_id"] != "E029" or manifest["status"] != "authorized_cpu_preflight_no_model_call":
    raise SystemExit("unexpected E029 preflight manifest identity or status")
contract = manifest["reranking_contract"]
if contract["depth_grid"] != [10, 20, 30, 40, 50, 60, 70] or contract["k_out"] != 10:
    raise SystemExit("unexpected E029 depth grid or K_out")
article = contract["representation"]["article"]
jp = contract["representation"]["jp"]
if article != {"source_field": "texte", "projection": "token_prefix", "token_cap": 192}:
    raise SystemExit("unexpected Article representation contract")
if jp != {"source_field": "synthese", "projection": "complete_unmodified"}:
    raise SystemExit("unexpected JP representation contract")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

for name, value in {
    "MODEL": manifest["model"]["id"],
    "REVISION": manifest["model"]["revision"],
    "A3_REL": manifest["a3"]["manifest_path"],
    "A3_SHA": manifest["a3"]["sha256"],
    "ARTICLE_PROMPT_REL": manifest["prompts"]["article"]["path"],
    "ARTICLE_PROMPT_SHA": manifest["prompts"]["article"]["sha256"],
    "JP_PROMPT_REL": manifest["prompts"]["jp"]["path"],
    "JP_PROMPT_SHA": manifest["prompts"]["jp"]["sha256"],
    "SOURCE_RECEIPT_REL": manifest["source_pools"]["receipt_path"],
    "SOURCE_RECEIPT_SHA": manifest["source_pools"]["receipt_sha256"],
    "SOURCE_POOLS_REL": manifest["source_pools"]["pools_root"],
    "OUT_REL": manifest["outputs"]["root"],
    "MODEL_SNAPSHOT_REL": manifest["model"]["tokenizer_snapshot"],
    "EXPECTED_CONDITIONS": manifest["totals"]["conditions"],
    "EXPECTED_JOBS": manifest["totals"]["model_calls"],
    "RUNNER_SHA": manifest["code_bundle"]["reranking_runner"]["sha256"],
    "AUDITOR_SHA": manifest["code_bundle"]["context_auditor"]["sha256"],
    "AGGREGATOR_SHA": manifest["code_bundle"]["context_audit_aggregator"]["sha256"],
}.items():
    emit(name, value)
PY
)"

A3="$LKG_REPO/$A3_REL"
ARTICLE_PROMPT="$LKG_REPO/$ARTICLE_PROMPT_REL"
JP_PROMPT="$LKG_REPO/$JP_PROMPT_REL"
SOURCE_RECEIPT="$LKG_DATA_ROOT/$SOURCE_RECEIPT_REL"
SOURCE_POOLS="$LKG_DATA_ROOT/$SOURCE_POOLS_REL"
OUT_ROOT="$LKG_DATA_ROOT/$OUT_REL"
MODEL_SNAPSHOT="$MODEL_SNAPSHOT_REL"
for path in "$A3" "$ARTICLE_PROMPT" "$JP_PROMPT" "$SOURCE_RECEIPT" "$SOURCE_POOLS" "$MODEL_SNAPSHOT"; do
  [[ -e "$path" ]] || { echo "missing frozen E029 preflight input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$A3" | awk '{print $1}')" == "$A3_SHA" ]] || { echo "A3 SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$ARTICLE_PROMPT" | awk '{print $1}')" == "$ARTICLE_PROMPT_SHA" ]] || { echo "Article prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_PROMPT" | awk '{print $1}')" == "$JP_PROMPT_SHA" ]] || { echo "JP prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_RECEIPT" | awk '{print $1}')" == "$SOURCE_RECEIPT_SHA" ]] || { echo "source-pool receipt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA" ]] || { echo "runner SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AUDITOR" | awk '{print $1}')" == "$AUDITOR_SHA" ]] || { echo "auditor SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AGGREGATOR" | awk '{print $1}')" == "$AGGREGATOR_SHA" ]] || { echo "aggregator SHA mismatch" >&2; exit 2; }

[[ ! -e "$OUT_ROOT" ]] || { echo "refusing to overwrite immutable E029 preflight: $OUT_ROOT" >&2; exit 2; }
mkdir -p "$OUT_ROOT/jobs" "$OUT_ROOT/audits"

"$PYTHON_BIN" - "$SOURCE_RECEIPT" "$SOURCE_POOLS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, pools_root = map(Path, sys.argv[1:])
receipt = json.load(receipt_path.open(encoding="utf-8"))
if receipt.get("status") != "complete" or receipt.get("completed_conditions") != 100:
    raise SystemExit("source pool receipt is not a complete 100-condition matrix")
want_depths = {10, 20, 30, 40, 50, 60, 70}
expected = []
for row in receipt.get("pools", []):
    if int(row["k_in"]) not in want_depths:
        continue
    path = Path(row["path"])
    if path.parent != pools_root:
        raise SystemExit(f"unexpected source pool location: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != row["sha256"]:
        raise SystemExit(f"source pool hash mismatch: {path}")
    expected.append(row)
if len(expected) != 70 or sum(int(row["questions"]) for row in expected) != 52780:
    raise SystemExit(f"unexpected selected source pool coverage: {len(expected)} conditions")
PY

prepare_shard() {
  local shard="$1"
  shift
  local jobs="$OUT_ROOT/jobs/${shard}.jsonl"
  local audit="$OUT_ROOT/audits/${shard}.json"
  local args=()
  local pool
  for pool in "$@"; do args+=(--pool "$pool"); done
  "$PYTHON_BIN" "$RUNNER" prepare "${args[@]}" \
    --prompt-article "$ARTICLE_PROMPT" --prompt-jp "$JP_PROMPT" --output "$jobs" \
    --model-id "$MODEL" --model-revision "$REVISION" --article-token-cap 192 \
    --tokenizer-snapshot "$MODEL_SNAPSHOT"
  "$PYTHON_BIN" "$AUDITOR" --jobs "$jobs" --prompt-article "$ARTICLE_PROMPT" --prompt-jp "$JP_PROMPT" \
    --model-id "$MODEL" --model-revision "$REVISION" --model-snapshot "$MODEL_SNAPSHOT" \
    --context-limit-tokens 16384 --max-output-tokens 256 --expected-questions 754 --output "$audit"
}

mapfile -t COSINE < <(find "$SOURCE_POOLS" -maxdepth 1 -type f \( -name 'cosine_article_kin*.jsonl' -o -name 'cosine_jp_kin*.jsonl' \) | awk '/kin(10|20|30|40|50|60|70)\.jsonl$/' | sort)
mapfile -t PPR < <(find "$SOURCE_POOLS" -maxdepth 1 -type f \( -name 'ppr_g6_article_kin*.jsonl' -o -name 'ppr_g6_jp_kin*.jsonl' \) | awk '/kin(10|20|30|40|50|60|70)\.jsonl$/' | sort)
for seed in 42 43 44; do
  mapfile -t "LGC_${seed}" < <(find "$SOURCE_POOLS" -maxdepth 1 -type f \( -name "lightgcn_g6_article_kin*_seed${seed}.jsonl" -o -name "lightgcn_g6_jp_kin*_seed${seed}.jsonl" \) | awk '/kin(10|20|30|40|50|60|70)_seed/' | sort)
done
[[ ${#COSINE[@]} -eq 14 && ${#PPR[@]} -eq 14 && ${#LGC_42[@]} -eq 14 && ${#LGC_43[@]} -eq 14 && ${#LGC_44[@]} -eq 14 ]] || {
  echo "unexpected selected source-pool shard cardinality" >&2; exit 2;
}

prepare_shard cosine "${COSINE[@]}"
prepare_shard ppr "${PPR[@]}"
prepare_shard lightgcn-seed42 "${LGC_42[@]}"
prepare_shard lightgcn-seed43 "${LGC_43[@]}"
prepare_shard lightgcn-seed44 "${LGC_44[@]}"

"$PYTHON_BIN" "$AGGREGATOR" \
  --report "$OUT_ROOT/audits/cosine.json" --report "$OUT_ROOT/audits/ppr.json" \
  --report "$OUT_ROOT/audits/lightgcn-seed42.json" --report "$OUT_ROOT/audits/lightgcn-seed43.json" \
  --report "$OUT_ROOT/audits/lightgcn-seed44.json" --expected-conditions "$EXPECTED_CONDITIONS" \
  --output "$OUT_ROOT/context_audit.json"

"$PYTHON_BIN" - "$OUT_ROOT" "$EXPECTED_JOBS" "$EXPECTED_CONDITIONS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_jobs, expected_conditions = map(int, sys.argv[2:])
audit = json.load((root / "context_audit.json").open(encoding="utf-8"))
if audit["compatible_conditions"] != expected_conditions or audit["overflow_questions"] != 0:
    raise SystemExit("context audit did not clear every frozen condition")
jobs = sorted((root / "jobs").glob("*.jsonl"))
counts = {path.name: sum(1 for line in path.open(encoding="utf-8") if line.strip()) for path in jobs}
if len(jobs) != 5 or sum(counts.values()) != expected_jobs:
    raise SystemExit(f"unexpected prepared-job coverage: {counts}")
receipt = {
    "schema_version": "b2-e029-k70-article192-preflight-receipt.v1",
    "status": "complete_no_model_call",
    "model_calls": 0,
    "jobs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": counts[path.name]} for path in jobs],
    "context_audit": {"path": str(root / "context_audit.json"), "sha256": hashlib.sha256((root / "context_audit.json").read_bytes()).hexdigest()},
}
(root / "preflight_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

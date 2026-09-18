#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e029-g6-preflight
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# E029 preparation only: materialize frozen G6 LightGCN pools and count exact
# prompt tokens. This script never starts vLLM and never calls a reranker.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the reproducibility checkout.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E029_G6_PREFLIGHT_MANIFEST_SHA:?Pass the SHA-256 of the frozen E029 G6 preflight manifest.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
MANIFEST="$ROOT/configs/b2_reranking_comparable_a3_lightgcn_g6_preflight_v1.json"
MATERIALIZER="$ROOT/scripts/101_materialize_b2_reranking_pools.py"
AUDITOR="$ROOT/scripts/103_audit_b2_fulltext_context.py"

for path in "$PYTHON_BIN" "$MANIFEST" "$MATERIALIZER" "$AUDITOR"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$E029_G6_PREFLIGHT_MANIFEST_SHA" ]] || {
  echo "E029 G6 preflight manifest SHA-256 mismatch" >&2
  exit 2
}

eval "$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import shlex
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload["experiment_id"] != "E029" or payload["status"] != "authorized_cpu_preflight_no_model_call":
    raise SystemExit("unexpected E029 G6 preflight manifest identity or status")
if payload["reranking_contract"]["candidate_text_policy"] != "full_text_unmodified":
    raise SystemExit("candidate text policy must remain full_text_unmodified")
if payload["reranking_contract"]["k_out"] != 10:
    raise SystemExit("K_out must be exactly 10")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

a3 = payload["a3"]
sources = payload["frozen_sources"]
rankings = sources["lightgcn_g6_final_rankings"]
outputs = payload["outputs"]
code = payload["code_bundle"]
emit("A3_REL", a3["manifest_path"])
emit("A3_SHA", a3["sha256"])
emit("QUESTIONS_REL", a3["evaluation_path"])
emit("QUESTIONS_SHA", a3["evaluation_sha256"])
emit("EXPECTED_QUESTIONS", a3["evaluation_questions"])
emit("RANKING_REL", rankings["path"])
emit("RANKING_SHA", rankings["sha256"])
emit("ARTICLE_METHOD", rankings["article_method"])
emit("JP_METHOD", rankings["jurisprudence_method"])
emit("ARTICLE_TEXTS_REL", sources["article_texts"]["path"])
emit("ARTICLE_TEXTS_SHA", sources["article_texts"]["sha256"])
emit("JP_TEXTS_REL", sources["jurisprudence_texts"]["path"])
emit("JP_TEXTS_SHA", sources["jurisprudence_texts"]["sha256"])
emit("ARTICLE_PROMPT_REL", payload["prompts"]["article"]["path"])
emit("ARTICLE_PROMPT_SHA", payload["prompts"]["article"]["sha256"])
emit("JP_PROMPT_REL", payload["prompts"]["jurisprudence"]["path"])
emit("JP_PROMPT_SHA", payload["prompts"]["jurisprudence"]["sha256"])
emit("OUT_REL", outputs["root"])
emit("K_INS", " ".join(map(str, payload["reranking_contract"]["k_in_sweep"])))
emit("REPLAY_SEEDS", " ".join(map(str, rankings["replay_seeds"])))
emit("CONTEXT_LIMIT", payload["model"]["context_limit_tokens"])
emit("MAX_OUTPUT", payload["model"]["max_output_tokens"])
emit("MODEL_ID", payload["model"]["id"])
emit("MODEL_REVISION", payload["model"]["revision"])
emit("MATERIALIZER_SHA", code["pool_materializer"]["sha256"])
emit("AUDITOR_SHA", code["context_auditor"]["sha256"])
PY
)"

A3="$LKG_REPO/$A3_REL"
QUESTIONS="$LKG_DATA_ROOT/$QUESTIONS_REL"
RANKING="$LKG_DATA_ROOT/$RANKING_REL"
ARTICLE_TEXTS="$LKG_DATA_ROOT/$ARTICLE_TEXTS_REL"
JP_TEXTS="$LKG_DATA_ROOT/$JP_TEXTS_REL"
ARTICLE_PROMPT="$LKG_REPO/$ARTICLE_PROMPT_REL"
JP_PROMPT="$LKG_REPO/$JP_PROMPT_REL"
OUT_ROOT="$LKG_DATA_ROOT/$OUT_REL"
MODEL_SNAPSHOT="${E029_MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/$MODEL_REVISION}"

for path in "$A3" "$QUESTIONS" "$RANKING" "$ARTICLE_TEXTS" "$JP_TEXTS" "$ARTICLE_PROMPT" "$JP_PROMPT" "$MODEL_SNAPSHOT"; do
  [[ -e "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
done

[[ "$(sha256sum "$A3" | awk '{print $1}')" == "$A3_SHA" ]] || { echo "A3 SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$QUESTIONS" | awk '{print $1}')" == "$QUESTIONS_SHA" ]] || { echo "evaluation SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$RANKING" | awk '{print $1}')" == "$RANKING_SHA" ]] || { echo "LightGCN ranking SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$ARTICLE_TEXTS" | awk '{print $1}')" == "$ARTICLE_TEXTS_SHA" ]] || { echo "article texts SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_TEXTS" | awk '{print $1}')" == "$JP_TEXTS_SHA" ]] || { echo "JP texts SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$ARTICLE_PROMPT" | awk '{print $1}')" == "$ARTICLE_PROMPT_SHA" ]] || { echo "article prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_PROMPT" | awk '{print $1}')" == "$JP_PROMPT_SHA" ]] || { echo "JP prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$MATERIALIZER" | awk '{print $1}')" == "$MATERIALIZER_SHA" ]] || { echo "materializer SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AUDITOR" | awk '{print $1}')" == "$AUDITOR_SHA" ]] || { echo "auditor SHA mismatch" >&2; exit 2; }

mkdir -p "$OUT_ROOT/pools"
exec > >(tee -a "$OUT_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
export LKG_REPO LKG_DATA_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[[ ! -e "$OUT_ROOT/preflight_receipt.json" ]] || {
  echo "E029 G6 preflight receipt already exists; leaving immutable output untouched."
  exit 0
}

for replay_seed in $REPLAY_SEEDS; do
  for modality in article jp; do
    if [[ "$modality" == article ]]; then
      method="$ARTICLE_METHOD"
      texts="$ARTICLE_TEXTS"
    else
      method="$JP_METHOD"
      texts="$JP_TEXTS"
    fi
    for k_in in $K_INS; do
      pool="$OUT_ROOT/pools/lightgcn_${modality}_seed${replay_seed}_k${k_in}.jsonl"
      if [[ -f "$pool" ]]; then
        lines="$(awk 'NF {count += 1} END {print count + 0}' "$pool")"
        [[ "$lines" == "$EXPECTED_QUESTIONS" ]] || { echo "partial immutable pool: $pool ($lines/$EXPECTED_QUESTIONS)" >&2; exit 1; }
        continue
      fi
      "$PYTHON_BIN" "$MATERIALIZER" \
        --ranking "$RANKING" --questions "$QUESTIONS" --texts "$texts" --output "$pool" \
        --family lightgcn --modality "$modality" --a3-manifest "$A3" --method "$method" \
        --replay-seed "$replay_seed" --k-in "$k_in"
    done
  done

  audit="$OUT_ROOT/fulltext_context_audit_lightgcn_g6_seed${replay_seed}.json"
  if [[ ! -f "$audit" ]]; then
    audit_args=()
    for modality in article jp; do
      for k_in in $K_INS; do
        audit_args+=(--jobs "$OUT_ROOT/pools/lightgcn_${modality}_seed${replay_seed}_k${k_in}.jsonl")
      done
    done
    "$PYTHON_BIN" "$AUDITOR" "${audit_args[@]}" \
      --prompt-article "$ARTICLE_PROMPT" --prompt-jp "$JP_PROMPT" \
      --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" --model-snapshot "$MODEL_SNAPSHOT" \
      --context-limit-tokens "$CONTEXT_LIMIT" --max-output-tokens "$MAX_OUTPUT" \
      --expected-questions "$EXPECTED_QUESTIONS" --output "$audit"
  fi
done

"$PYTHON_BIN" - "$OUT_ROOT" "$actual_manifest_sha" "$MATERIALIZER" "$AUDITOR" "$REPLAY_SEEDS" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = sys.argv[5].split()
audits = []
for seed in seeds:
    path = root / f"fulltext_context_audit_lightgcn_g6_seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", [])
    if len(conditions) != 20 or any(row.get("questions") != 754 for row in conditions):
        raise SystemExit(f"invalid audit coverage for seed {seed}")
    audits.append({"seed": int(seed), "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "conditions": conditions})
receipt = root / "preflight_receipt.json"
if receipt.exists():
    raise SystemExit("refusing to overwrite immutable preflight receipt")
receipt.write_text(json.dumps({
    "experiment_id": "E029",
    "kind": "LightGCN_G6_fulltext_context_preflight",
    "manifest_sha256": sys.argv[2],
    "materializer_sha256": hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),
    "auditor_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
    "seeds": audits,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "model_calls": 0,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

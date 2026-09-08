#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e029-prepare-jobs
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Freeze E029 jobs after checking every selected full-text pool against its
# completed CPU audit receipt.  This is deliberately CPU-only: no vLLM binary
# is referenced by this script.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the reproducibility checkout.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout.}"
: "${E029_JOBS_PREPARATION_MANIFEST_SHA:?Pass the SHA-256 of the frozen manifest.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
MANIFEST_NAME="${E029_JOBS_PREPARATION_MANIFEST_FILENAME:-b2_reranking_comparable_a3_jobs_preparation_v1.json}"
MANIFEST="$ROOT/configs/$MANIFEST_NAME"
RUNNER="$ROOT/scripts/102_run_b2_comparable_reranking.py"
POOL_RESOLVER="$ROOT/scripts/e029_audited_pool_resolver.py"

for path in "$PYTHON_BIN" "$MANIFEST" "$RUNNER" "$POOL_RESOLVER"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$E029_JOBS_PREPARATION_MANIFEST_SHA" ]] || {
  echo "E029 jobs-preparation manifest SHA-256 mismatch" >&2
  exit 2
}

eval "$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import shlex
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload["experiment_id"] != "E029" or payload["status"] not in {
    "authorized_cpu_jobs_preparation_no_model_call",
    "authorized_cpu_jobs_preparation_no_model_call_successor_v2",
}:
    raise SystemExit("unexpected E029 jobs-preparation manifest identity or status")
if payload["reranking_contract"]["candidate_text_policy"] != "full_text_unmodified":
    raise SystemExit("candidate text policy must remain full_text_unmodified")
if payload["reranking_contract"]["truncation"] != "forbidden":
    raise SystemExit("truncation must remain forbidden")
if payload["reranking_contract"]["k_out"] != 10:
    raise SystemExit("K_out must be exactly 10")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

a3 = payload["a3"]
inputs = payload["inputs"]
emit("A3_REL", a3["manifest_path"])
emit("A3_SHA", a3["sha256"])
emit("QUESTIONS_REL", a3["evaluation_path"])
emit("QUESTIONS_SHA", a3["evaluation_sha256"])
emit("AUDIT_REL", inputs["cosine_ppr_audit"]["path"])
emit("AUDIT_SHA", inputs["cosine_ppr_audit"]["sha256"])
emit("SEEDED_RECEIPT_REL", inputs["lightgcn_seeded_preflight"]["path"])
emit("SEEDED_RECEIPT_SHA", inputs["lightgcn_seeded_preflight"]["sha256"])
emit("COSINE_PPR_ROOT_REL", inputs["pool_roots"]["cosine_ppr"])
emit("LIGHTGCN_ROOT_REL", inputs["pool_roots"]["lightgcn_seeded"])
emit("ARTICLE_PROMPT_REL", payload["prompts"]["article"]["path"])
emit("ARTICLE_PROMPT_SHA", payload["prompts"]["article"]["sha256"])
emit("JP_PROMPT_REL", payload["prompts"]["jp"]["path"])
emit("JP_PROMPT_SHA", payload["prompts"]["jp"]["sha256"])
emit("OUT_REL", payload["outputs"]["root"])
emit("MODEL", payload["model"]["id"])
emit("REVISION", payload["model"]["revision"])
emit("RUNNER_SHA", payload["code_bundle"]["reranking_runner"]["sha256"])
emit("POOL_RESOLVER_SHA", payload["code_bundle"]["audited_pool_resolver"]["sha256"])
PY
)"

A3="$LKG_REPO/$A3_REL"
QUESTIONS="$LKG_DATA_ROOT/$QUESTIONS_REL"
AUDIT="$LKG_DATA_ROOT/$AUDIT_REL"
SEEDED_RECEIPT="$LKG_DATA_ROOT/$SEEDED_RECEIPT_REL"
COSINE_PPR_ROOT="$LKG_DATA_ROOT/$COSINE_PPR_ROOT_REL"
LIGHTGCN_ROOT="$LKG_DATA_ROOT/$LIGHTGCN_ROOT_REL"
ARTICLE_PROMPT="$LKG_REPO/$ARTICLE_PROMPT_REL"
JP_PROMPT="$LKG_REPO/$JP_PROMPT_REL"
OUT_ROOT="$LKG_DATA_ROOT/$OUT_REL"

for path in "$A3" "$QUESTIONS" "$AUDIT" "$SEEDED_RECEIPT" "$COSINE_PPR_ROOT" "$LIGHTGCN_ROOT" "$ARTICLE_PROMPT" "$JP_PROMPT"; do
  [[ -e "$path" ]] || { echo "missing frozen E029 input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$A3" | awk '{print $1}')" == "$A3_SHA" ]] || { echo "A3 SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$QUESTIONS" | awk '{print $1}')" == "$QUESTIONS_SHA" ]] || { echo "evaluation SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$AUDIT" | awk '{print $1}')" == "$AUDIT_SHA" ]] || { echo "cosine/PPR audit SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$SEEDED_RECEIPT" | awk '{print $1}')" == "$SEEDED_RECEIPT_SHA" ]] || { echo "seeded preflight receipt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$ARTICLE_PROMPT" | awk '{print $1}')" == "$ARTICLE_PROMPT_SHA" ]] || { echo "article prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$JP_PROMPT" | awk '{print $1}')" == "$JP_PROMPT_SHA" ]] || { echo "JP prompt SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA" ]] || { echo "reranking runner SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$POOL_RESOLVER" | awk '{print $1}')" == "$POOL_RESOLVER_SHA" ]] || { echo "audited pool resolver SHA mismatch" >&2; exit 2; }

mkdir -p "$OUT_ROOT/jobs"
exec > >(tee -a "$OUT_ROOT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
export LKG_REPO LKG_DATA_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[[ ! -e "$OUT_ROOT/jobs_preparation_receipt.json" ]] || {
  echo "E029 jobs receipt already exists; leaving immutable output untouched."
  exit 0
}

"$PYTHON_BIN" - "$MANIFEST" "$AUDIT" "$SEEDED_RECEIPT" "$LIGHTGCN_ROOT" "$OUT_ROOT" "$RUNNER" "$POOL_RESOLVER" "$LKG_DATA_ROOT" "$ARTICLE_PROMPT" "$JP_PROMPT" "$MODEL" "$REVISION" <<'PY'
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

manifest, audit_path, seeded_path, lightgcn_root, output_root, runner, resolver_path, data_root, article_prompt, jp_prompt = map(Path, sys.argv[1:11])
model, revision = sys.argv[11:13]
payload = json.loads(manifest.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
seeded = json.loads(seeded_path.read_text(encoding="utf-8"))
output_root.mkdir(parents=True, exist_ok=True)

resolver_spec = importlib.util.spec_from_file_location("e029_audited_pool_resolver", resolver_path)
if resolver_spec is None or resolver_spec.loader is None:
    raise SystemExit(f"cannot load audited pool resolver: {resolver_path}")
resolver_module = importlib.util.module_from_spec(resolver_spec)
resolver_spec.loader.exec_module(resolver_module)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

seeded_hashes = {}
for seed in seeded["seeds"]:
    for row in seed["pools"]:
        seeded_hashes[Path(row["path"]).name] = row["sha256"]

def cosine_ppr_pool(family, modality, k_in):
    try:
        return resolver_module.resolve_audited_pool(
            audit["job_files"],
            data_root=data_root,
            family=family,
            modality=modality,
            k_in=k_in,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

def lightgcn_pool(seed, modality, k_in):
    path = lightgcn_root / f"lightgcn_{modality}_seed{seed}_k{k_in}.jsonl"
    expected = seeded_hashes.get(path.name)
    if expected is None or sha256(path) != expected:
        raise SystemExit(f"unverified LightGCN pool: {path}")
    return path

def pools_for(shard):
    if shard["id"] == "cosine-jp":
        return [cosine_ppr_pool("cosine", "jp", k) for k in shard["k_in"]]
    if shard["id"] == "ppr":
        pools = []
        for condition in shard["conditions"]:
            pools.extend(cosine_ppr_pool("ppr", condition["modality"], k) for k in condition["k_in"])
        return pools
    seed = shard["replay_seed"]
    pools = []
    for condition in shard["conditions"]:
        pools.extend(lightgcn_pool(seed, condition["modality"], k) for k in condition["k_in"])
    return pools

receipt_shards = []
for shard in payload["shards"]:
    pools = pools_for(shard)
    if len(pools) != shard["expected_conditions"]:
        raise SystemExit(f"wrong condition count for {shard['id']}")
    output = output_root / "jobs" / f"{shard['id']}.jsonl"
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable jobs: {output}")
    command = [sys.executable, str(runner), "prepare"]
    for pool in pools:
        command += ["--pool", str(pool)]
    command += [
        "--prompt-article", str(article_prompt), "--prompt-jp", str(jp_prompt),
        "--output", str(output), "--model-id", model, "--model-revision", revision,
    ]
    subprocess.run(command, check=True)
    jobs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_jobs = int(shard["expected_jobs"])
    if len(jobs) != expected_jobs:
        raise SystemExit(f"wrong job count for {shard['id']}: {len(jobs)}/{expected_jobs}")
    keys = {(row["family"], row["modality"], int(row["k_in"]), str(row.get("replay_seed") or ""), row["qid"]) for row in jobs}
    if len(keys) != len(jobs):
        raise SystemExit(f"duplicate frozen job key in {shard['id']}")
    receipt_shards.append({
        "id": shard["id"], "conditions": shard["expected_conditions"], "jobs": len(jobs),
        "path": str(output), "sha256": sha256(output),
        "pools": [{"path": str(path), "sha256": sha256(path)} for path in pools],
    })

receipt = {
    "experiment_id": "E029", "kind": "comparable_reranking_frozen_jobs_preparation",
    "manifest_sha256": sha256(manifest), "runner_sha256": sha256(runner),
    "input_receipts": {"cosine_ppr_audit": sha256(audit_path), "lightgcn_seeded_preflight": sha256(seeded_path)},
    "shards": receipt_shards, "model_calls": 0,
}
(output_root / "jobs_preparation_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

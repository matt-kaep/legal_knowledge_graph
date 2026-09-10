#!/usr/bin/env bash
#SBATCH --job-name=lkg-ppr-audit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Audit only: it reads the existing PPR final outputs and writes a separate
# hash-bound audit report. It never starts PPR and never writes in _final_grouped_v2.
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the code checkout before submitting.}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the data checkout before submitting.}"

PYTHON_BIN="${LKG_PYTHON:-$HOME/work/.venv-benchmark/bin/python}"
ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
DATA_BENCH="$LKG_DATA_ROOT/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
CAMPAIGN_MANIFEST="$ROOT/configs/confirmatory_campaign_grouped_v2_repro_v1_cluster_node_runtime.json"
AUDIT_MANIFEST="$LKG_REPO/experiments/confirmatory-recovery/manifest_ppr_final_audit_v2.json"
RUN_ROOT="$DATA_BENCH/_protocol/ppr_final_audit_v2"
LOG_PATH="$RUN_ROOT/job-${SLURM_JOB_ID:-manual}.log"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

for path in "$PYTHON_BIN" "$CAMPAIGN_MANIFEST" "$AUDIT_MANIFEST"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done

export LKG_REPO LKG_DATA_ROOT
"$PYTHON_BIN" "$ROOT/scripts/71_audit_ppr_final_recovery.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" \
  --recovery-manifest "$AUDIT_MANIFEST" \
  --data-root "$LKG_DATA_ROOT" \
  --output "$RUN_ROOT/audit.json"

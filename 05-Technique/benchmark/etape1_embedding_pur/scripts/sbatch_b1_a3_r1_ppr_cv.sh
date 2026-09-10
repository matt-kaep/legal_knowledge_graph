#!/usr/bin/env bash
#SBATCH --job-name=lkg-b1r1-ppr-cv
#SBATCH --partition=CPU
#SBATCH --array=0-10%11
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail

: "${LKG_REPO:?Set LKG_REPO to the remote code checkout}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the remote data root}"
: "${LKG_PYTHON:?Set LKG_PYTHON to the verified remote Python runtime}"
graphs=(
  G1
  G6-citation-AA-knn5
  G6-citation-JJ-knn5
  G7-citation-AA-cit1-sem025-knn5
  G7-citation-AA-cit1-sem050-knn5
  G7-citation-AA-cit1-sem100-knn5
  G7-citation-AA-cit025-sem1-knn5
  G7-citation-JJ-cit1-sem025-knn5
  G7-citation-JJ-cit1-sem050-knn5
  G7-citation-JJ-cit1-sem100-knn5
  G7-citation-JJ-cit025-sem1-knn5
)
graph_id="${graphs[$SLURM_ARRAY_TASK_ID]}"
exec "$LKG_PYTHON" "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/94_run_b1_a3_campaign.py" \
  --manifest "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r1.json" \
  --stage ppr-cv --graph-id "$graph_id"

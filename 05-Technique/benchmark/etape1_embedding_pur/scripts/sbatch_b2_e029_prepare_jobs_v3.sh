#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e029-prepare-jobs-v3
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Successor to v2: Slurm spools this file away from the repository, so locate
# the hash-checked core through the required LKG_REPO rather than dirname($0).
set -eEuo pipefail

: "${LKG_REPO:?Set LKG_REPO to the reproducibility checkout.}"
export E029_JOBS_PREPARATION_MANIFEST_FILENAME="b2_reranking_comparable_a3_jobs_preparation_v3.json"
exec bash "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_b2_e029_prepare_jobs_v1.sh"

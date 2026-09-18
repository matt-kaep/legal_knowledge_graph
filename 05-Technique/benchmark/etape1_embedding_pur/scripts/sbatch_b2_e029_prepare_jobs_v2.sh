#!/usr/bin/env bash
#SBATCH --job-name=lkg-b2-e029-prepare-jobs-v2
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Successor to v1: retain all v1 artifacts immutable and use the v2 manifest,
# whose audited cosine/PPR paths may legitimately span preflight and depth roots.
set -eEuo pipefail

export E029_JOBS_PREPARATION_MANIFEST_FILENAME="b2_reranking_comparable_a3_jobs_preparation_v2.json"
exec bash "$(dirname "$0")/sbatch_b2_e029_prepare_jobs_v1.sh"

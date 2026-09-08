#!/usr/bin/env bash
#SBATCH --job-name=lkg-b1-ppr-scoped-replay
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

: "${LKG_REPO:?Set LKG_REPO to the remote code checkout}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the remote data root}"
: "${LKG_PYTHON:?Set LKG_PYTHON to the verified remote Python runtime}"

exec "$LKG_PYTHON" "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/102_replay_b1_a3_scoped_ppr.py" \
  --manifest "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_ppr_scoped_missing_replays_v1.json"

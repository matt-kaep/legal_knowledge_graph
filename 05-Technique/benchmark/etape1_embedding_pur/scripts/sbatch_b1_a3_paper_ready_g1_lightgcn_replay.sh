#!/usr/bin/env bash
#SBATCH --job-name=lkg-b1-g1-lightgcn-replay
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

: "${LKG_REPO:?Set LKG_REPO to the remote code checkout}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the remote data root}"
: "${LKG_PYTHON:?Set LKG_PYTHON to the verified remote Python runtime}"

manifest="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_paper_ready_g1.json"
"$LKG_PYTHON" "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/101_aggregate_b1_a3_scoped_lightgcn.py" --manifest "$manifest"
"$LKG_PYTHON" "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/95_freeze_b1_a3_champions.py" --manifest "$manifest" --family lightgcn
exec "$LKG_PYTHON" "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/96_replay_b1_a3_champions.py" --manifest "$manifest" --family lightgcn

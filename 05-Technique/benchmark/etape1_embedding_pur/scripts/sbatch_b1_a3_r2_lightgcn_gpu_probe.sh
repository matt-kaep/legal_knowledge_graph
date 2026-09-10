#!/usr/bin/env bash
#SBATCH --job-name=lkg-b1r2-lightgcn-gpu-probe
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

b1_root="$LKG_DATA_ROOT/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_campaign_b1_a3_effective_retrieval_r2_cuda_atomic_20260903"
task_plan="$b1_root/lightgcn_cv/task_plan.json"
preflight="$b1_root/lightgcn_cv/preflight.json"
task_root="$b1_root/lightgcn_cv/tasks"

exec "$LKG_PYTHON" "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/97_run_b1_a3_r2_lightgcn_tasks.py" \
  --manifest "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r2.json" \
  --task-plan "$task_plan" \
  --preflight "$preflight" \
  --task-root "$task_root" \
  --task-index 0

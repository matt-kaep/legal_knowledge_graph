#!/bin/bash
#SBATCH --job-name=e017-lgcn-cv
#SBATCH --output=e017_lightgcn_cv_%A_%a.out
#SBATCH --error=e017_lightgcn_cv_%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=CPU
#SBATCH --mem=48G
#SBATCH --array=0-10%10

set -euo pipefail

export LKG_REPO="${LKG_REPO:-$HOME/legal_knowledge_graph}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
E017_PYTHON="${E017_PYTHON:-/usr/bin/python3}"
cd "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
for seed_offset in 0 1 2; do
  task_index=$((SLURM_ARRAY_TASK_ID * 3 + seed_offset))
  "$E017_PYTHON" scripts/82_run_e017_lightgcn_task.py \
    --stage cv \
    --task-index "$task_index"
done

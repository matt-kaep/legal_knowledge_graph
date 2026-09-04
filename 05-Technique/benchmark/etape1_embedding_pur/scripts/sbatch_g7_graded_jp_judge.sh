#!/bin/bash
#SBATCH --job-name=g7_graded_jp
#SBATCH --output=g7_graded_jp_%j.out
#SBATCH --error=g7_graded_jp_%j.err
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mm
#SBATCH --qos=qos-mm
#SBATCH --mem=80G

set -euo pipefail

export LKG_REPO="${LKG_REPO:-$HOME/legal_knowledge_graph}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
cd "$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
bash scripts/run_g7_graded_jp_judge_on_cluster.sh "${1:-full}"

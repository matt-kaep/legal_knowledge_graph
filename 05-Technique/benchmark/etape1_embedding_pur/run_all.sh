#!/usr/bin/env bash
# Orchestration end-to-end de l'Étape 1.
# Pré-requis : ./scripts/_setup_legi.sh exécuté au moins une fois.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

if [ ! -f data/legi/legi.sqlite ]; then
  echo "→ Build SQLite LEGI (one-shot)"
  ./scripts/_setup_legi.sh
fi

python scripts/02_fetch_articles.py
python scripts/01_token_stats.py
python scripts/03_embed.py --device mps --batch 32
python scripts/04_eval_recall.py

echo
echo "✓ Étape 1 complète. Artefacts dans data/ :"
ls -lh data/*.npy data/*.parquet data/*.csv data/*.json 2>/dev/null

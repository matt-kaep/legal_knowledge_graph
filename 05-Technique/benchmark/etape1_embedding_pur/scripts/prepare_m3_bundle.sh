#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# prepare_m3_bundle.sh — assemble (en LOCAL, sur le Mac) le tarball à expédier
# au cluster pour le run M3. Reconstruit l'arborescence relative attendue sous
# $LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/ :
#   etape1/                       (package config — depuis main checkout)
#   scripts/23_eval_m3_llm_judge.py, run_m3_judge_on_cluster.sh (worktree)
#   prompts/m3_judge_{art,jp}_v1.txt, schemas/m3_judge_format.json (worktree)
#   data/global_bench/{rankings.parquet,bench_global.json}        (main)
#   data/{articles_all.parquet,jp_summaries_penal.parquet}        (main)
#
# Le script 23 NE charge PAS les embeddings .npy → bundle léger (parquets seuls).
# Sortie : m3_bundle.tar.gz (à scp/rsync vers le cluster).
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

MAIN="/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph"
WT="$MAIN/.claude/worktrees/etat-lieux-johnny-2026-05-28"
WE="05-Technique/benchmark/etape1_embedding_pur"   # chemin relatif commun

TMPROOT="$(mktemp -d)"
STAGE="$TMPROOT/$WE"
mkdir -p "$STAGE"/{scripts,prompts,schemas,data/global_bench}

echo "→ staging dans $STAGE"

# Code (worktree)
cp "$WT/$WE/scripts/23_eval_m3_llm_judge.py"      "$STAGE/scripts/"
cp "$WT/$WE/scripts/run_m3_judge_on_cluster.sh"   "$STAGE/scripts/"
cp "$WT/$WE/prompts/m3_judge_art_v1.txt"          "$STAGE/prompts/"
cp "$WT/$WE/prompts/m3_judge_jp_v1.txt"           "$STAGE/prompts/"
cp "$WT/$WE/schemas/m3_judge_format.json"         "$STAGE/schemas/"

# Package config (main checkout)
cp -R "$MAIN/$WE/etape1"                          "$STAGE/"
# nettoie les caches/pyc
find "$STAGE/etape1" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Données (main checkout)
cp "$MAIN/$WE/data/global_bench/rankings.parquet"   "$STAGE/data/global_bench/"
cp "$MAIN/$WE/data/global_bench/bench_global.json"   "$STAGE/data/global_bench/"
cp "$MAIN/$WE/data/articles_all.parquet"             "$STAGE/data/"
cp "$MAIN/$WE/data/jp_summaries_penal.parquet"       "$STAGE/data/"

# Tarball (racine = 05-Technique/... pour untar direct sous $LKG_REPO)
OUT="$MAIN/m3_bundle.tar.gz"
ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$STAGE")")")")"   # remonte à la racine du mktemp
tar -czf "$OUT" -C "$ROOT" "$WE"

echo "✓ bundle : $OUT"
echo "  taille : $(du -h "$OUT" | cut -f1)"
echo "  contenu :"
tar -tzf "$OUT" | sed 's/^/    /'
rm -rf "$ROOT"

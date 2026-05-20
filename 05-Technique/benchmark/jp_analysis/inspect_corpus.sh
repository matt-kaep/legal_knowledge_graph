#!/usr/bin/env bash
# inspect_corpus.sh — détecte le format réel des fichiers du corpus
# Usage :  bash inspect_corpus.sh [/chemin/du/dossier]
set -u
DIR="${1:-/home/ids/kaeppelin-22/work/database-judilibre}"

echo "=== Contenu de $DIR ==="
ls -la "$DIR" 2>/dev/null || { echo "ERREUR : dossier introuvable"; exit 1; }

echo
echo "=== Type réel (magic bytes) de chaque entrée ==="
for f in "$DIR"/*; do
  [ -f "$f" ] || continue
  printf '  %-60s  ' "$(basename "$f")"
  file -b "$f" | head -c 100
  echo
done

echo
echo "=== Premières lignes / schéma de chaque fichier ==="
for f in "$DIR"/*; do
  [ -f "$f" ] || continue
  echo "---- $(basename "$f") ----"
  # Test parquet via magic bytes (4 derniers octets = PAR1)
  if [ "$(tail -c 4 "$f" 2>/dev/null)" = "PAR1" ]; then
    echo "[parquet détecté — schéma :]"
    python3 -c "
import pyarrow.parquet as pq
pf = pq.ParquetFile('$f')
print(pf.schema)
print('num_rows =', pf.metadata.num_rows)
" 2>&1 | head -25
  else
    echo "[non-parquet — 1ère ligne (200 chars max) :]"
    head -n 1 "$f" | head -c 200
    echo
    echo "[nombre de lignes :] $(wc -l < "$f")"
  fi
  echo
done

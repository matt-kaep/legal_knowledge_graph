#!/usr/bin/env bash
# Télécharge et construit la SQLite LEGI à partir du dump DILA.
# Idempotent : skip si la SQLite existe déjà avec >100k articles.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEGI_DIR="$ROOT/data/legi"
mkdir -p "$LEGI_DIR/fonds"
cd "$LEGI_DIR"

if [ -f legi.sqlite ]; then
  N=$(sqlite3 legi.sqlite 'SELECT COUNT(*) FROM articles;' 2>/dev/null || echo 0)
  if [ "${N:-0}" -gt 100000 ]; then
    echo "✓ legi.sqlite déjà construite ($N articles) — skip"
    exit 0
  fi
fi

echo "→ Téléchargement archives LEGI (échanges DILA, ftp://echanges.dila.gouv.fr/LEGI/)"
echo "   ~3 GB, peut prendre 30-60 min selon réseau"
# Récupérer dernier freemium tar.gz + delta tar.gz
# Si `python -m legi.download` n'existe pas, utiliser le fallback FTP manuel ci-dessous.
if python -c "import legi.download" 2>/dev/null; then
    python -m legi.download fonds/
else
    echo "   (legi.download absent — fallback FTP manuel)"
    # Lister la dernière archive Freemium globale et la télécharger
    LATEST=$(curl -s "ftp://echanges.dila.gouv.fr/LEGI/" | grep "Freemium_legi_global_" | tail -1 | awk '{print $NF}')
    if [ -z "${LATEST:-}" ]; then
        echo "✗ Impossible de lister l'archive Freemium. Télécharger manuellement dans fonds/"
        exit 1
    fi
    echo "   Latest: $LATEST"
    curl -o "fonds/$LATEST" "ftp://echanges.dila.gouv.fr/LEGI/$LATEST"
fi

echo "→ Construction SQLite via legi.tar2sqlite"
python -m legi.tar2sqlite legi.sqlite fonds/

N=$(sqlite3 legi.sqlite 'SELECT COUNT(*) FROM articles;')
echo "✓ legi.sqlite OK ($N articles)"

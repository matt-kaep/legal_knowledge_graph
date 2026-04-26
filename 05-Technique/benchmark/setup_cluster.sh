#!/usr/bin/env bash
# =============================================================================
# setup_cluster.sh — Bootstrap minimal : venv + Jupyter via uv
#
# Ce script ne fait QUE le strict minimum pour ouvrir le notebook :
#   1. Installe uv localement si absent (sans sudo, dans ~/.local/bin)
#   2. Crée un virtualenv dédié via uv (~/.venv-benchmark)
#   3. Installe Jupyter dedans
#   4. Lance jupyter notebook sur le port 8888
#
# Tout le reste (install vllm, download modèle, démarrage serveur) est fait
# depuis le notebook lui-même (section « 0. Setup cluster »).
#
# Pourquoi uv plutôt que python3 -m venv ?
#   Sur Debian/Ubuntu sans python3-venv installé (cas fréquent sur cluster HPC
#   sans sudo), `python3 -m venv` échoue car `ensurepip` est absent. uv embarque
#   sa propre logique de bootstrap pip et marche sans dépendance système.
#
# Variables d'environnement :
#   VENV_DIR       : chemin du venv        (défaut: $HOME/.venv-benchmark)
#   JUPYTER_PORT   : port d'écoute         (défaut: 8888)
#   JUPYTER_TOKEN  : token d'accès         (défaut: benchmark-legal-kg)
#   PYTHON_VERSION : version Python cible  (défaut: 3.12)
#
# Depuis votre machine locale, créer le tunnel SSH :
#   ssh -L 8888:localhost:8888 -L 8000:localhost:8000 user@cluster
# =============================================================================

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.venv-benchmark}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-benchmark-legal-kg}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# ── Installation de uv (sans sudo) si absent ─────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "→ Installation de uv dans ~/.local/bin (sans sudo)…"
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "ERREUR : ni curl ni wget disponible pour télécharger uv."
        exit 1
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "→ $(uv --version) détecté."

# ── Création du venv via uv ──────────────────────────────────────────────────
# On teste bin/activate plutôt que le dossier, car un échec antérieur de
# `python3 -m venv` peut laisser un dossier vide/incomplet.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "→ Nettoyage d'un venv incomplet : $VENV_DIR"
        rm -rf "$VENV_DIR"
    fi
    echo "→ Création du virtualenv (uv) : $VENV_DIR  (Python ${PYTHON_VERSION})"
    # --seed : pré-installe pip/setuptools/wheel dans le venv, pour que
    # `%pip install` dans le notebook fonctionne sans bootstrap manuel.
    uv venv --seed --python "${PYTHON_VERSION}" "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Jupyter dans le venv ─────────────────────────────────────────────────────
echo "→ Installation / mise à jour de Jupyter dans le venv…"
uv pip install -q --upgrade pip
uv pip install -q jupyter ipywidgets

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<EOF

══════════════════════════════════════════════════════════
  Jupyter démarre sur le port ${JUPYTER_PORT}
══════════════════════════════════════════════════════════

  Depuis votre machine locale, ouvrir un tunnel SSH :
    ssh -L ${JUPYTER_PORT}:localhost:${JUPYTER_PORT} \\
        -L 8000:localhost:8000 \\
        user@cluster

  URL Jupyter :
    http://localhost:${JUPYTER_PORT}/?token=${JUPYTER_TOKEN}

  Ouvrir : benchmark_m1_m6_sample.ipynb
  → exécuter la section « 0. Setup cluster »
    (install vllm + download modèle + démarrage serveur)

══════════════════════════════════════════════════════════

EOF

exec jupyter notebook \
    --no-browser \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT}" \
    --NotebookApp.token="${JUPYTER_TOKEN}" \
    --notebook-dir="${NOTEBOOK_DIR}"

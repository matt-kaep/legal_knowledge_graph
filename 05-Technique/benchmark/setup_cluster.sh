#!/usr/bin/env bash
# =============================================================================
# setup_cluster.sh — Mise en place de l'environnement benchmark sur cluster GPU
#
# Usage :
#   bash setup_cluster.sh [OPTIONS]
#
# Options (variables d'environnement) :
#   MODEL_ID      : ID HuggingFace du modèle (défaut: google/gemma-4-31B-it)
#   NUM_GPUS      : Nombre de GPUs pour le tensor parallelism (défaut: auto-detect)
#   MAX_LEN       : Longueur de contexte max en tokens (défaut: 8192)
#   VLLM_PORT     : Port du serveur vLLM (défaut: 8000)
#   HF_TOKEN      : Token HuggingFace (ou exporter dans l'env)
#   SKIP_INSTALL  : "1" pour sauter l'installation pip (si déjà fait)
#   DOWNLOAD_ONLY : "1" pour ne télécharger le modèle que sans démarrer le serveur
#
# Exemples :
#   # Run complet (install + download + serveur)
#   bash setup_cluster.sh
#
#   # Modèle MoE (moins de VRAM, plus rapide)
#   MODEL_ID=google/gemma-4-pt-27b-it bash setup_cluster.sh
#
#   # Serveur uniquement (modèle déjà en cache)
#   SKIP_INSTALL=1 bash setup_cluster.sh
#
#   # Sur SLURM (dans un script de job)
#   #SBATCH --gres=gpu:4
#   #SBATCH --mem=200G
#   bash setup_cluster.sh
# =============================================================================

set -euo pipefail

# ── Configuration par défaut ─────────────────────────────────────────────────

MODEL_ID="${MODEL_ID:-google/gemma-4-31B-it}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
DOWNLOAD_ONLY="${DOWNLOAD_ONLY:-0}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR"

# ── Couleurs ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Détection GPU ─────────────────────────────────────────────────────────────

detect_gpus() {
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        ok "Détecté : ${NUM_GPUS} GPU(s), ${GPU_MEM} MiB VRAM chacun"
    else
        warn "nvidia-smi non disponible — NUM_GPUS=1 par défaut"
        NUM_GPUS=1
        GPU_MEM=0
    fi
    export NUM_GPUS
}

# ── Recommandation de config selon VRAM ──────────────────────────────────────

recommend_config() {
    local total_vram=$(( GPU_MEM * NUM_GPUS ))
    info "VRAM totale : ~${total_vram} MiB sur ${NUM_GPUS} GPU(s)"

    if [ "$total_vram" -ge 160000 ]; then
        info "Config recommandée : google/gemma-4-31B-it en bf16 (TP=${NUM_GPUS})"
    elif [ "$total_vram" -ge 80000 ]; then
        info "Config recommandée : google/gemma-4-31B-it en bf16 (1x H100)"
    elif [ "$total_vram" -ge 40000 ]; then
        warn "VRAM limite pour 31B — utilisez le modèle MoE :"
        warn "  MODEL_ID=google/gemma-4-pt-27b-it bash setup_cluster.sh"
    elif [ "$total_vram" -ge 20000 ]; then
        warn "VRAM insuffisante pour 31B bf16 — MoE 4-bit requis :"
        warn "  MODEL_ID=bartowski/google_gemma-4-31B-it-GGUF bash setup_cluster.sh"
    fi
}

# ── Installation des dépendances ─────────────────────────────────────────────

install_deps() {
    if [ "$SKIP_INSTALL" = "1" ]; then
        warn "SKIP_INSTALL=1 — installation pip ignorée"
        return
    fi

    info "Installation des dépendances Python…"
    pip install -q --upgrade pip

    # vLLM dernière version stable
    pip install -q "vllm>=0.8.5" 2>&1 | tail -3

    # Dépendances notebook + data
    pip install -q \
        "openai>=1.50.0" \
        "pydantic>=2.0.0" \
        "pandas>=2.0.0" \
        "pyarrow>=14.0.0" \
        "huggingface_hub>=0.25.0" \
        "datasets>=3.0.0" \
        "rich>=13.0.0" \
        "matplotlib>=3.8.0" \
        "seaborn>=0.13.0" \
        "jupyter>=1.0.0" \
        "ipywidgets>=8.0.0" \
        "tqdm>=4.65.0" 2>&1 | tail -3

    ok "Dépendances installées"
}

# ── Authentification HuggingFace ─────────────────────────────────────────────

setup_hf_auth() {
    if [ -n "${HF_TOKEN:-}" ]; then
        info "HF_TOKEN fourni — configuration…"
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
        ok "Authentification HuggingFace configurée"
    elif huggingface-cli whoami &>/dev/null 2>&1; then
        ok "Déjà authentifié sur HuggingFace : $(huggingface-cli whoami)"
    else
        warn "HuggingFace non authentifié. Gemma 4 est un modèle GATED."
        warn "Options :"
        warn "  1. Exporter HF_TOKEN=hf_xxxx avant de relancer ce script"
        warn "  2. Accepter les conditions sur : https://huggingface.co/google/gemma-4-31B-it"
        warn "Continuer malgré tout (peut échouer au téléchargement)…"
    fi
}

# ── Pré-téléchargement du modèle ─────────────────────────────────────────────

download_model() {
    info "Pré-téléchargement du modèle ${MODEL_ID}…"
    info "(Gemma 4 31B = ~62 GB en bf16 — peut prendre 30-60 min selon la bande passante)"

    python - <<EOF
from huggingface_hub import snapshot_download
import sys

model_id = "${MODEL_ID}"
print(f"Téléchargement de {model_id}…")
try:
    path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.md", "*.txt", "original/*"],
    )
    print(f"Modèle en cache : {path}")
except Exception as e:
    print(f"ERREUR : {e}", file=sys.stderr)
    sys.exit(1)
EOF
    ok "Modèle ${MODEL_ID} disponible en cache"
}

# ── Démarrage du serveur vLLM ────────────────────────────────────────────────

start_vllm_server() {
    info "Démarrage du serveur vLLM…"
    info "  Modèle    : ${MODEL_ID}"
    info "  GPU(s)    : ${NUM_GPUS}"
    info "  Contexte  : ${MAX_LEN} tokens"
    info "  Port      : ${VLLM_PORT}"

    local LOG_FILE="${LOG_DIR}/vllm_server.log"

    # Tuer un éventuel serveur précédent sur ce port
    if lsof -ti:"$VLLM_PORT" &>/dev/null; then
        warn "Port ${VLLM_PORT} déjà utilisé — arrêt du processus existant…"
        lsof -ti:"$VLLM_PORT" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    # Lancement en arrière-plan
    nohup vllm serve "${MODEL_ID}" \
        --tensor-parallel-size "${NUM_GPUS}" \
        --max-model-len "${MAX_LEN}" \
        --gpu-memory-utilization 0.90 \
        --port "${VLLM_PORT}" \
        --enable-auto-tool-choice \
        --trust-remote-code \
        > "${LOG_FILE}" 2>&1 &

    VLLM_PID=$!
    echo "$VLLM_PID" > "${LOG_DIR}/vllm.pid"
    info "Serveur vLLM démarré (PID=${VLLM_PID}), logs : ${LOG_FILE}"

    # ── Attente que le serveur soit prêt (max 10 min) ─────────────────────
    info "Attente du démarrage (chargement du modèle en VRAM)…"
    local RETRIES=120
    local WAIT=5
    for i in $(seq 1 $RETRIES); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" &>/dev/null; then
            ok "Serveur vLLM prêt sur http://localhost:${VLLM_PORT}"
            break
        fi
        if [ "$i" -eq "$RETRIES" ]; then
            error "Serveur vLLM non disponible après $((RETRIES * WAIT))s — voir ${LOG_FILE}"
        fi
        printf "  Tentative %d/%d…\r" "$i" "$RETRIES"
        sleep $WAIT
    done

    # Afficher les modèles disponibles
    info "Modèles disponibles sur le serveur :"
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" | python -c "
import json,sys
data = json.load(sys.stdin)
for m in data.get('data', []):
    print(f'  - {m[\"id\"]}')
" 2>/dev/null || true
}

# ── Lancement du notebook Jupyter ────────────────────────────────────────────

launch_jupyter() {
    local NOTEBOOK_DIR
    NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    info "Lancement du serveur Jupyter…"
    info "Notebook : ${NOTEBOOK_DIR}/benchmark_m1_m6_sample.ipynb"

    # Générer un token fixe pour simplifier la connexion
    local JUPYTER_TOKEN="benchmark-legal-kg"

    nohup jupyter notebook \
        --no-browser \
        --ip=0.0.0.0 \
        --port=8888 \
        --NotebookApp.token="$JUPYTER_TOKEN" \
        --notebook-dir="$NOTEBOOK_DIR" \
        > "${LOG_DIR}/jupyter.log" 2>&1 &

    local JUPYTER_PID=$!
    echo "$JUPYTER_PID" > "${LOG_DIR}/jupyter.pid"

    sleep 3
    ok "Jupyter démarré (PID=${JUPYTER_PID})"
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Connexion Jupyter (depuis votre machine locale)${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
    echo ""
    echo "  SSH tunnel (depuis votre machine) :"
    echo "    ssh -L 8888:localhost:8888 -L ${VLLM_PORT}:localhost:${VLLM_PORT} user@cluster"
    echo ""
    echo "  URL Jupyter : http://localhost:8888/?token=${JUPYTER_TOKEN}"
    echo ""
    echo "  Ouvrir : benchmark_m1_m6_sample.ipynb"
    echo ""
    echo "  Variables à configurer dans la cellule [Configuration] :"
    echo "    VLLM_BASE_URL = \"http://localhost:${VLLM_PORT}/v1\""
    echo "    MODEL_ID      = \"${MODEL_ID}\""
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
# Point d'entrée principal
# ══════════════════════════════════════════════════════════════════════

main() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Benchmark M1-M6 Legal KG — Setup Cluster${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo ""

    detect_gpus
    recommend_config
    echo ""

    install_deps
    setup_hf_auth
    download_model

    if [ "$DOWNLOAD_ONLY" = "1" ]; then
        ok "DOWNLOAD_ONLY=1 — modèle téléchargé, serveur non démarré"
        exit 0
    fi

    start_vllm_server
    launch_jupyter
}

main "$@"

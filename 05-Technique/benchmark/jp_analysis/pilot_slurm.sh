#!/usr/bin/env bash
# =============================================================================
# pilot_slurm.sh — Job SLURM autonome : pilote JP-analysis Step 1 (Plan Task 13)
#
#   Tu soumets, tu te déconnectes. Le job :
#     1. crée un venv ISOLÉ dédié via uv (ne touche PAS l'env --user partagé,
#        fragile — cf. mémoire cluster-user-env-fragile / prérequis R1)
#     2. installe vLLM + les deps du pipeline (pinné, dans le venv isolé)
#     3. lance le serveur vLLM gemma4-31B-AWQ en arrière-plan
#     4. attend qu'il réponde (/v1/models), sinon échoue proprement (R1)
#     5. exécute le pilote 30 JP
#     6. exécute le benchmark de concurrence {1,8,16,32}
#     7. arrête le serveur et termine (NE LANCE PAS le run 1,12 M — gate humain)
#
#   Tout est écrit dans le fichier de log SLURM (-o ci-dessous). Le run complet
#   reste une décision séparée APRÈS revue de PILOT.md.
#
#   SOUMISSION (depuis le dossier 05-Technique/benchmark/jp_analysis du repo
#   sur le cluster) :
#       sbatch pilot_slurm.sh
#   Puis tu peux fermer ta session. Suivi : `tail -f logs/pilot_<jobid>.out`
#   ou `squeue -u $USER`.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  ### À ADAPTER À TON CLUSTER (les 6 lignes #SBATCH ci-dessous) ###
#  Mets le nom réel de ta partition GPU / ton compte. Le reste est raisonnable.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=jp-pilot
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.out
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1                 # 1 GPU (L40S 48GB). Variante possible: --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# #SBATCH --partition=gpu            # <-- DÉCOMMENTE + mets ta partition GPU
# #SBATCH --account=ton_compte       # <-- DÉCOMMENTE si ton cluster exige -A
# #SBATCH --constraint=l40s          # <-- DÉCOMMENTE si tu dois cibler la L40S

set -euo pipefail

# ── Config (surchargeables via env au sbatch : `sbatch --export=...`) ────────
JP_DIR="${JP_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
VENV_DIR="${VENV_DIR:-$HOME/.venv-jp-analysis}"     # venv ISOLÉ dédié (pas le partagé)
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MODEL_ID="${MODEL_ID:-QuantTrio/gemma-4-31B-it-AWQ}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_VERSION="${VLLM_VERSION:-}"                    # vide = dernière ; sinon ex. "0.6.3"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"            # s d'attente du chargement modèle (31B AWQ = lent)
PILOT_N="${PILOT_N:-30}"
BENCH_LEVELS="${BENCH_LEVELS:-1 8 16 32}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# export HF_TOKEN=...   # <-- décommente/exporte si le repo HF du modèle est gated

cd "$JP_DIR"
mkdir -p logs outputs
if [ ! -f "run_step1.py" ]; then
  echo "ERREUR : run_step1.py introuvable dans $JP_DIR." >&2
  echo "Soumets depuis 05-Technique/benchmark/jp_analysis, ou exporte JP_DIR." >&2
  exit 2
fi

echo "=============================================================="
echo " JP-analysis pilote — job ${SLURM_JOB_ID:-local} — $(date -Is)"
echo " hôte=$(hostname)  JP_DIR=$JP_DIR  venv=$VENV_DIR"
echo " modèle=$MODEL_ID  max_model_len=$MAX_MODEL_LEN  port=$VLLM_PORT"
echo "=============================================================="
nvidia-smi || { echo "ERREUR : pas de GPU visible (vérifie --gres/--partition)."; exit 2; }

# ── uv (sans sudo) ───────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
  echo "→ Installation de uv (~/.local/bin, sans sudo)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "→ $(uv --version)"

# ── venv ISOLÉ + deps pinnées (ne touche pas l'env --user partagé : R1) ──────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  rm -rf "$VENV_DIR"
  uv venv --seed --python "$PYTHON_VERSION" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
uv pip install -q --upgrade pip
echo "→ Installation des deps du pipeline (requirements.txt)…"
uv pip install -q -r requirements.txt
if [ -n "$VLLM_VERSION" ]; then
  echo "→ Installation vllm==$VLLM_VERSION"
  uv pip install -q "vllm==$VLLM_VERSION"
else
  echo "→ Installation vllm (dernière compatible)"
  uv pip install -q vllm
fi
python -c "import vllm, transformers, openai, pyarrow, pydantic, rapidfuzz, json_repair; \
print('deps OK — vllm', vllm.__version__, '| transformers', transformers.__version__)"

# ── Lancement serveur vLLM en arrière-plan ───────────────────────────────────
SERVER_LOG="logs/vllm_${SLURM_JOB_ID:-local}.log"
echo "→ Démarrage serveur vLLM (log : $SERVER_LOG)…"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$VLLM_PORT" \
  --guided-decoding-backend xgrammar \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!
# Arrêt propre du serveur quoi qu'il arrive (fin, erreur, timeout SLURM)
trap 'echo "→ Arrêt serveur vLLM (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# ── Attente santé : /v1/models répond, sinon échec explicite (R1) ────────────
echo "→ Attente du serveur (timeout ${HEALTH_TIMEOUT}s — chargement 31B AWQ lent)…"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERREUR : le serveur vLLM s'est arrêté pendant le chargement." >&2
    echo "  → Très probablement le prérequis R1 (env/gemma4 arch, cf. mémoire" >&2
    echo "    cluster-user-env-fragile). Dernières lignes de $SERVER_LOG :" >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 3
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERREUR : timeout — serveur non prêt après ${HEALTH_TIMEOUT}s." >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 3
  fi
  sleep 15
done
echo "→ Serveur prêt ✓ ($(date -Is))"

BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
COMMON_ARGS=( --max-model-len "$MAX_MODEL_LEN" --model "$MODEL_ID"
              --tokenizer-id "$MODEL_ID" --base-url "$BASE_URL" )

# ── 1) Pilote 30 JP stratifiées ──────────────────────────────────────────────
echo "=============================================================="
echo " PILOTE — ${PILOT_N} JP stratifiées (CC/CA/TJ)"
echo "=============================================================="
python run_step1.py --pilot "$PILOT_N" "${COMMON_ARGS[@]}" \
  --concurrency 16 --out outputs/step1_pilot

# ── 2) Benchmark de concurrence ──────────────────────────────────────────────
echo "=============================================================="
echo " BENCHMARK CONCURRENCE — niveaux : $BENCH_LEVELS"
echo "=============================================================="
printf '%-12s %-12s %-14s\n' "concurrency" "wall_s" "records/min"
for C in $BENCH_LEVELS; do
  rm -rf "outputs/step1_bench_${C}"
  start=$(date +%s)
  python run_step1.py --pilot "$PILOT_N" "${COMMON_ARGS[@]}" \
    --concurrency "$C" --out "outputs/step1_bench_${C}" \
    > "logs/bench_c${C}_${SLURM_JOB_ID:-local}.log" 2>&1
  end=$(date +%s); wall=$(( end - start ))
  n=$(cat "outputs/step1_bench_${C}"/*/part-*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  rpm=$(awk -v n="${n:-0}" -v w="$wall" 'BEGIN{ if (w>0) printf "%.1f", n*60.0/w; else print "n/a" }')
  printf '%-12s %-12s %-14s\n' "$C" "$wall" "$rpm"
done

echo "=============================================================="
echo " TERMINÉ ✓  $(date -Is)"
echo " Sorties : outputs/step1_pilot/ , outputs/step1_bench_*/"
echo " Métriques latence : outputs/step1_pilot/_metrics.jsonl (duration_ms)"
echo " Anomalies thèmes  : outputs/step1_pilot/_themes_anomalies.jsonl"
echo ""
echo " PROCHAINE ÉTAPE (décision humaine, hors de ce job) :"
echo "  - rédiger PILOT.md (qualité FR, % themes_valid, taux Autre:, verbatim"
echo "    attendu_cle, débit vs concurrence) ;"
echo "  - SI validé : lancer le run complet 1,12 M (job séparé, sans --pilot,"
echo "    avec la concurrence retenue ici)."
echo "=============================================================="

#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# run_m3_judge_on_cluster.sh — M3 (LLM-as-judge) sur le nœud GPU du cluster.
#
# Démarre vLLM (Gemma 4 26B, AWQ), attend /health, lance 23_eval_m3_llm_judge.py,
# puis tue vLLM proprement. PAS de pip install (env --user fragile, cf. mémoire
# cluster-user-env-fragile) : l'env vLLM doit déjà être présent.
#
# Usage :
#   ./run_m3_judge_on_cluster.sh pilot    # 20 questions (défaut) — À INSPECTER
#   ./run_m3_judge_on_cluster.sh full     # run complet (~10-12h background)
#
# Pré-requis à rsync sur le cluster (sous $LKG_REPO, mêmes chemins relatifs) :
#   05-Technique/benchmark/etape1_embedding_pur/etape1/            (package config)
#   05-Technique/benchmark/etape1_embedding_pur/prompts/m3_judge_*.txt
#   05-Technique/benchmark/etape1_embedding_pur/schemas/m3_judge_format.json
#   05-Technique/benchmark/etape1_embedding_pur/scripts/23_eval_m3_llm_judge.py
#   data/global_bench/{rankings.parquet,bench_global.json}
#   data/{articles_all.parquet,jp_summaries_penal.parquet}
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-pilot}"

# ── Repo : surcharge le chemin Mac hardcodé. Adapter au checkout cluster.
export LKG_REPO="${LKG_REPO:-$HOME/legal_knowledge_graph}"
SCRIPT_DIR="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts"

# ── Modèle vLLM (cf. doctrine_qgen MODEL_REGISTRY / MODEL_REVISION)
MODEL_ID="${MODEL_ID:-cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}"
REVISION="${REVISION:-519bdca117c8}"   # commit pré-régression (cf. doctrine_qgen)
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"
WORKERS="${WORKERS:-32}"

LOG_DIR="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur/data/global_bench/m3_logs"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm_m3.log"

NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || echo 1)"
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

echo "══ M3 judge — mode=$MODE  modèle=$MODEL_ID  GPUs=$NUM_GPUS ══"
[ -z "${HF_TOKEN:-}" ] && echo "[WARN] HF_TOKEN non défini — pull AWQ gated peut échouer"

# ── Démarrage vLLM
echo "→ start vLLM (log: $VLLM_LOG)"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --revision "$REVISION" \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "→ kill vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Attente health (max 15 min)
echo "→ attente /health…"
for i in $(seq 1 180); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[FAIL] vLLM mort au démarrage — voir $VLLM_LOG"; exit 2
    fi
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "→ vLLM prêt (${i}×5s)"; break
    fi
    sleep 5
done

# ── Lancement du juge
ARGS=(--model-id "$MODEL_ID" --port "$PORT" --workers "$WORKERS")
if [ "$MODE" = "pilot" ]; then
    ARGS+=(--pilot 20)
    echo "→ PILOTE 20 questions — inspecter la ligne SANITY (GT-singleton ⇒ majorité n2)"
elif [ "$MODE" != "full" ]; then
    echo "[FAIL] mode inconnu '$MODE' (attendu: pilot | full)"; exit 1
fi

echo "→ python 23_eval_m3_llm_judge.py ${ARGS[*]}"
python3 "$SCRIPT_DIR/23_eval_m3_llm_judge.py" "${ARGS[@]}"

echo "══ Terminé (mode=$MODE) ══"

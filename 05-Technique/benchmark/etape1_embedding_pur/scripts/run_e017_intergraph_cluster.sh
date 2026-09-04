#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-inventory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${LKG_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
ROOT="$REPO/05-Technique/benchmark/etape1_embedding_pur"
CONFIG="$ROOT/configs/e017_intergraph_graded_jp_cluster.json"
CAMPAIGN_ROOT="$ROOT/data/doctrine_v3plus_bench/_protocol/e017_intergraph_graded_jp"
TRANSFER_PATHS="$CAMPAIGN_ROOT/transfer_paths.txt"
REMOTE_HOST="${REMOTE_HOST:-kaeppelin-22@gpu-gw.enst.fr}"
REMOTE_REPO="${REMOTE_REPO:-legal_knowledge_graph}"
REMOTE_CAMPAIGN="$REMOTE_REPO/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_protocol/e017_intergraph_graded_jp"
RSYNC_SSH="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

materialize() {
  python3 "$ROOT/scripts/81_e017_intergraph_cluster_contract.py" \
    --config "$CONFIG" \
    --out-dir "$CAMPAIGN_ROOT"
}

inventory() {
  materialize
  rsync -ani \
    --checksum \
    -e "$RSYNC_SSH" \
    --relative \
    --files-from="$TRANSFER_PATHS" \
    --out-format='%i %n%L' \
    "$REPO/" \
    "$REMOTE_HOST:$REMOTE_REPO/"
}

sync_inputs() {
  materialize
  copied=0
  for attempt in 1 2 3; do
    if rsync -a \
      --checksum \
      --partial \
      -e "$RSYNC_SSH" \
      --relative \
      --files-from="$TRANSFER_PATHS" \
      "$REPO/" \
      "$REMOTE_HOST:$REMOTE_REPO/"; then
      copied=1
      break
    fi
    if [ "$attempt" -lt 3 ]; then
      sleep 5
    fi
  done
  if [ "$copied" -ne 1 ]; then
    echo "E017 input synchronization failed after 3 attempts" >&2
    exit 1
  fi
  rsync -a --checksum -e "$RSYNC_SSH" \
    "$CAMPAIGN_ROOT/campaign_manifest.json" \
    "$CAMPAIGN_ROOT/tasks.jsonl" \
    "$CAMPAIGN_ROOT/transfer_paths.txt" \
    "$REMOTE_HOST:$REMOTE_CAMPAIGN/"
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$REMOTE_HOST" \
    "LKG_REPO=\"\$HOME/$REMOTE_REPO\" \"\$HOME/.venv-benchmark/bin/python\" \"\$HOME/$REMOTE_REPO/05-Technique/benchmark/etape1_embedding_pur/scripts/81_e017_intergraph_cluster_contract.py\" --verify-campaign \"\$HOME/$REMOTE_CAMPAIGN/campaign_manifest.json\""
}

submit_cpu() {
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$REMOTE_HOST" 'bash -s' <<EOF
set -euo pipefail
export LKG_REPO="\$HOME/$REMOTE_REPO"
cd "\$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
cv_job=\$(sbatch --parsable scripts/sbatch_e017_lightgcn_cv.sh)
replay_job=\$(sbatch --parsable --dependency="aftercorr:\$cv_job" scripts/sbatch_e017_lightgcn_replay.sh)
python3 - "\$cv_job" "\$replay_job" <<'PY'
import json
import sys
print(json.dumps({"cv_job_id": sys.argv[1], "replay_job_id": sys.argv[2]}))
PY
EOF
}

case "$MODE" in
  inventory)
    inventory
    ;;
  sync)
    sync_inputs
    ;;
  submit-cpu)
    submit_cpu
    ;;
  *)
    echo "usage: $0 {inventory|sync|submit-cpu}" >&2
    exit 2
    ;;
esac

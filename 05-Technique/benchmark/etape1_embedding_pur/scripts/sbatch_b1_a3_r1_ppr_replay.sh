#!/usr/bin/env bash
#SBATCH --job-name=lkg-b1r1-ppr-replay
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

: "${LKG_REPO:?Set LKG_REPO to the remote code checkout}"
: "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT to the remote data root}"
: "${LKG_PYTHON:?Set LKG_PYTHON to the verified remote Python runtime}"

ROOT="$LKG_REPO/05-Technique/benchmark/etape1_embedding_pur"
PARENT_MANIFEST="$ROOT/configs/confirmatory_campaign_b1_a3_r1.json"
DISPATCH_MANIFEST="$ROOT/configs/b1_a3_r1_ppr_replay_dispatch.json"
FINAL_ROOT="$LKG_DATA_ROOT/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_campaign_b1_a3_effective_retrieval_r1_20260902/ppr_final"

if [[ -d "$FINAL_ROOT" ]] && find "$FINAL_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty B1-r1 PPR final output: $FINAL_ROOT" >&2
  exit 1
fi

export LKG_REPO LKG_DATA_ROOT
"$LKG_PYTHON" - "$DISPATCH_MANIFEST" "$PARENT_MANIFEST" "$LKG_DATA_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


dispatch_path, parent_path, data_root = map(Path, sys.argv[1:])
dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
if sha256(parent_path) != dispatch["parent_campaign"]["sha256"]:
    raise SystemExit("parent B1-r1 manifest hash mismatch")
frozen_path = data_root / dispatch["frozen_champions"]["path"]
if sha256(frozen_path) != dispatch["frozen_champions"]["sha256"]:
    raise SystemExit("frozen PPR champions hash mismatch")
frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
if frozen.get("selection_data") != dispatch["frozen_champions"]["selection_data"]:
    raise SystemExit("frozen PPR champions are not train/CV-only")
code_root = parent_path.parents[4]
for name, item in dispatch["code_bundle"].items():
    if sha256(code_root / item["path"]) != item["sha256"]:
        raise SystemExit(f"code hash mismatch: {name}")
print(json.dumps({"dispatch_id": dispatch["dispatch_id"], "parent_manifest_sha256": sha256(parent_path), "frozen_champions_sha256": sha256(frozen_path)}, sort_keys=True))
PY

exec "$LKG_PYTHON" "$ROOT/scripts/96_replay_b1_a3_champions.py" \
  --manifest "$PARENT_MANIFEST" \
  --family ppr

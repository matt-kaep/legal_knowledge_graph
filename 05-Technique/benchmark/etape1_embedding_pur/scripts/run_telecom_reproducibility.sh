#!/usr/bin/env bash
# Submit and inspect the controlled Télécom recovery jobs from a local checkout.
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${LKG_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST, for example user@gpu-gw.enst.fr.}"
REMOTE_REPO="${REMOTE_REPO:?Set REMOTE_REPO to the remote code checkout name.}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-}"
REMOTE_PYTHON="${REMOTE_PYTHON:-}"
REMOTE_BRANCH="${REMOTE_BRANCH:-paper/ecir-2027-reproducibility}"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10)

sync_code() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_BRANCH" <<'REMOTE'
set -euo pipefail
repo_name="$1"
branch="$2"
repo="$HOME/$repo_name"
cd "$repo"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "remote tracked changes present; refusing to update the checkout" >&2
  git status --short >&2
  exit 2
fi
git fetch origin "$branch"
git switch "$branch"
git pull --ff-only origin "$branch"
git log -1 --oneline
git status --short
REMOTE
}

submit() {
  local batch_script="$1"
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_DATA_ROOT" "$REMOTE_PYTHON" "$batch_script" <<'REMOTE'
set -euo pipefail
repo_name="$1"
data_root="${2:-}"
python_bin="${3:-}"
batch_script="$4"
repo="$HOME/$repo_name"
data_root="${data_root:-$HOME/${repo_name}_data}"
python_bin="${python_bin:-$HOME/work/.venv-benchmark/bin/python}"
test -d "$repo"
test -d "$data_root"
test -x "$python_bin"
cd "$repo"
sbatch --parsable --export="ALL,LKG_REPO=$repo,LKG_DATA_ROOT=$data_root,LKG_PYTHON=$python_bin" "$repo/$batch_script"
REMOTE
}

queue() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" 'squeue -u "$USER" -o "%.18i %.12T %.12P %.28j %.10M %.30R"'
}

inventory() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_DATA_ROOT" <<'REMOTE'
set -euo pipefail
repo_name="$1"
data_root="${2:-}"
repo="$HOME/$repo_name"
data_root="${data_root:-$HOME/${repo_name}_data}"
cd "$repo"
git branch --show-current
git log -1 --oneline
git status --short
printf 'PPR final files: '
find "$data_root/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_final_grouped_v2" -type f \( -name selected_champions.json -o -name final_champions_summary.csv -o -name rankings.parquet \) | wc -l
printf 'E021 jobs/responses: '
wc -l "$data_root/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_e021_jobs/E021-cluster-gpu-runtime-v5/jobs.jsonl" "$data_root/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_e021_jobs/E021-cluster-gpu-runtime-v5/responses.jsonl"
REMOTE
}

case "$MODE" in
  sync-code) sync_code ;;
  inventory) inventory ;;
  queue) queue ;;
  submit-ppr-audit)
    submit "05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_ppr_final_audit.sh"
    ;;
  submit-e021-resume)
    submit "05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_e021_reranking_resume.sh"
    ;;
  *)
    echo "usage: $0 {sync-code|inventory|queue|submit-ppr-audit|submit-e021-resume}" >&2
    exit 2
    ;;
esac

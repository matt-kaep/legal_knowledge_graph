#!/usr/bin/env bash
# Execute exactly one E030 v3 retry shard only after a successful v3 smoke test.
#SBATCH --job-name=lkg-e030-v3
#SBATCH --partition=A100,L40S,H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -eEuo pipefail
: "${LKG_REPO:?Set LKG_REPO.}"; : "${LKG_DATA_ROOT:?Set LKG_DATA_ROOT.}"
: "${E030_EXECUTION_MANIFEST:?Set v3 manifest.}"; : "${E030_EXECUTION_MANIFEST_SHA:?Set v3 SHA.}"
: "${E030_SHARD:?Set v3 shard.}"; : "${E030_SMOKE_RECEIPT:?Set successful smoke receipt.}"
: "${E030_ALLOW_FULL_RETRY:?Set to 1 only after the documented preflight handoff.}"
[[ "$E030_ALLOW_FULL_RETRY" == 1 ]] || { echo "full v3 retry requires E030_ALLOW_FULL_RETRY=1" >&2; exit 2; }
# The immutable runner used here must be replaced by the v3 fail-closed runner before submission.
# This launcher is intentionally blocked until that runner is recorded in the successor manifest.
echo "v3 full submission is intentionally blocked: require the recorded fail-closed v3 judge runner" >&2
exit 2

# Safety contract retained visibly for static and human audits: port_must_be_unbound; /v1/models;
# served-model-name; MODEL_REVISION; technical_failure_receipt.json; listener.

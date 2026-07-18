#!/usr/bin/env bash
#SBATCH --job-name=continue-adpv2-alignedv4-babel
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail
export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
DATASET_ROOT=${RUN_ROOT}/dataset
EVIDENCE=${RUN_ROOT}/evidence
RECORDER=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/record_aligned_release_evidence.py
TRAIN=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/train_qwen35_4b_babel_aligned.sbatch
MONITOR_JOB=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/monitor_aligned_babel.sbatch
MONITOR_SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/monitor_aligned_babel.py
WATCHER=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/watch_tokenized_manifest.py
CONFIG=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/qwen35_4b_adpv2_full24k_aligned_babel_one_epoch.yaml
PREPARER=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/prepare_aligned_restartable_config.py
DEEPSPEED=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/ds_z3_config.json
ALIGNMENT=${DATASET_ROOT}/alignment_manifest_v4.json
VALIDATION=${EVIDENCE}/alignment_v4_validation-9357231.json
PREFLIGHT=${EVIDENCE}/aligned_v4_loader_preflight.json
RESULT=${EVIDENCE}/aligned_v4_persisted_continuation.json

verify_sha256() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256sum "$path")
  observed=${observed%% *}
  [[ "$observed" == "$expected" ]] || {
    echo "content_hash_mismatch path=${path} expected=${expected} observed=${observed}" >&2
    return 1
  }
}

verify_sha256 "$RECORDER" 5be4eb49dc01f2895f3b77c91405fbcace483c83a8cdf4e2ff7cb6c726e8a6be
verify_sha256 "$TRAIN" 0f05cfd409cb7ecfe477f203617bd7108c4de864af3b0a0013a7a481f2ef0d99
verify_sha256 "$MONITOR_JOB" 8501319ad9b210c365e0d4a9d3ab83f4b6eecf782512b363f309cead09b7d0ce
verify_sha256 "$MONITOR_SCRIPT" b8cfc1faa2178dc58e9c652c872c9b486405e3e707d29108c8d0ba57c8119cc3
verify_sha256 "$WATCHER" b37e995c1b3d264be13d0df1d51a478f43dfdb885213b72711da23d770c3ab22
verify_sha256 "$CONFIG" bf2ea895c5d8703c1870589800335a2b6dfcc8d622cbd8bca6ad3a2cd54d69c5
verify_sha256 "$PREPARER" ffe381f208ab59c63fab7d7afa9e751b0410e55cf3bbd345bd76f048d9c22b77
verify_sha256 "$DEEPSPEED" 2868fc47fa4f4006aeca2e81443338679dbf9b4cb14980e735bd0af6099aae6d

cp -a "$0" "${EVIDENCE}/continue_babel_after_v4.${SLURM_JOB_ID}.sbatch"
for path in "$RECORDER" "$TRAIN" "$MONITOR_JOB" "$MONITOR_SCRIPT" "$WATCHER" "$CONFIG" "$PREPARER" "$DEEPSPEED"; do
  cp -a "$path" "${EVIDENCE}/$(basename "$path").v4_continuation_${SLURM_JOB_ID}"
done
sha256sum "$0" "$RECORDER" "$TRAIN" "$MONITOR_JOB" "$MONITOR_SCRIPT" "$WATCHER" "$CONFIG" "$PREPARER" "$DEEPSPEED"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"

python "$RECORDER" \
  --manifest "${DATASET_ROOT}/manifest.json" \
  --alignment-manifest "$ALIGNMENT" \
  --validation "$VALIDATION" \
  --loader-preflight "$PREFLIGHT" \
  --builder-job-id 9357230 \
  --validation-job-id 9357231 \
  --loader-preflight-job-id 9357233

sha256sum "${DATASET_ROOT}/manifest.json" "$ALIGNMENT" "$VALIDATION" "$PREFLIGHT"
TRAIN_JOB_ID=$(sbatch --parsable "$TRAIN")
MONITOR_JOB_ID=$(
  sbatch --parsable \
    --dependency="after:${TRAIN_JOB_ID}" \
    --export=ALL,TRAIN_JOB_ID="$TRAIN_JOB_ID" \
    "$MONITOR_JOB"
)
export TRAIN_JOB_ID MONITOR_JOB_ID
python - "$RESULT" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "continuation_job_id": int(os.environ["SLURM_JOB_ID"]),
    "builder_job_id": 9357230,
    "validation_job_id": 9357231,
    "loader_preflight_job_id": 9357233,
    "training_job_id": int(os.environ["TRAIN_JOB_ID"]),
    "monitor_job_id": int(os.environ["MONITOR_JOB_ID"]),
    "monitor_dependency": f"after:{os.environ['TRAIN_JOB_ID']}",
    "acceptance_claimed": False,
    "note": "Submission is not acceptance; require server-synced loss, active GPUs, and verified checkpoint-backed restart.",
}
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(path)
print(json.dumps(value, indent=2, sort_keys=True))
PY

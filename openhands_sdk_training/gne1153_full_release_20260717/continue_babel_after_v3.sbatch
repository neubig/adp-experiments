#!/usr/bin/env bash
#SBATCH --job-name=continue-adpv2-alignedv3-babel
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
DATASET_ROOT=${RUN_ROOT}/dataset
EVIDENCE=${RUN_ROOT}/evidence
RECORDER=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/record_aligned_release_evidence.py
TRAIN=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/train_qwen35_4b_babel_aligned.sbatch
MONITOR=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/monitor_aligned_babel.sbatch
ALIGNMENT=${DATASET_ROOT}/alignment_manifest_v3.json
VALIDATION=${EVIDENCE}/alignment_v3_validation-9357001.json
PREFLIGHT=${EVIDENCE}/aligned_v3_loader_preflight.json
RESULT=${EVIDENCE}/aligned_v3_persisted_continuation.json

cp -a "$0" "${EVIDENCE}/continue_babel_after_v3.${SLURM_JOB_ID}.sbatch"
cp -a "$RECORDER" "${EVIDENCE}/record_aligned_release_evidence.${SLURM_JOB_ID}.py"
cp -a "$TRAIN" "${EVIDENCE}/train_qwen35_4b_babel_aligned.source.${SLURM_JOB_ID}.sbatch"
cp -a "$MONITOR" "${EVIDENCE}/monitor_aligned_babel.source.${SLURM_JOB_ID}.sbatch"
sha256sum "$0" "$RECORDER" "$TRAIN" "$MONITOR"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"

python "$RECORDER" \
  --manifest "${DATASET_ROOT}/manifest.json" \
  --alignment-manifest "$ALIGNMENT" \
  --validation "$VALIDATION" \
  --loader-preflight "$PREFLIGHT" \
  --builder-job-id 9357000 \
  --validation-job-id 9357001 \
  --loader-preflight-job-id 9357006

sha256sum "${DATASET_ROOT}/manifest.json" "$ALIGNMENT" "$VALIDATION" "$PREFLIGHT"
TRAIN_JOB_ID=$(sbatch --parsable "$TRAIN")
MONITOR_JOB_ID=$(sbatch --parsable --export=ALL,TRAIN_JOB_ID="$TRAIN_JOB_ID" "$MONITOR")
export TRAIN_JOB_ID MONITOR_JOB_ID
python - "$RESULT" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "continuation_job_id": int(os.environ["SLURM_JOB_ID"]),
    "builder_job_id": 9357000,
    "validation_job_id": 9357001,
    "loader_preflight_job_id": 9357006,
    "training_job_id": int(os.environ["TRAIN_JOB_ID"]),
    "monitor_job_id": int(os.environ["MONITOR_JOB_ID"]),
    "acceptance_claimed": False,
    "note": "Submission is not acceptance; monitor requires loss, active GPUs, W&B, complete checkpoint, and verified restart.",
}
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(path)
print(json.dumps(value, indent=2, sort_keys=True))
PY

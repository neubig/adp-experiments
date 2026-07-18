#!/usr/bin/env bash
#SBATCH --job-name=adpv2-v3-marker-audit
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/scan_adpv2_function_content.py
DATASET_INFO=${RUN_ROOT}/dataset_aligned_v3/dataset_info.json
RESULT=${RUN_ROOT}/evidence/alignment_v3_function_marker_audit-${SLURM_JOB_ID}.json

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
sha256sum "$0" "$SCRIPT" "$DATASET_INFO"
cp -a "$0" "${RUN_ROOT}/evidence/scan_v3_function_markers.${SLURM_JOB_ID}.sbatch"
cp -a "$SCRIPT" "${RUN_ROOT}/evidence/scan_adpv2_function_content.${SLURM_JOB_ID}.py"

python "$SCRIPT" \
  --dataset-info "$DATASET_INFO" \
  --workers 8 \
  --omit-samples >"$RESULT"

echo "finished_at=$(date -Is)"
sha256sum "$RESULT"
jq '{entries:length, rows:map(.rows)|add, function_messages:map(.function_messages)|add, issue_messages:map(.issue_messages)|add, affected:[.[]|select(.issue_messages>0)|{dataset_name,path,rows,function_messages,issue_messages,issue_counts}]}' "$RESULT"

#!/usr/bin/env bash
#SBATCH --job-name=adpv2-align-validate-v4
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/alignment_v4_validation-%j.json
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/alignment_v4_validation-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/validate_aligned_release.py

cp -a "$0" "${RUN_ROOT}/evidence/validate_alignment_v4.${SLURM_JOB_ID}.sbatch"
cp -a "$SCRIPT" "${RUN_ROOT}/evidence/validate_aligned_release.v4.${SLURM_JOB_ID}.py"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)" >&2
sha256sum "$0" "$SCRIPT" \
  "${RUN_ROOT}/dataset/alignment_manifest_v4.json" \
  "${RUN_ROOT}/dataset_aligned_v4/dataset_info.json" >&2

python "$SCRIPT" \
  --dataset-root "${RUN_ROOT}/dataset" \
  --alignment-manifest-name alignment_manifest_v4.json \
  --workers 8

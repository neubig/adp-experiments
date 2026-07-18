#!/usr/bin/env bash
#SBATCH --job-name=adpv2-loader-preflight-v4
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/aligned-v4-loader-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/aligned-v4-loader-%j.err

set -euo pipefail
export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/preflight_aligned_loader.py
OUTPUT=${RUN_ROOT}/evidence/aligned_v4_loader_preflight.json

cp -a "$0" "${RUN_ROOT}/evidence/preflight_alignment_v4.${SLURM_JOB_ID}.sbatch"
cp -a "$SCRIPT" "${RUN_ROOT}/evidence/preflight_aligned_loader.v4.${SLURM_JOB_ID}.py"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
sha256sum "$0" "$SCRIPT" \
  "${RUN_ROOT}/dataset/alignment_manifest_v4.json" \
  "${RUN_ROOT}/dataset_aligned_v4/dataset_info.json"

python "$SCRIPT" \
  --dataset-root "${RUN_ROOT}/dataset" \
  --alignment-manifest-name alignment_manifest_v4.json \
  --output "$OUTPUT"
sha256sum "$OUTPUT"

#!/usr/bin/env bash
#SBATCH --job-name=adpv2full24k-project-schema
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/project_training_schema.py

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
sha256sum "$SCRIPT"
cp -a "$SCRIPT" "${RUN_ROOT}/evidence/project_training_schema.py.${SLURM_JOB_ID}"
cp -a "$0" "${RUN_ROOT}/evidence/project_training_schema.sbatch.${SLURM_JOB_ID}"
python "$SCRIPT" --dataset-root "${RUN_ROOT}/dataset"
echo "finished_at=$(date -Is)"
sha256sum "${RUN_ROOT}/dataset/manifest.json" \
  "${RUN_ROOT}/dataset/training_projection_manifest.json" \
  "${RUN_ROOT}/dataset/dataset_info.json"

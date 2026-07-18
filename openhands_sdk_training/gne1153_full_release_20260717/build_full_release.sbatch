#!/usr/bin/env bash
#SBATCH --job-name=gne1153-adpv2-full24k-build
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/build-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/build-%j.err

set -euo pipefail

SOURCE_REPO=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
SOURCE_ROOT=/home/gneubig/exp/adp/datasets/v1
RUN_PARENT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
OUTPUT_ROOT=${RUN_PARENT}/dataset
ADAPTER=/home/gneubig/work/adp/agent-data-protocol-pr244/agents/openhands_sdk/sft_to_llamafactory.py

mkdir -p "${RUN_PARENT}/evidence"
echo "job_id=${SLURM_JOB_ID}"
echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "source_repo_commit=$(git -C "$SOURCE_REPO" rev-parse HEAD)"
echo "adapter_repo_commit=$(git -C /home/gneubig/work/adp/agent-data-protocol-pr244 rev-parse HEAD)"
df -h "$SOURCE_ROOT" "$RUN_PARENT"
python "$SOURCE_REPO/openhands_sdk_training/gne1153_full_release_20260717/build_full_release.py" \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --adapter "$ADAPTER"
echo "finished_at=$(date -Is)"

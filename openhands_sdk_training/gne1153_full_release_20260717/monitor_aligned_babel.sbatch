#!/usr/bin/env bash
#SBATCH --job-name=monitor-adpv2-aligned-babel
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=11:55:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail
: "${TRAIN_JOB_ID:?submit with --export=ALL,TRAIN_JOB_ID=<aligned training job>}"

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/monitor_aligned_babel.py
PREPARE=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/prepare_aligned_restartable_config.py
SNAPSHOT=${RUN_ROOT}/evidence/monitor_aligned_${TRAIN_JOB_ID}.snapshot.json

cp -a "$0" "${RUN_ROOT}/evidence/monitor_aligned_babel.${SLURM_JOB_ID}.sbatch"
cp -a "$SCRIPT" "${RUN_ROOT}/evidence/monitor_aligned_babel.${SLURM_JOB_ID}.py"
cp -a "$PREPARE" "${RUN_ROOT}/evidence/prepare_aligned_restartable_config.monitor.${SLURM_JOB_ID}.py"
sha256sum "$0" "$SCRIPT" "$PREPARE"
python "$SCRIPT" \
  --job-id "$TRAIN_JOB_ID" \
  --run-root "$RUN_ROOT" \
  --output "$SNAPSHOT"

#!/usr/bin/env bash
#SBATCH --job-name=monitor-adpv2-orchard-35b
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=11:55:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail
: "${TRAIN_JOB_ID:?submit with --export=ALL,TRAIN_JOB_ID=<Orchard training job>}"

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RECIPE_ROOT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717
DATA_RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
EVIDENCE_DIR=${DATA_RUN_ROOT}/evidence
TOKENIZED_MANIFEST=${DATA_RUN_ROOT}/dataset/tokenized_aligned_v4_qwen35_35b_a3b_manifest.json
TOKENIZED_PATH=/project/flame/gneubig/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/tokenized_aligned_v4_qwen35_35b_a3b_qwen3_5_nothink_seq32768
SCRIPT=${RECIPE_ROOT}/monitor_orchard_35b.py
VERIFIER=${RECIPE_ROOT}/verify_wandb_metrics.py
SNAPSHOT=${EVIDENCE_DIR}/monitor_orchard_${TRAIN_JOB_ID}.snapshot.json
WANDB_EVIDENCE=${EVIDENCE_DIR}/wandb_orchard_${TRAIN_JOB_ID}.json

cp -a "$0" "${EVIDENCE_DIR}/monitor_orchard_35b.${SLURM_JOB_ID}.sbatch"
cp -a "$SCRIPT" "${EVIDENCE_DIR}/monitor_orchard_35b.${SLURM_JOB_ID}.py"
sha256sum "$0" "$SCRIPT" "$VERIFIER"

exec /home/gneubig/exp/adp-smoke/.venv/bin/python "$SCRIPT" \
  --job-id "$TRAIN_JOB_ID" \
  --slurm-log-dir "$RECIPE_ROOT" \
  --evidence-dir "$EVIDENCE_DIR" \
  --tokenized-manifest "$TOKENIZED_MANIFEST" \
  --tokenized-path "$TOKENIZED_PATH" \
  --wandb-verifier "$VERIFIER" \
  --wandb-evidence "$WANDB_EVIDENCE" \
  --output "$SNAPSHOT"

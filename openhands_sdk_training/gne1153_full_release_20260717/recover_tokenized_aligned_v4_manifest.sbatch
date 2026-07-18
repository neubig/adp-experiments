#!/usr/bin/env bash
#SBATCH --job-name=recover-adpv2-v4-token-manifest
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=11:55:00
#SBATCH --open-mode=append
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/watch_tokenized_manifest.py
MANIFEST=${RUN_ROOT}/dataset/manifest.json
TOKENIZED=${RUN_ROOT}/tokenized_aligned_v4_qwen35_4b_base_qwen3_5_nothink_seq32768
OUTPUT=${RUN_ROOT}/dataset/tokenized_aligned_v4_manifest.json
TRAIN_LOG=${RUN_ROOT}/evidence/adpv2full24k-alignedv4-q35-4b-babel-1ep-9357329.err
EVIDENCE=${RUN_ROOT}/evidence
EXECUTED=${EVIDENCE}/recover_tokenized_aligned_v4_manifest.${SLURM_JOB_ID}.sbatch
SCRIPT_SNAPSHOT=${EVIDENCE}/watch_tokenized_manifest.recovery.${SLURM_JOB_ID}.py

export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH

cp -a "$0" "$EXECUTED"
cp -a "$SCRIPT" "$SCRIPT_SNAPSHOT"
sha256sum "$EXECUTED" "$SCRIPT_SNAPSHOT"
echo "job_id=${SLURM_JOB_ID}"
echo "started_at=$(date -Is)"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
echo "training_job_id=9357329"
echo "manifest=${MANIFEST}"
echo "tokenized_path=${TOKENIZED}"
echo "output=${OUTPUT}"

python "$SCRIPT_SNAPSHOT" \
  --manifest "$MANIFEST" \
  --tokenized-path "$TOKENIZED" \
  --output "$OUTPUT" \
  --training-log "$TRAIN_LOG" \
  --require-zero-filtering \
  --wait-seconds 39600

echo "finished_at=$(date -Is)"
sha256sum "$OUTPUT" "$MANIFEST"

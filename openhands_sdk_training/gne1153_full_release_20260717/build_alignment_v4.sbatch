#!/usr/bin/env bash
#SBATCH --job-name=adpv2-align-extend-v4
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/alignment_v4_builder-%j.json
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/alignment_v4_builder-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
SCRIPT=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/extend_aligned_release.py
NORMALIZER=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/normalize_adpv2_projection.py
BASE=${RUN_ROOT}/dataset/alignment_manifest_v3.json
AUDIT=${RUN_ROOT}/evidence/alignment_v3_function_marker_audit-9357175.json

[[ $(sha256sum "$BASE" | cut -d' ' -f1) == fe2d3b4a4e2dbe84e545afde17375f767bc2efb97f025a0af0c26db1b06b77aa ]]
[[ $(sha256sum "$AUDIT" | cut -d' ' -f1) == adcd61539ff0dac666bf0e86899e974c47376cd472393395e734f24cc8afec79 ]]
cp -a "$0" "${RUN_ROOT}/evidence/build_alignment_v4.${SLURM_JOB_ID}.sbatch"
cp -a "$SCRIPT" "${RUN_ROOT}/evidence/extend_aligned_release.${SLURM_JOB_ID}.py"
cp -a "$NORMALIZER" "${RUN_ROOT}/evidence/normalize_adpv2_projection.v4.${SLURM_JOB_ID}.py"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)" >&2
sha256sum "$0" "$SCRIPT" "$NORMALIZER" "$BASE" "$AUDIT" >&2

python "$SCRIPT" \
  --dataset-root "${RUN_ROOT}/dataset" \
  --base-alignment-manifest "$BASE" \
  --targets \
    14_coderforge_preview.llamafactory.jsonl \
    20_gair_davinci_dev.llamafactory.jsonl \
    32_nebius_SWE-rebench-openhands-trajectories.llamafactory.jsonl \
    36_nvidia_SWE-Zero-openhands-trajectories.llamafactory.jsonl \
    49_toolmind.llamafactory.jsonl \
  --workers 5

sha256sum \
  "${RUN_ROOT}/dataset/alignment_manifest_v4.json" \
  "${RUN_ROOT}/dataset_aligned_v4/dataset_info.json" >&2

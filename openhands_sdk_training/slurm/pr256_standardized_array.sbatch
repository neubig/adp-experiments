#!/usr/bin/env bash
#SBATCH --job-name=adp-pr256-std
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/software_agent_pr256/logs/%x-%A_%a-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/software_agent_pr256/logs/%x-%A_%a-%j.err

set -euo pipefail

DATASETS=(
  "SALT-NLP_SWE-chat"
  "allenai_Sera-4.6-Lite-T2"
  "coderforge_preview"
  "codescout"
  "gair_davinci_dev"
  "hybrid-gym"
  "jupyter-agent-dataset"
  "kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k"
  "logicstar_swe-star"
  "mini-coder"
  "nebius_SWE-agent-trajectories"
  "nebius_SWE-rebench-openhands-trajectories"
  "nvidia_SWE-Zero-openhands-trajectories"
  "openhands"
  "scale_swe_distilled"
  "swe-gym_openhands_sampled_trajectories"
  "swe-play-trajectories"
  "swe-smith"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID for ${#DATASETS[@]} datasets" >&2
  exit 2
fi
DATASET=${DATASETS[$TASK_ID]}

export ADP_REPO=${ADP_REPO:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/agent-data-protocol}
export ADP_EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
export ADP_PYTHON=${ADP_PYTHON:-$ADP_REPO/.venv/bin/python}
export ADP_EXPECTED_BRANCH=${ADP_EXPECTED_BRANCH:-main}
export ADP_EXPECTED_COMMIT=${ADP_EXPECTED_COMMIT:-3824726}
export ADP_STD_OUT_ROOT=${ADP_STD_OUT_ROOT:-/home/gneubig/exp/adp/datasets/software_agent_standardized_pr256}

RUNNER=${ADP_STD_RUNNER:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/adp-experiments/openhands_sdk_training/scripts/generate_pr256_standardized.sh}

echo "dataset=$DATASET"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-}"
echo "job_id=${SLURM_JOB_ID:-}"
echo "started_at=$(date -Is)"
bash "$RUNNER" "$DATASET"

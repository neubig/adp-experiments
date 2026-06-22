#!/usr/bin/env bash
#SBATCH --job-name=adp-pr256-cond
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --requeue
#SBATCH --output=/home/gneubig/exp/adp/runs/software_agent_pr256/logs/%x-%A_%a-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/software_agent_pr256/logs/%x-%A_%a-%j.err

set -u

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
export ADP_MAX_TOKENS=${ADP_MAX_TOKENS:-24000}
export ADP_TOKEN_LABEL=${ADP_TOKEN_LABEL:-24k}
export ADP_COND_OUT_ROOT=${ADP_COND_OUT_ROOT:-/home/gneubig/exp/adp/datasets/software_agent_condenser_${ADP_TOKEN_LABEL}_pr256}
export ADP_CONDENSER_CONCURRENCY=${ADP_CONDENSER_CONCURRENCY:-24}
export ADP_CONDENSER_CHUNK_SIZE=${ADP_CONDENSER_CHUNK_SIZE:-96}
export ADP_CONDENSER_ROW_TIMEOUT=${ADP_CONDENSER_ROW_TIMEOUT:-1800}

RUNNER=${ADP_CONDENSER_RUNNER:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/adp-experiments/openhands_sdk_training/scripts/generate_pr256_openhands_sdk_condenser.sh}
RUN_TIMEOUT=${ADP_CONDENSER_RUN_TIMEOUT:-42300s}
FINAL_JSONL=$ADP_COND_OUT_ROOT/$DATASET/full_sft/full_sft_openhands_sdk_condensed_${ADP_TOKEN_LABEL}.jsonl

is_finished() {
  [ -s "$FINAL_JSONL" ]
}

requeue_if_unfinished() {
  reason=$1
  if is_finished; then
    echo "dataset=$DATASET token_label=$ADP_TOKEN_LABEL finished after $reason; not requeueing"
    return 0
  fi
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    echo "dataset=$DATASET token_label=$ADP_TOKEN_LABEL unfinished after $reason; requeueing job_id=$SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    return 0
  fi
  echo "dataset=$DATASET token_label=$ADP_TOKEN_LABEL unfinished after $reason, but no Slurm job id is available" >&2
  return 1
}

on_term() {
  echo "received SIGTERM at $(date -Is)"
  if [ -n "${runner_pid:-}" ]; then
    kill "$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
  fi
  requeue_if_unfinished "slurm-signal"
  exit $?
}
trap on_term TERM

echo "dataset=$DATASET"
echo "token_label=$ADP_TOKEN_LABEL"
echo "max_tokens=$ADP_MAX_TOKENS"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-}"
echo "job_id=${SLURM_JOB_ID:-}"
echo "final_jsonl=$FINAL_JSONL"
echo "run_timeout=$RUN_TIMEOUT"
echo "started_at=$(date -Is)"

if is_finished; then
  echo "dataset=$DATASET token_label=$ADP_TOKEN_LABEL already finished; skipping"
  exit 0
fi

timeout "$RUN_TIMEOUT" bash "$RUNNER" "$DATASET" &
runner_pid=$!
wait "$runner_pid"
status=$?
unset runner_pid

echo "runner_status=$status"
echo "finished_check_at=$(date -Is)"

if is_finished; then
  echo "dataset=$DATASET token_label=$ADP_TOKEN_LABEL finished"
  exit 0
fi

case "$status" in
  124|137|143)
    requeue_if_unfinished "timeout-status-$status"
    exit $?
    ;;
  *)
    echo "dataset=$DATASET token_label=$ADP_TOKEN_LABEL failed with status=$status" >&2
    exit "$status"
    ;;
esac

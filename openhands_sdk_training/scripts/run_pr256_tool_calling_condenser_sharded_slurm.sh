#!/usr/bin/env bash
#SBATCH --job-name=adp-tool-shard
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclusive

set -euo pipefail

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: sbatch $0 DATASET_NAME" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ADP_REPO=${ADP_REPO:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/agent-data-protocol}
export ADP_EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
export ADP_PYTHON=${ADP_PYTHON:-$ADP_REPO/.venv/bin/python}
export ADP_STD_OUT_ROOT=${ADP_STD_OUT_ROOT:-$ADP_EXP_ROOT/datasets/tool_calling_standardized_pr256}
export ADP_COND_OUT_ROOT=${ADP_COND_OUT_ROOT:-$ADP_EXP_ROOT/datasets/tool_calling_condenser_24k_pr256}
export ADP_MAX_TOKENS=${ADP_MAX_TOKENS:-24000}
export ADP_TOKEN_LABEL=${ADP_TOKEN_LABEL:-24k}
export ADP_CONDENSER_WORKERS=${ADP_CONDENSER_WORKERS:-12}
export ADP_CONDENSER_LLM_CONCURRENCY=${ADP_CONDENSER_LLM_CONCURRENCY:-20}
export ADP_CONDENSER_MAX_IN_FLIGHT_ROWS=${ADP_CONDENSER_MAX_IN_FLIGHT_ROWS:-50}
export ADP_CONDENSER_ROW_TIMEOUT=${ADP_CONDENSER_ROW_TIMEOUT:-1800}

echo "dataset=$DATASET"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname)"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "started_at=$(date -Is)"
echo "workers=$ADP_CONDENSER_WORKERS"
echo "llm_concurrency_per_worker=$ADP_CONDENSER_LLM_CONCURRENCY"
echo "total_llm_concurrency=$((ADP_CONDENSER_WORKERS * ADP_CONDENSER_LLM_CONCURRENCY))"

"$SCRIPT_DIR/generate_pr256_standardized.sh" "$DATASET"
"$SCRIPT_DIR/generate_pr256_openhands_sdk_condenser_sharded.sh" "$DATASET"

echo "finished_at=$(date -Is)"

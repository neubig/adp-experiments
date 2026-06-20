#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO=/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/agent-data-protocol
if [ ! -d "$DEFAULT_REPO" ]; then
  DEFAULT_REPO=/home/gneubig/work/adp/agent-data-protocol-pr244
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${ADP_REPO:-$DEFAULT_REPO}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
RUN_ROOT=${ADP_RUN_ROOT:-$EXP_ROOT/runs/all_datasets_pr256}
STD_OUT_ROOT=${ADP_STD_OUT_ROOT:-$EXP_ROOT/datasets/all_agent_standardized_pr256}
ARRAY_LIMIT=${ADP_ARRAY_LIMIT:-4}
PARTITION=${ADP_SLURM_PARTITION:-debug}
TIME=${ADP_SLURM_TIME:-24:00:00}
CPUS=${ADP_SLURM_CPUS:-8}
MEM=${ADP_SLURM_MEM:-64G}
DRY_RUN=0
SUBMIT_STD=1
SUBMIT_COND=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --standardize-only)
      SUBMIT_STD=1
      SUBMIT_COND=0
      ;;
    --condense-only)
      SUBMIT_STD=0
      SUBMIT_COND=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/logs"
DATASET_LIST=$RUN_ROOT/all_datasets.txt
"$SCRIPT_DIR/list_adp_datasets.py" --repo "$REPO" > "$DATASET_LIST"
DATASET_COUNT=$(wc -l < "$DATASET_LIST" | tr -d ' ')
if [ "$DATASET_COUNT" -eq 0 ]; then
  echo "No runnable datasets found under $REPO/datasets" >&2
  exit 1
fi
ARRAY_SPEC="0-$((DATASET_COUNT - 1))%$ARRAY_LIMIT"

STD_SBATCH=$RUN_ROOT/slurm/standardize_all.sbatch
COND_SBATCH=$RUN_ROOT/slurm/condense_all.sbatch

cat > "$STD_SBATCH" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=adp-all-std
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --array=$ARRAY_SPEC
#SBATCH --output=$RUN_ROOT/logs/adp-all-std-%A_%a.out
#SBATCH --error=$RUN_ROOT/logs/adp-all-std-%A_%a.err

set -euo pipefail
DATASET=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$DATASET_LIST")
echo "dataset=\$DATASET"
echo "array_job_id=\${SLURM_ARRAY_JOB_ID:-}"
echo "array_task_id=\${SLURM_ARRAY_TASK_ID:-}"
ADP_REPO="$REPO" \\
ADP_EXP_ROOT="$EXP_ROOT" \\
ADP_STD_OUT_ROOT="$STD_OUT_ROOT" \\
"$SCRIPT_DIR/run_standardize_dataset_slurm.sh" "\$DATASET"
SBATCH

cat > "$COND_SBATCH" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=adp-all-cond
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --array=$ARRAY_SPEC
#SBATCH --output=$RUN_ROOT/logs/adp-all-cond-%x-%A_%a.out
#SBATCH --error=$RUN_ROOT/logs/adp-all-cond-%x-%A_%a.err

set -euo pipefail
DATASET=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$DATASET_LIST")
echo "dataset=\$DATASET"
echo "array_job_id=\${SLURM_ARRAY_JOB_ID:-}"
echo "array_task_id=\${SLURM_ARRAY_TASK_ID:-}"
ADP_REPO="$REPO" \\
ADP_EXP_ROOT="$EXP_ROOT" \\
ADP_STD_OUT_ROOT="$STD_OUT_ROOT" \\
ADP_TOKEN_LABEL="\${ADP_TOKEN_LABEL:?}" \\
ADP_MAX_TOKENS="\${ADP_MAX_TOKENS:?}" \\
ADP_COND_OUT_ROOT="\${ADP_COND_OUT_ROOT:?}" \\
"$SCRIPT_DIR/run_condense_standardized_dataset_slurm.sh" "\$DATASET"
SBATCH

echo "dataset_count=$DATASET_COUNT"
echo "dataset_list=$DATASET_LIST"
echo "std_sbatch=$STD_SBATCH"
echo "cond_sbatch=$COND_SBATCH"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry_run=1"
  exit 0
fi

STD_JOB_ID=
if [ "$SUBMIT_STD" -eq 1 ]; then
  STD_JOB_ID=$(sbatch --parsable "$STD_SBATCH")
  echo "standardize_job_id=$STD_JOB_ID"
fi

if [ "$SUBMIT_COND" -eq 1 ]; then
  DEP_ARGS=()
  if [ -n "$STD_JOB_ID" ]; then
    DEP_ARGS=(--dependency=afterany:"$STD_JOB_ID")
  fi
  for spec in 24k:24000 110k:110000; do
    label=${spec%%:*}
    max_tokens=${spec##*:}
    out_root=$EXP_ROOT/datasets/all_agent_condenser_${label}_pr256
    job_id=$(ADP_TOKEN_LABEL="$label" ADP_MAX_TOKENS="$max_tokens" ADP_COND_OUT_ROOT="$out_root" \
      sbatch --parsable "${DEP_ARGS[@]}" "$COND_SBATCH")
    echo "condense_${label}_job_id=$job_id"
  done
fi

#!/usr/bin/env bash

set -euo pipefail

EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
RUN_ROOT=$EXP_ROOT/runs/software_agent_pr256
LOG_DIR=$RUN_ROOT/logs
SCRIPT_ROOT=${ADP_EXPERIMENTS_ROOT:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/adp-experiments}
SBATCH_FILE=$SCRIPT_ROOT/openhands_sdk_training/slurm/pr256_condenser_array.sbatch
STATE_FILE=$RUN_ROOT/pr256_condenser_monitor.tsv
MAX_PR256_JOBS=${ADP_MONITOR_MAX_PR256_JOBS:-12}
SLEEP_SECONDS=${ADP_MONITOR_SLEEP_SECONDS:-120}

mkdir -p "$RUN_ROOT" "$LOG_DIR"
touch "$STATE_FILE"

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

root_for_label() {
  local label=$1
  printf '%s/datasets/software_agent_condenser_%s_pr256' "$EXP_ROOT" "$label"
}

manifest_exists() {
  local label=$1
  local dataset=$2
  test -s "$(root_for_label "$label")/$dataset/manifest.json"
}

cancel_competing_debug_jobs() {
  squeue -u "$USER" -h -o '%i %j %P' \
    | awk '$2 ~ /^adp-swe-score$/ && $3 == "debug" {print $1}' \
    | xargs -r scancel
}

release_held_pr256_jobs() {
  squeue -u "$USER" -h -o '%i|%j|%T|%R' \
    | awk -F '|' '$2 == "adp-pr256-cond" && $3 == "PENDING" && tolower($4) ~ /held/ {print $1}' \
    | xargs -r scontrol release
}

active_pr256_count() {
  squeue -u "$USER" -h -o '%j' | awk '$1 == "adp-pr256-cond" {count++} END {print count + 0}'
}

active_pair_keys() {
  local job array_job task task_raw out label dataset state_label state_key

  squeue -u "$USER" -h -o '%i|%F|%K|%j' | awk -F '|' '$4 == "adp-pr256-cond" {print $1 "\t" $2 "\t" $3}' |
  while IFS=$'\t' read -r job array_job task_raw; do
    task=${task_raw%%%*}
    [[ "$task" =~ ^[0-9]+$ ]] || continue
    dataset=${DATASETS[$task]:-}
    [ -n "$dataset" ] || continue

    state_key="${array_job}_${task}"
    state_label=$(awk -v job="$job" -v state_key="$state_key" -F '\t' '$1 == job || $1 == state_key {label=$3} END {print label}' "$STATE_FILE")
    if [ -n "$state_label" ]; then
      printf '%s\t%s\n' "$state_label" "$dataset"
      continue
    fi

    out=$(scontrol show job "$job" 2>/dev/null | awk -F= '/StdOut=/{print $2; exit}')
    if [ -n "$out" ] && [ -s "$out" ]; then
      label=$(awk -F= '$1 == "token_label" {print $2; exit}' "$out")
      if [ -n "$label" ]; then
        printf '%s\t%s\n' "$label" "$dataset"
      fi
    fi
  done
}

submit_pair() {
  local label=$1
  local max_tokens=$2
  local idx=$3
  local dataset=${DATASETS[$idx]}
  local out_root
  local jobid

  out_root=$(root_for_label "$label")
  jobid=$(
    sbatch --parsable --array="${idx}%1" \
      --export=ALL,ADP_MAX_TOKENS="$max_tokens",ADP_TOKEN_LABEL="$label",ADP_COND_OUT_ROOT="$out_root" \
      "$SBATCH_FILE"
  )
  printf '%s_%s\t%s\t%s\t%s\t%s\n' "$jobid" "$idx" "$idx" "$label" "$dataset" "$(date -Is)" >> "$STATE_FILE"
  printf '%s submitted %s %s %s\n' "$(date -Is)" "$jobid" "$label" "$dataset"
}

missing_pairs() {
  local idx dataset label
  for idx in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$idx]}
    for label in 24k 110k; do
      if ! manifest_exists "$label" "$dataset"; then
        printf '%s\t%s\t%s\n' "$label" "$idx" "$dataset"
      fi
    done
  done
}

while true; do
  cancel_competing_debug_jobs
  release_held_pr256_jobs

  std_count=$(find "$EXP_ROOT/datasets/software_agent_standardized_pr256" -maxdepth 2 -name manifest.json 2>/dev/null | wc -l)
  count_24k=$(find "$(root_for_label 24k)" -path '*/manifest.json' 2>/dev/null | wc -l)
  count_110k=$(find "$(root_for_label 110k)" -path '*/manifest.json' 2>/dev/null | wc -l)
  current_jobs=$(active_pr256_count)

  printf '%s standardized=%s 24k=%s 110k=%s active_pr256_jobs=%s\n' \
    "$(date -Is)" "$std_count" "$count_24k" "$count_110k" "$current_jobs"

  if [ "$count_24k" -eq 18 ] && [ "$count_110k" -eq 18 ]; then
    echo "$(date -Is) all condenser manifests present"
    exit 0
  fi

  active_keys=$(mktemp)
  missing=$(mktemp)
  active_pair_keys | sort -u > "$active_keys"
  missing_pairs > "$missing"

  while [ "$current_jobs" -lt "$MAX_PR256_JOBS" ]; do
    next=$(
      while IFS=$'\t' read -r label idx dataset; do
        if ! grep -Fxq "$label"$'\t'"$dataset" "$active_keys"; then
          printf '%s\t%s\t%s\n' "$label" "$idx" "$dataset"
          break
        fi
      done < "$missing"
    )

    [ -n "$next" ] || break
    IFS=$'\t' read -r label idx dataset <<< "$next"
    if [ "$label" = 24k ]; then
      submit_pair "$label" 24000 "$idx"
    else
      submit_pair "$label" 110000 "$idx"
    fi
    printf '%s\t%s\n' "$label" "$dataset" >> "$active_keys"
    current_jobs=$((current_jobs + 1))
  done

  rm -f "$active_keys" "$missing"
  sleep "$SLEEP_SECONDS"
done

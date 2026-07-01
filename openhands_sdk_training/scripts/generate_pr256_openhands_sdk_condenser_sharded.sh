#!/usr/bin/env bash

set -euo pipefail

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: $0 DATASET_NAME" >&2
  exit 2
fi

REPO=${ADP_REPO:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/agent-data-protocol}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
PYTHON=${ADP_PYTHON:-$REPO/.venv/bin/python}
STD_ROOT=${ADP_STD_OUT_ROOT:-$EXP_ROOT/datasets/software_agent_standardized_pr256}
MAX_TOKENS=${ADP_MAX_TOKENS:-24000}
TOKEN_LABEL=${ADP_TOKEN_LABEL:-}
if [ -z "$TOKEN_LABEL" ]; then
  if [ "$MAX_TOKENS" -eq 24000 ]; then
    TOKEN_LABEL=24k
  elif [ "$MAX_TOKENS" -eq 110000 ]; then
    TOKEN_LABEL=110k
  else
    TOKEN_LABEL=${MAX_TOKENS}
  fi
fi
OUT_ROOT=${ADP_COND_OUT_ROOT:-$EXP_ROOT/datasets/software_agent_condenser_${TOKEN_LABEL}_pr256}
OUT_DIR=$OUT_ROOT/$DATASET
LOG_DIR=$OUT_ROOT/logs
FULL_SFT_DIR=$OUT_DIR/full_sft
SPLIT_DIR=$OUT_DIR/shards/std
PART_DIR=$FULL_SFT_DIR/shards

WORKERS=${ADP_CONDENSER_WORKERS:-12}
LLM_CONCURRENCY=${ADP_CONDENSER_LLM_CONCURRENCY:-5}
MAX_IN_FLIGHT_ROWS=${ADP_CONDENSER_MAX_IN_FLIGHT_ROWS:-50}
ROW_TIMEOUT=${ADP_CONDENSER_ROW_TIMEOUT:-1800}

mkdir -p "$OUT_DIR" "$LOG_DIR" "$FULL_SFT_DIR" "$SPLIT_DIR" "$PART_DIR"

if [ -f "$EXP_ROOT/.env" ]; then
  set -a
  . "$EXP_ROOT/.env"
  set +a
fi

if [ -z "${LLM_MODEL:-}" ]; then
  echo "LLM_MODEL must be set in $EXP_ROOT/.env or the environment" >&2
  exit 1
fi
if [ -z "${LLM_API_KEY:-}" ]; then
  echo "LLM_API_KEY must be set in $EXP_ROOT/.env or the environment" >&2
  exit 1
fi

STD_JSONL=$STD_ROOT/$DATASET/full_std.jsonl
CONDENSER_JSONL=$FULL_SFT_DIR/full_sft_openhands_sdk_condensed_${TOKEN_LABEL}.jsonl
CONDENSER_TMP=$CONDENSER_JSONL.tmp
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
repo_branch=$(git -C "$REPO" branch --show-current 2>/dev/null || true)
repo_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)
repo_commit_short=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)

echo "dataset=$DATASET"
echo "repo=$REPO"
echo "repo_branch=$repo_branch"
echo "repo_commit=$repo_commit_short"
echo "std_jsonl=$STD_JSONL"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"
echo "llm_model=$LLM_MODEL"
echo "python=$PYTHON"
echo "max_tokens=$MAX_TOKENS"
echo "token_label=$TOKEN_LABEL"
echo "workers=$WORKERS"
echo "llm_concurrency_per_worker=$LLM_CONCURRENCY"
echo "total_llm_concurrency=$((WORKERS * LLM_CONCURRENCY))"
echo "max_in_flight_rows_per_worker=$MAX_IN_FLIGHT_ROWS"
echo "row_timeout=$ROW_TIMEOUT"

EXPECTED_ADP_BRANCH=${ADP_EXPECTED_BRANCH:-main}
if [ -n "$EXPECTED_ADP_BRANCH" ] && [ "$repo_branch" != "$EXPECTED_ADP_BRANCH" ]; then
  echo "Expected ADP repo branch $EXPECTED_ADP_BRANCH, got $repo_branch" >&2
  exit 1
fi

EXPECTED_ADP_COMMIT=${ADP_EXPECTED_COMMIT:-}
if [ -n "$EXPECTED_ADP_COMMIT" ] && [ "${repo_commit#$EXPECTED_ADP_COMMIT}" = "$repo_commit" ]; then
  echo "Expected ADP commit prefix $EXPECTED_ADP_COMMIT, got $repo_commit" >&2
  exit 1
fi

if [ ! -s "$STD_JSONL" ]; then
  echo "Standardized JSONL not found or empty: $STD_JSONL" >&2
  exit 1
fi

std_lines=$(wc -l < "$STD_JSONL" 2>/dev/null || echo 0)
echo "std_lines=$std_lines"

if [ -s "$CONDENSER_JSONL" ]; then
  cond_status=0
  cond_lines=$(wc -l < "$CONDENSER_JSONL" 2>/dev/null || echo 0)
  echo "condensation_status=0 condensation_lines=$cond_lines reused=$CONDENSER_JSONL"
else
  if [ -e "$CONDENSER_TMP" ]; then
    archived="$CONDENSER_TMP.pre_sharded_$(date +%Y%m%dT%H%M%S)"
    mv "$CONDENSER_TMP" "$archived"
    echo "archived_existing_tmp=$archived"
  fi
  rm -f "$SPLIT_DIR"/std_shard_*.jsonl
  rm -f "$PART_DIR"/part_*.jsonl.tmp "$PART_DIR"/part_*.jsonl

  "$PYTHON" - "$STD_JSONL" "$SPLIT_DIR" "$WORKERS" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
workers = int(sys.argv[3])
handles = [
    (out_dir / f"std_shard_{idx:02d}.jsonl").open("w", encoding="utf-8")
    for idx in range(workers)
]
try:
    with src.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            handles[line_no % workers].write(line)
finally:
    for handle in handles:
        handle.close()
PY

  pids=()
  for shard in $(seq 0 $((WORKERS - 1))); do
    shard_id=$(printf "%02d" "$shard")
    shard_input="$SPLIT_DIR/std_shard_${shard_id}.jsonl"
    part_tmp="$PART_DIR/part_${shard_id}.jsonl.tmp"
    log_file="$LOG_DIR/${DATASET}.shard_${shard_id}.openhands_sdk_condensation.stderr"
    (
      cd "$REPO"
      ADP_USE_SOURCE_ROW_HASH=1 \
      MY_DATASET="$DATASET" \
      PYTHONPATH="$REPO:${PYTHONPATH:-}" \
        "$PYTHON" agents/openhands_sdk/condensation_sft.py \
          --max-tokens "$MAX_TOKENS" \
          --model "$LLM_MODEL" \
          --max-in-flight-rows "$MAX_IN_FLIGHT_ROWS" \
          --llm-concurrency "$LLM_CONCURRENCY" \
          --row-timeout "$ROW_TIMEOUT" \
          --continue-on-error \
          < "$shard_input" >> "$part_tmp" 2>> "$log_file"
    ) &
    pids+=("$!")
    pid_index=$((${#pids[@]} - 1))
    echo "started_shard=$shard_id pid=${pids[$pid_index]} input=$shard_input output=$part_tmp"
  done

  cond_status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      cond_status=1
    fi
  done

  if [ "$cond_status" -eq 0 ]; then
    rm -f "$CONDENSER_TMP"
    for shard in $(seq 0 $((WORKERS - 1))); do
      shard_id=$(printf "%02d" "$shard")
      cat "$PART_DIR/part_${shard_id}.jsonl.tmp" >> "$CONDENSER_TMP"
      mv "$PART_DIR/part_${shard_id}.jsonl.tmp" "$PART_DIR/part_${shard_id}.jsonl"
    done
    cond_lines=$(wc -l < "$CONDENSER_TMP" 2>/dev/null || echo 0)
    echo "condensation_status=$cond_status condensation_lines=$cond_lines"
    if [ "$cond_lines" -gt 0 ]; then
      mv "$CONDENSER_TMP" "$CONDENSER_JSONL"
    else
      echo "condensation_sft produced no rows; keeping empty partial $CONDENSER_TMP" >&2
      cond_status=1
    fi
  else
    cond_lines=$(find "$PART_DIR" -name 'part_*.jsonl.tmp' -print0 | xargs -0 cat 2>/dev/null | wc -l)
    echo "condensation_status=$cond_status partial_condensation_lines=$cond_lines" >&2
    echo "one or more shards failed; leaving part files in $PART_DIR" >&2
  fi
fi

finished_at=$(date -Is)
cat > "$MANIFEST" <<JSON
{
  "dataset": "$DATASET",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "repo": "$REPO",
  "repo_branch": "$repo_branch",
  "repo_commit": "$repo_commit",
  "std_lines": $std_lines,
  "condensation_status": $cond_status,
  "condensation_lines": $cond_lines,
  "max_tokens": $MAX_TOKENS,
  "token_label": "$TOKEN_LABEL",
  "llm_model": "$LLM_MODEL",
  "workers": $WORKERS,
  "llm_concurrency_per_worker": $LLM_CONCURRENCY,
  "max_in_flight_rows_per_worker": $MAX_IN_FLIGHT_ROWS,
  "std_jsonl": "$STD_JSONL",
  "condensed_openhands_sdk_jsonl": "$CONDENSER_JSONL"
}
JSON

echo "finished_at=$finished_at"
if [ "$cond_status" -ne 0 ] || [ "$cond_lines" -eq 0 ]; then
  exit 1
fi

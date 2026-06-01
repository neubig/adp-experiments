#!/usr/bin/env bash

EXP_ROOT=/home/gneubig/exp/adp
BASE=$EXP_ROOT/datasets/condenser_sft/openhands_nonweb_12k_llm
REPO=/home/gneubig/work/adp/agent-data-protocol-pr244
SCRIPT=/home/gneubig/work/adp/adp-experiments/openhands_sdk_training/scripts/run_condenser_sft_batch.py
JOBS=${JOBS:-4}
SHARD_LINES=${SHARD_LINES:-500}

mkdir -p "$BASE/generated" "$BASE/errors" "$BASE/stats" "$BASE/run_logs" "$BASE/shards"
rm -f "$BASE/tasks.tsv"

if [ -f "$EXP_ROOT/.env" ]; then
  set -a
  . "$EXP_ROOT/.env"
  set +a
fi

if [ -z "${LLM_MODEL:-}" ]; then
  echo 'LLM_MODEL is not set; put it in ~/exp/adp/.env or export it before launching.' >&2
  exit 1
fi
if [ -z "${LLM_API_KEY:-}" ]; then
  echo 'LLM_API_KEY is not set; put it in ~/exp/adp/.env or export it before launching.' >&2
  exit 1
fi

if [ ! -f "$BASE/prepare_manifest.json" ]; then
  echo "Missing $BASE/prepare_manifest.json; run scripts/prepare_condenser_sft.py first." >&2
  exit 1
fi

if compgen -G "$BASE/metadata/*.metadata.json" > /dev/null; then
  for metadata in "$BASE"/metadata/*.metadata.json; do
    dataset=$(basename "$metadata" .metadata.json)
    target="$REPO/datasets/$dataset/metadata.json"
    if [ ! -f "$target" ]; then
      cp "$metadata" "$target"
    fi
  done
fi


for input in "$BASE"/standardized/*.jsonl; do
  name=$(basename "$input" .jsonl)
  split=${name%%_*}
  dataset=${name#${split}_}
  shard_dir="$BASE/shards/$name"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"
  line_count=$(wc -l < "$input")
  if [ "$line_count" -le "$SHARD_LINES" ]; then
    ln -s "$input" "$shard_dir/shard_000.jsonl"
  else
    split -l "$SHARD_LINES" -d -a 3 --additional-suffix=.jsonl "$input" "$shard_dir/shard_"
  fi
  for shard in "$shard_dir"/*.jsonl; do
    shard_name=$(basename "$shard" .jsonl)
    task_name="${name}_${shard_name}"
    output="$BASE/generated/${task_name}.jsonl"
    errors="$BASE/errors/${task_name}.errors.jsonl"
    stats="$BASE/stats/${task_name}.stats.json"
    log="$BASE/run_logs/${task_name}.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$dataset" "$shard" "$output" "$errors" "$stats" "$log" >> "$BASE/tasks.tsv"
  done
done

export REPO SCRIPT LLM_MODEL
cat "$BASE/tasks.tsv" | xargs -P "$JOBS" -n 6 bash -c '
  dataset="$1"
  input="$2"
  output="$3"
  errors="$4"
  stats="$5"
  log="$6"
  (
    cd "$REPO" || exit 1
    PYTHONPATH="$REPO:${PYTHONPATH:-}" python "$SCRIPT" \
      --dataset-name "$dataset" \
      --input "$input" \
      --output "$output" \
      --errors "$errors" \
      --stats "$stats" \
      --max-tokens 12000 \
      --model "$LLM_MODEL"
  ) > "$log" 2>&1
' _

#!/usr/bin/env bash

DATASET=$1
if [ -z "$DATASET" ]; then
  echo "Usage: $0 DATASET_NAME" >&2
  exit 2
fi

REPO=${ADP_REPO:-/home/gneubig/work/adp/agent-data-protocol-pr244}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
PYTHON=${ADP_PYTHON:-/home/gneubig/work/adp/agent-data-protocol-pr244/.venv/bin/python}
OUT_ROOT=${ADP_OUT_ROOT:-$EXP_ROOT/datasets/software_agent_condenser_12k}
OUT_DIR=$OUT_ROOT/$DATASET
LOG_DIR=$OUT_ROOT/logs
FULL_SFT_DIR=$OUT_DIR/full_sft

mkdir -p "$OUT_DIR" "$LOG_DIR" "$FULL_SFT_DIR"

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

RAW_JSONL=$OUT_DIR/full_raw.jsonl
STD_JSONL=$OUT_DIR/full_std.jsonl
CONDENSER_JSONL=$FULL_SFT_DIR/full_sft_openhands_sdk_condensed_12k.jsonl
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
echo "dataset=$DATASET"
echo "repo=$REPO"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"
echo "llm_model=$LLM_MODEL"

if [ ! -d "$REPO/datasets/$DATASET" ]; then
  echo "Dataset directory not found: $REPO/datasets/$DATASET" >&2
  exit 1
fi

(
  cd "$REPO/datasets/$DATASET" || exit 1
  env -u PYTHONPATH "$PYTHON" extract_raw.py
) > "$RAW_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.extract_raw.stderr"
extract_status=$?
raw_lines=$(wc -l < "$RAW_JSONL.tmp" 2>/dev/null || echo 0)
echo "extract_status=$extract_status raw_lines=$raw_lines"
if [ "$extract_status" -ne 0 ] || [ "$raw_lines" -eq 0 ]; then
  echo "extract_raw failed or produced no rows" >&2
  exit 1
fi
mv "$RAW_JSONL.tmp" "$RAW_JSONL"

(
  cd "$REPO/datasets/$DATASET" || exit 1
  PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" raw_to_standardized.py < "$RAW_JSONL"
) > "$STD_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.raw_to_standardized.stderr"
std_status=$?
std_lines=$(wc -l < "$STD_JSONL.tmp" 2>/dev/null || echo 0)
echo "std_status=$std_status std_lines=$std_lines"
if [ "$std_status" -ne 0 ] || [ "$std_lines" -eq 0 ]; then
  echo "raw_to_standardized failed or produced no rows" >&2
  exit 1
fi
mv "$STD_JSONL.tmp" "$STD_JSONL"

(
  cd "$REPO" || exit 1
  MY_DATASET="$DATASET" PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" \
    agents/openhands_sdk/condensation_sft.py \
      --max-tokens 12000 \
      --model "$LLM_MODEL" \
      < "$STD_JSONL"
) > "$CONDENSER_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.openhands_sdk_condensation.stderr"
cond_status=$?
cond_lines=$(wc -l < "$CONDENSER_JSONL.tmp" 2>/dev/null || echo 0)
echo "condensation_status=$cond_status condensation_lines=$cond_lines"
if [ "$cond_status" -eq 0 ] && [ "$cond_lines" -gt 0 ]; then
  mv "$CONDENSER_JSONL.tmp" "$CONDENSER_JSONL"
else
  echo "condensation_sft failed or produced no rows" >&2
  rm -f "$CONDENSER_JSONL.tmp"
fi

finished_at=$(date -Is)
cat > "$MANIFEST" <<JSON
{
  "dataset": "$DATASET",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "raw_lines": $raw_lines,
  "std_lines": $std_lines,
  "condensation_status": $cond_status,
  "condensation_lines": $cond_lines,
  "max_tokens": 12000,
  "llm_model": "$LLM_MODEL",
  "raw_jsonl": "$RAW_JSONL",
  "std_jsonl": "$STD_JSONL",
  "condensed_openhands_sdk_jsonl": "$CONDENSER_JSONL"
}
JSON

echo "finished_at=$finished_at"
if [ "$cond_status" -ne 0 ] || [ "$cond_lines" -eq 0 ]; then
  exit 1
fi

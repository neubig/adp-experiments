#!/usr/bin/env bash
set -uo pipefail

DATASET=${1:-}
if [ -z "$DATASET" ]; then
  echo "Usage: $0 DATASET_NAME" >&2
  exit 2
fi

DEFAULT_REPO=/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/agent-data-protocol
if [ ! -d "$DEFAULT_REPO" ]; then
  DEFAULT_REPO=/home/gneubig/work/adp/agent-data-protocol-pr244
fi

REPO=${ADP_REPO:-$DEFAULT_REPO}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
PYTHON=${ADP_PYTHON:-$REPO/.venv/bin/python}
if [ ! -x "$PYTHON" ]; then
  PYTHON=/home/gneubig/work/adp/.venvs/openhands_sdk_training/bin/python
fi
MAX_TOKENS=${ADP_MAX_TOKENS:-24000}
TOKEN_LABEL=${ADP_TOKEN_LABEL:-}
if [ -z "$TOKEN_LABEL" ]; then
  TOKEN_LABEL=$((MAX_TOKENS / 1000))k
fi
STD_ROOT=${ADP_STD_OUT_ROOT:-$EXP_ROOT/datasets/all_agent_standardized_pr256}
OUT_ROOT=${ADP_COND_OUT_ROOT:-$EXP_ROOT/datasets/all_agent_condenser_${TOKEN_LABEL}_pr256}
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

STD_JSONL=$STD_ROOT/$DATASET/full_std.jsonl
CONDENSER_JSONL=$FULL_SFT_DIR/full_sft_openhands_sdk_condensed_${TOKEN_LABEL}.jsonl
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
echo "dataset=$DATASET"
echo "repo=$REPO"
echo "std_jsonl=$STD_JSONL"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"
echo "max_tokens=$MAX_TOKENS"
echo "token_label=$TOKEN_LABEL"
echo "llm_model=$LLM_MODEL"
echo "python=$PYTHON"
repo_branch=$(git -C "$REPO" branch --show-current 2>/dev/null || true)
repo_commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
echo "repo_branch=$repo_branch"
echo "repo_commit=$repo_commit"

if [ ! -s "$STD_JSONL" ]; then
  echo "Missing standardized input: $STD_JSONL" >&2
  exit 1
fi
std_lines=$(wc -l < "$STD_JSONL" 2>/dev/null || echo 0)

if [ -s "$CONDENSER_JSONL" ]; then
  cond_status=0
  cond_lines=$(wc -l < "$CONDENSER_JSONL" 2>/dev/null || echo 0)
  echo "condensation_status=0 condensation_lines=$cond_lines reused=$CONDENSER_JSONL"
else
  CONDENSER_TMP="$CONDENSER_JSONL.tmp"
  RESUME_STD_JSONL="$OUT_DIR/full_std.resume.jsonl"
  if [ -s "$CONDENSER_TMP" ]; then
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" - "$STD_JSONL" "$CONDENSER_TMP" "$RESUME_STD_JSONL" <<'PY_INNER'
import json
import sys
from pathlib import Path

std_path = Path(sys.argv[1])
partial_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
processed_ids = set()
processed_rows = set()
with partial_path.open(errors="replace") as handle:
    for line in handle:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        metadata = row.get("metadata", {})
        source_row_id = metadata.get("source_row_id")
        source_id = metadata.get("source_trajectory_id")
        if source_row_id:
            processed_rows.add(source_row_id)
        elif source_id:
            processed_ids.add(source_id)
with std_path.open() as in_handle, out_path.open("w") as out_handle:
    kept = skipped = 0
    for line in in_handle:
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = row.get("id")
        if row_id in processed_rows or row_id in processed_ids:
            skipped += 1
            continue
        out_handle.write(line if line.endswith("\n") else line + "\n")
        kept += 1
print(
    f"resume_processed_ids={len(processed_ids)} "
    f"resume_processed_rows={len(processed_rows)} "
    f"resume_skipped={skipped} resume_remaining={kept}",
    flush=True,
)
PY_INNER
    COND_INPUT="$RESUME_STD_JSONL"
  else
    : > "$CONDENSER_TMP"
    COND_INPUT="$STD_JSONL"
  fi

  remaining_lines=$(wc -l < "$COND_INPUT" 2>/dev/null || echo 0)
  if [ "$remaining_lines" -eq 0 ]; then
    cond_status=0
    cond_lines=$(wc -l < "$CONDENSER_TMP" 2>/dev/null || echo 0)
    echo "condensation_status=0 condensation_lines=$cond_lines resumed_complete=1"
    mv "$CONDENSER_TMP" "$CONDENSER_JSONL"
  else
    (
      cd "$REPO" || exit 1
      MY_DATASET="$DATASET" PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" \
        agents/openhands_sdk/condensation_sft.py \
          --max-tokens "$MAX_TOKENS" \
          --model "$LLM_MODEL" \
          --concurrency "${ADP_CONDENSER_CONCURRENCY:-8}" \
          --chunk-size "${ADP_CONDENSER_CHUNK_SIZE:-8}" \
          --continue-on-error \
          < "$COND_INPUT"
    ) >> "$CONDENSER_TMP" 2>> "$LOG_DIR/${DATASET}.openhands_sdk_condensation.stderr"
    cond_status=$?
    cond_lines=$(wc -l < "$CONDENSER_TMP" 2>/dev/null || echo 0)
    echo "condensation_status=$cond_status condensation_lines=$cond_lines"
    if [ "$cond_status" -eq 0 ] && [ "$cond_lines" -gt 0 ]; then
      mv "$CONDENSER_TMP" "$CONDENSER_JSONL"
    else
      echo "condensation_sft failed or produced no rows; keeping partial $CONDENSER_TMP" >&2
    fi
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
  "std_jsonl": "$STD_JSONL",
  "condensed_openhands_sdk_jsonl": "$CONDENSER_JSONL"
}
JSON

echo "finished_at=$finished_at"
if [ "$cond_status" -ne 0 ] || [ "$cond_lines" -eq 0 ]; then
  exit 1
fi

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
OUT_ROOT=${ADP_STD_OUT_ROOT:-$EXP_ROOT/datasets/all_agent_standardized_pr256}
OUT_DIR=$OUT_ROOT/$DATASET
LOG_DIR=$OUT_ROOT/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

RAW_JSONL=$OUT_DIR/full_raw.jsonl
ATIF_JSONL=$OUT_DIR/full_atif.jsonl
STD_JSONL=$OUT_DIR/full_std.jsonl
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
echo "dataset=$DATASET"
echo "repo=$REPO"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"
echo "python=$PYTHON"
repo_branch=$(git -C "$REPO" branch --show-current 2>/dev/null || true)
repo_commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
echo "repo_branch=$repo_branch"
echo "repo_commit=$repo_commit"

if [ ! -d "$REPO/datasets/$DATASET" ]; then
  echo "Dataset directory not found: $REPO/datasets/$DATASET" >&2
  exit 1
fi

if [ -s "$RAW_JSONL" ]; then
  extract_status=0
  raw_lines=$(wc -l < "$RAW_JSONL" 2>/dev/null || echo 0)
  echo "extract_status=0 raw_lines=$raw_lines reused=$RAW_JSONL"
else
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
fi

if [ -s "$STD_JSONL" ]; then
  atif_status=0
  atif_lines=$(wc -l < "$ATIF_JSONL" 2>/dev/null || echo 0)
  std_status=0
  std_lines=$(wc -l < "$STD_JSONL" 2>/dev/null || echo 0)
  echo "std_status=0 std_lines=$std_lines reused=$STD_JSONL"
elif [ -f "$REPO/datasets/$DATASET/raw_to_standardized.py" ]; then
  atif_status=0
  atif_lines=0
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
else
  if [ -s "$ATIF_JSONL" ]; then
    atif_status=0
    atif_lines=$(wc -l < "$ATIF_JSONL" 2>/dev/null || echo 0)
    echo "atif_status=0 atif_lines=$atif_lines reused=$ATIF_JSONL"
  else
    (
      cd "$REPO/datasets/$DATASET" || exit 1
      PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" raw_to_atif.py < "$RAW_JSONL"
    ) > "$ATIF_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.raw_to_atif.stderr"
    atif_status=$?
    atif_lines=$(wc -l < "$ATIF_JSONL.tmp" 2>/dev/null || echo 0)
    echo "atif_status=$atif_status atif_lines=$atif_lines"
    if [ "$atif_status" -ne 0 ] || [ "$atif_lines" -eq 0 ]; then
      echo "raw_to_atif failed or produced no rows" >&2
      exit 1
    fi
    mv "$ATIF_JSONL.tmp" "$ATIF_JSONL"
  fi

  (
    cd "$REPO/datasets/$DATASET" || exit 1
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" atif_to_std.py < "$ATIF_JSONL"
  ) > "$STD_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.atif_to_std.stderr"
  std_status=$?
  std_lines=$(wc -l < "$STD_JSONL.tmp" 2>/dev/null || echo 0)
  echo "std_status=$std_status std_lines=$std_lines"
  if [ "$std_status" -ne 0 ] || [ "$std_lines" -eq 0 ]; then
    echo "atif_to_std failed or produced no rows" >&2
    exit 1
  fi
  mv "$STD_JSONL.tmp" "$STD_JSONL"
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
  "extract_status": $extract_status,
  "raw_lines": $raw_lines,
  "atif_status": $atif_status,
  "atif_lines": $atif_lines,
  "std_status": $std_status,
  "std_lines": $std_lines,
  "raw_jsonl": "$RAW_JSONL",
  "atif_jsonl": "$ATIF_JSONL",
  "std_jsonl": "$STD_JSONL"
}
JSON

echo "finished_at=$finished_at"

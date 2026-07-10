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
OUT_ROOT=${ADP_STD_OUT_ROOT:-$EXP_ROOT/datasets/software_agent_standardized_pr256}
RAW_FALLBACK_ROOTS=${ADP_RAW_FALLBACK_ROOTS:-$EXP_ROOT/datasets/software_agent_condenser_24k:$EXP_ROOT/datasets/software_agent_condenser_12k:$EXP_ROOT/datasets/software_agent_pipeline}
OUT_DIR=$OUT_ROOT/$DATASET
LOG_DIR=$OUT_ROOT/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

RAW_JSONL=$OUT_DIR/full_raw.jsonl
ATIF_JSONL=$OUT_DIR/full_atif.jsonl
STD_JSONL=$OUT_DIR/full_std.jsonl
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
repo_branch=$(git -C "$REPO" branch --show-current 2>/dev/null || true)
repo_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)
repo_commit_short=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)

echo "dataset=$DATASET"
echo "repo=$REPO"
echo "repo_branch=$repo_branch"
echo "repo_commit=$repo_commit_short"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"
echo "python=$PYTHON"

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

if [ ! -d "$REPO/datasets/$DATASET" ]; then
  echo "Dataset directory not found: $REPO/datasets/$DATASET" >&2
  exit 1
fi
for script_name in extract_raw.py raw_to_atif.py atif_to_std.py; do
  if [ ! -f "$REPO/datasets/$DATASET/$script_name" ]; then
    echo "Missing required script: $REPO/datasets/$DATASET/$script_name" >&2
    exit 1
  fi
done

install_dataset_requirements() {
  req_file=$REPO/datasets/$DATASET/requirements.txt
  if [ ! -s "$req_file" ]; then
    return 0
  fi
  req_hash=$(sha256sum "$req_file" | awk '{print $1}')
  marker_dir=$REPO/.venv/.adp_dataset_requirements
  marker=$marker_dir/$DATASET.$req_hash
  lock_file=$marker_dir/install.lock
  mkdir -p "$marker_dir"
  if [ -e "$marker" ]; then
    echo "dataset_requirements=already_installed file=$req_file"
    return 0
  fi
  (
    flock 9
    if [ ! -e "$marker" ]; then
      echo "dataset_requirements=installing file=$req_file"
      if "$PYTHON" -m pip --version >/dev/null 2>&1; then
        "$PYTHON" -m pip install -r "$req_file"
      else
        uv pip install --python "$PYTHON" -r "$req_file"
      fi
      touch "$marker"
    fi
  ) 9>"$lock_file"
}

copy_raw_fallback() {
  old_ifs=$IFS
  IFS=:
  for root in $RAW_FALLBACK_ROOTS; do
    candidate=$root/$DATASET/full_raw.jsonl
    if [ -s "$candidate" ]; then
      echo "raw_fallback=$candidate"
      cp "$candidate" "$RAW_JSONL"
      IFS=$old_ifs
      return 0
    fi
  done
  IFS=$old_ifs
  return 1
}

if [ -s "$RAW_JSONL" ]; then
  extract_status=0
  raw_lines=$(wc -l < "$RAW_JSONL" 2>/dev/null || echo 0)
  echo "extract_status=0 raw_lines=$raw_lines reused=$RAW_JSONL"
else
  install_dataset_requirements
  set +e
  (
    cd "$REPO/datasets/$DATASET"
    env -u PYTHONPATH "$PYTHON" extract_raw.py
  ) > "$RAW_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.extract_raw.stderr"
  extract_status=$?
  set -e
  raw_lines=$(wc -l < "$RAW_JSONL.tmp" 2>/dev/null || echo 0)
  echo "extract_status=$extract_status raw_lines=$raw_lines"
  if [ "$extract_status" -ne 0 ] || [ "$raw_lines" -eq 0 ]; then
    echo "extract_raw failed or produced no rows; trying raw fallback roots" >&2
    rm -f "$RAW_JSONL.tmp"
    if copy_raw_fallback; then
      extract_status=0
      raw_lines=$(wc -l < "$RAW_JSONL" 2>/dev/null || echo 0)
      echo "extract_status=0 raw_lines=$raw_lines fallback=1"
    else
      echo "extract_raw failed or produced no rows, and no raw fallback was found" >&2
      exit 1
    fi
  else
    mv "$RAW_JSONL.tmp" "$RAW_JSONL"
  fi
fi

if [ -s "$ATIF_JSONL" ]; then
  atif_status=0
  atif_lines=$(wc -l < "$ATIF_JSONL" 2>/dev/null || echo 0)
  echo "atif_status=0 atif_lines=$atif_lines reused=$ATIF_JSONL"
else
  set +e
  (
    cd "$REPO/datasets/$DATASET"
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" raw_to_atif.py < "$RAW_JSONL"
  ) > "$ATIF_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.raw_to_atif.stderr"
  atif_status=$?
  set -e
  atif_lines=$(wc -l < "$ATIF_JSONL.tmp" 2>/dev/null || echo 0)
  echo "atif_status=$atif_status atif_lines=$atif_lines"
  if [ "$atif_status" -ne 0 ] || [ "$atif_lines" -eq 0 ]; then
    echo "raw_to_atif failed or produced no rows" >&2
    rm -f "$ATIF_JSONL.tmp"
    exit 1
  fi
  mv "$ATIF_JSONL.tmp" "$ATIF_JSONL"
fi

if [ -s "$STD_JSONL" ]; then
  std_status=0
  std_lines=$(wc -l < "$STD_JSONL" 2>/dev/null || echo 0)
  echo "std_status=0 std_lines=$std_lines reused=$STD_JSONL"
else
  set +e
  (
    cd "$REPO/datasets/$DATASET"
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" atif_to_std.py < "$ATIF_JSONL"
  ) > "$STD_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.atif_to_std.stderr"
  std_status=$?
  set -e
  std_lines=$(wc -l < "$STD_JSONL.tmp" 2>/dev/null || echo 0)
  echo "std_status=$std_status std_lines=$std_lines"
  if [ "$std_status" -ne 0 ] || [ "$std_lines" -eq 0 ]; then
    echo "atif_to_std failed or produced no rows" >&2
    rm -f "$STD_JSONL.tmp"
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
  "raw_lines": $raw_lines,
  "atif_lines": $atif_lines,
  "std_lines": $std_lines,
  "raw_jsonl": "$RAW_JSONL",
  "atif_jsonl": "$ATIF_JSONL",
  "std_jsonl": "$STD_JSONL"
}
JSON

echo "finished_at=$finished_at"

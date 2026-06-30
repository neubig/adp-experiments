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
CONDENSER_TMP=$CONDENSER_JSONL.tmp
SOURCE_ROW_HASH_MARKER=$OUT_DIR/.use_source_row_hash
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
  RESUME_STD_JSONL="$OUT_DIR/full_std.resume.jsonl"
  if [ ! -e "$CONDENSER_TMP" ]; then
    : > "$SOURCE_ROW_HASH_MARKER"
  fi
  if [ -e "$SOURCE_ROW_HASH_MARKER" ]; then
    export ADP_USE_SOURCE_ROW_HASH=1
    echo "source_row_hash=1"
  fi
  if [ -s "$CONDENSER_TMP" ]; then
    PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" - "$STD_JSONL" "$CONDENSER_TMP" "$RESUME_STD_JSONL" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

std_path = Path(sys.argv[1])
partial_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
use_source_row_hash = os.getenv("ADP_USE_SOURCE_ROW_HASH") == "1"

dedup_path = partial_path.with_name(partial_path.name + ".dedup")
seen_record_ids = set()
partial_sources = set()
completed_sources = set()
input_records = 0
kept_records = 0
deduped_records = 0
invalid_records = 0
with partial_path.open(errors="replace") as handle:
    with dedup_path.open("w") as dedup_handle:
        for line in handle:
            if not line.strip():
                continue
            input_records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue
            metadata = row.get("metadata", {})
            source_id = metadata.get("source_row_id") or metadata.get(
                "source_trajectory_id"
            )
            record_id = row.get("id")
            dedup_key = record_id or json.dumps(row, sort_keys=True, ensure_ascii=False)
            if dedup_key in seen_record_ids:
                deduped_records += 1
                continue
            seen_record_ids.add(dedup_key)
            if source_id:
                partial_sources.add(source_id)
                if metadata.get("record_type") == "trajectory":
                    completed_sources.add(source_id)
            dedup_handle.write(line if line.endswith("\n") else line + "\n")
            kept_records += 1
dedup_path.replace(partial_path)

def source_row_id(row, trajectory_id):
    canonical = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{trajectory_id}__row_{digest}"

def std_source_id(row):
    trajectory_id = row.get("trajectory_id") or row.get("id") or row.get("session_id")
    if use_source_row_hash and trajectory_id:
        return source_row_id(row, trajectory_id)
    return trajectory_id

with std_path.open() as in_handle, out_path.open("w") as out_handle:
    kept = skipped = missing_id = 0
    for line in in_handle:
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = std_source_id(row)
        if source_id and source_id in completed_sources:
            skipped += 1
            continue
        if not source_id:
            missing_id += 1
        out_handle.write(line if line.endswith("\n") else line + "\n")
        kept += 1
print(
    "resume_processed="
    f"{len(completed_sources)} resume_partial_sources={len(partial_sources)} "
    f"resume_skipped={skipped} resume_remaining={kept} "
    f"resume_missing_source_id={missing_id} partial_records={input_records} "
    f"partial_kept_records={kept_records} partial_deduped_records={deduped_records} "
    f"partial_invalid_records={invalid_records}",
    flush=True,
)
PY
    COND_INPUT="$RESUME_STD_JSONL"
  else
    : > "$CONDENSER_TMP"
    COND_INPUT="$STD_JSONL"
  fi

  remaining_lines=$(wc -l < "$COND_INPUT" 2>/dev/null || echo 0)
  echo "remaining_lines=$remaining_lines"
  if [ "$remaining_lines" -eq 0 ]; then
    cond_status=0
    cond_lines=$(wc -l < "$CONDENSER_TMP" 2>/dev/null || echo 0)
    echo "condensation_status=0 condensation_lines=$cond_lines resumed_complete=1"
    mv "$CONDENSER_TMP" "$CONDENSER_JSONL"
  else
    echo "max_in_flight_rows=${ADP_CONDENSER_MAX_IN_FLIGHT_ROWS:-500}"
    echo "llm_concurrency=${ADP_CONDENSER_LLM_CONCURRENCY:-50}"
    echo "row_timeout=${ADP_CONDENSER_ROW_TIMEOUT:-1800}"
    (
      cd "$REPO"
      MY_DATASET="$DATASET" PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" \
        agents/openhands_sdk/condensation_sft.py \
          --max-tokens "$MAX_TOKENS" \
          --model "$LLM_MODEL" \
          --max-in-flight-rows "${ADP_CONDENSER_MAX_IN_FLIGHT_ROWS:-500}" \
          --llm-concurrency "${ADP_CONDENSER_LLM_CONCURRENCY:-50}" \
          --row-timeout "${ADP_CONDENSER_ROW_TIMEOUT:-1800}" \
          --continue-on-error \
          < "$COND_INPUT"
    ) >> "$CONDENSER_TMP" 2>> "$LOG_DIR/${DATASET}.openhands_sdk_condensation.stderr"
    cond_status=$?
    if [ "$cond_status" -eq 0 ]; then
      PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" - "$CONDENSER_TMP" <<'PY'
import json
import sys
from pathlib import Path

partial_path = Path(sys.argv[1])
dedup_path = partial_path.with_name(partial_path.name + ".dedup")
seen = set()
with partial_path.open(errors="replace") as in_handle, dedup_path.open("w") as out_handle:
    for line in in_handle:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("id") or json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out_handle.write(line if line.endswith("\n") else line + "\n")
dedup_path.replace(partial_path)
PY
    fi
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

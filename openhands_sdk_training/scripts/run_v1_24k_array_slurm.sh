#!/usr/bin/env bash
#SBATCH --job-name=adp-v1-24k-array
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR=${ADP_EXPERIMENTS_SCRIPT_DIR:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/adp-experiments/openhands_sdk_training/scripts}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
DATASET_LIST=${ADP_DATASET_LIST:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/adp-experiments/openhands_sdk_training/dataset_lists/v1_remaining_24k.txt}
V1_ROOT=${ADP_V1_ROOT:-$EXP_ROOT/datasets/v1}
STAGE_ROOT=${ADP_STAGE_ROOT:-$EXP_ROOT/datasets/_v1_array_stage_pr256}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script must be run as a Slurm array job." >&2
  exit 2
fi

mapfile -t DATASETS < <(grep -vE '^[[:space:]]*(#|$)' "$DATASET_LIST")
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#DATASETS[@]}" ]; then
  echo "Array index $SLURM_ARRAY_TASK_ID outside dataset list length ${#DATASETS[@]}" >&2
  exit 2
fi

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
STD_ROOT=$STAGE_ROOT/standardized
COND_ROOT=$STAGE_ROOT/condensed_24k

export ADP_REPO=${ADP_REPO:-/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/agent-data-protocol}
export ADP_EXP_ROOT=$EXP_ROOT
export ADP_PYTHON=${ADP_PYTHON:-$ADP_REPO/.venv/bin/python}
export ADP_MAX_TOKENS=${ADP_MAX_TOKENS:-24000}
export ADP_TOKEN_LABEL=${ADP_TOKEN_LABEL:-24k}
export ADP_CONDENSER_WORKERS=${ADP_CONDENSER_WORKERS:-12}
export ADP_CONDENSER_LLM_CONCURRENCY=${ADP_CONDENSER_LLM_CONCURRENCY:-20}
export ADP_CONDENSER_MAX_IN_FLIGHT_ROWS=${ADP_CONDENSER_MAX_IN_FLIGHT_ROWS:-50}
export ADP_CONDENSER_ROW_TIMEOUT=${ADP_CONDENSER_ROW_TIMEOUT:-1800}
export ADP_CONDENSER_LLM_RETRIES=${ADP_CONDENSER_LLM_RETRIES:-3}
export ADP_CONDENSER_LLM_RETRY_MIN_WAIT=${ADP_CONDENSER_LLM_RETRY_MIN_WAIT:-1}
export ADP_CONDENSER_LLM_RETRY_MAX_WAIT=${ADP_CONDENSER_LLM_RETRY_MAX_WAIT:-30}

echo "dataset=$DATASET"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "node=$(hostname)"
echo "started_at=$(date -Is)"
echo "dataset_list=$DATASET_LIST"
echo "stage_root=$STAGE_ROOT"
echo "v1_root=$V1_ROOT"

mkdir -p "$STD_ROOT" "$COND_ROOT" "$V1_ROOT"

if [ -e "$V1_ROOT/$DATASET/raw/manifest.json" ] && [ -e "$V1_ROOT/$DATASET/24k/manifest.json" ]; then
  echo "v1_complete=1 dataset=$DATASET"
  exit 0
fi

ADP_STD_OUT_ROOT="$STD_ROOT" "$SCRIPT_DIR/generate_pr256_standardized.sh" "$DATASET"
ADP_STD_OUT_ROOT="$STD_ROOT" \
ADP_COND_OUT_ROOT="$COND_ROOT" \
  "$SCRIPT_DIR/generate_pr256_openhands_sdk_condenser_sharded.sh" "$DATASET"

"$ADP_PYTHON" - "$DATASET" "$STD_ROOT" "$COND_ROOT" "$V1_ROOT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

dataset = sys.argv[1]
std_root = Path(sys.argv[2])
cond_root = Path(sys.argv[3])
v1_root = Path(sys.argv[4])

src_raw = std_root / dataset
src_24k = cond_root / dataset
dst_dataset = v1_root / dataset
dst_raw = dst_dataset / "raw"
dst_24k = dst_dataset / "24k"

if not (src_raw / "manifest.json").exists():
    raise SystemExit(f"missing raw manifest: {src_raw / 'manifest.json'}")
if not (src_24k / "manifest.json").exists():
    raise SystemExit(f"missing 24k manifest: {src_24k / 'manifest.json'}")
with (src_24k / "manifest.json").open() as handle:
    cond_manifest = json.load(handle)
if cond_manifest.get("condensation_status") != 0:
    raise SystemExit(f"condensation did not finish cleanly: {cond_manifest.get('condensation_status')}")

dst_dataset.mkdir(parents=True, exist_ok=True)
for src, dst in ((src_raw, dst_raw), (src_24k, dst_24k)):
    if dst.exists():
        raise SystemExit(f"destination already exists, refusing to overwrite: {dst}")
    shutil.move(str(src), str(dst))

def rewrite_manifest(path: Path) -> None:
    manifest = path / "manifest.json"
    with manifest.open() as handle:
        data = json.load(handle)
    replacements = {
        "raw_jsonl": path / "full_raw.jsonl",
        "atif_jsonl": path / "full_atif.jsonl",
        "std_jsonl": dst_raw / "full_std.jsonl",
    }
    for key, value in replacements.items():
        if key in data and value.exists():
            data[key] = str(value)
    if "condensed_openhands_sdk_jsonl" in data:
        output = path / "full_sft" / Path(data["condensed_openhands_sdk_jsonl"]).name
        if output.exists():
            data["condensed_openhands_sdk_jsonl"] = str(output)
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

rewrite_manifest(dst_raw)
rewrite_manifest(dst_24k)
print(f"moved_to_v1={dst_dataset}", flush=True)
PY

echo "finished_at=$(date -Is)"

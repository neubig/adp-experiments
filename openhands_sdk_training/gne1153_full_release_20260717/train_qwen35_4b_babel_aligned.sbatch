#!/usr/bin/env bash
#SBATCH --job-name=adpv2full24k-aligned-q35-4b-babel-1ep
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717/evidence/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUNTIME_ROOT=/home/gneubig/work/adp/adp-experiments/openhands_sdk_training
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
DATASET_ROOT=${RUN_ROOT}/dataset
MANIFEST=${DATASET_ROOT}/manifest.json
ALIGNMENT_MANIFEST=${DATASET_ROOT}/alignment_manifest.json
CONFIG_SOURCE=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/qwen35_4b_adpv2_full24k_aligned_babel_one_epoch.yaml
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
CONFIG=${RUN_ROOT}/evidence/qwen35_4b_adpv2_full24k_aligned_babel_one_epoch.runtime.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.yaml
PREPARE_CONFIG=${SOURCE_ROOT}/openhands_sdk_training/scripts/gne1153_prepare_restartable_config.py
PROBE=${SOURCE_ROOT}/openhands_sdk_training/scripts/gne1153_torch_distributed_probe.py
TOKENIZED=${RUN_ROOT}/tokenized_aligned_qwen35_4b_base_qwen3_5_nothink_seq32768
TOKENIZED_MANIFEST=${DATASET_ROOT}/tokenized_aligned_manifest.json
WATCHER=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/watch_tokenized_manifest.py
GPU_LOG=${RUN_ROOT}/evidence/gpu_${SLURM_JOB_ID}_restart_${RESTART_COUNT}.csv

python - "$MANIFEST" "$ALIGNMENT_MANIFEST" "$CONFIG_SOURCE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
a = json.load(open(sys.argv[2]))
assert m["status"] in {"training_projection_complete", "tokenized_complete"}, m["status"]
assert m["release"]["revision"] == "eb95b66ca30ef071d5e49495d5d780c29f9aef7b"
assert m["release"]["expected_config_count"] == 52
assert m["observed_config_count"] == 52
assert m["source_rows"] == 9640563
assert len(m["datasets"]) == 52
assert m["adapted_train_rows"] == 9196689
assert m["evaluation"]["removed_from_training"] == 0
projection = m["training_projection"]
assert projection["status"] == "complete"
assert projection["config_count"] == 52
assert projection["train_rows"] == m["adapted_train_rows"]
assert projection["evaluation"]["rows"] == 500
assert a["status"] == "alignment_complete"
assert a["source_adapted_rows"] == m["adapted_train_rows"]
assert a["source_config_count"] == 52
assert a["nonempty_config_count"] == 51
assert a["empty_dataset_names"] == ["adpv2_full24k_37"]
assert a["normalized_rows"] > 0
assert a["merged_messages"] >= a["normalized_rows"]
zero_trainable = [d for d in m["datasets"] if d["adapted_rows"] == 0]
assert [(d["index"], d["config_name"], d["source_rows_observed"], d["skipped_rows"]) for d in zero_trainable] == [
    (37, "omniact", 6789, 6789)
]
expected_train_datasets = [d["dataset_name"] for d in m["datasets"] if d["adapted_rows"] > 0]
dataset_line = next(
    line for line in open(sys.argv[3], encoding="utf-8") if line.startswith("dataset: ")
)
configured_train_datasets = dataset_line.removeprefix("dataset: ").strip().split(",")
assert configured_train_datasets == expected_train_datasets
assert len(configured_train_datasets) == 51
print("release_manifest_preflight=ok")
print(f"alignment_manifest_preflight=ok normalized_rows={a['normalized_rows']} merged_messages={a['merged_messages']}")
print("zero_trainable_config_excluded_from_loader=37:omniact:6789_source:6789_adapter_skips")
PY

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/wandb" "${RUN_ROOT}/wandb_cache"
EXECUTED_SCRIPT=${RUN_ROOT}/evidence/train_qwen35_4b_babel_aligned.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.sbatch
CONFIG_SNAPSHOT=${RUN_ROOT}/evidence/qwen35_4b_adpv2_full24k_aligned_babel_one_epoch.source.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.yaml
DS_SNAPSHOT=${RUN_ROOT}/evidence/ds_z3_config.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.json
cp -a "$0" "$EXECUTED_SCRIPT"
cp -a "$CONFIG_SOURCE" "$CONFIG_SNAPSHOT"
cp -a "${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/ds_z3_config.json" "$DS_SNAPSHOT"
sha256sum "$EXECUTED_SCRIPT" "$CONFIG_SNAPSHOT" "$DS_SNAPSHOT"

if [[ -d "$TOKENIZED" && ! -f "$TOKENIZED/dataset_dict.json" ]]; then
  mv "$TOKENIZED" "${RUN_ROOT}/evidence/$(basename "$TOKENIZED").incomplete.$(date +%Y%m%dT%H%M%S)"
fi

export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH
python "$PREPARE_CONFIG" --source "$CONFIG_SOURCE" --destination "$CONFIG"
cd "$RUNTIME_ROOT"

export HF_HOME=/home/gneubig/exp/adp-smoke/hf_cache
export WANDB_ENTITY=gneubig
export WANDB_PROJECT=adp-experiments
export WANDB_RUN_ID=adpv2-full24k-aligned-qwen35-4b-babel-1ep-eb95b66-20260717
export WANDB_NAME=adpv2-full-24k-aligned-qwen35-4b-base-babel-one-epoch-eb95b66-20260717
export WANDB_RESUME=allow
export WANDB_DIR=${RUN_ROOT}/wandb
export WANDB_CACHE_DIR=${RUN_ROOT}/wandb_cache
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 20000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/gne1153-adpv2-full24k-triton-${SLURM_JOB_ID}-${RESTART_COUNT}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
mkdir -p "$TRITON_CACHE_DIR"
unset CC CXX

echo "job_id=${SLURM_JOB_ID}"
echo "restart_count=${RESTART_COUNT}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "runtime_commit=$(git -C /home/gneubig/work/adp/adp-experiments rev-parse HEAD)"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"
echo "alignment_manifest=${ALIGNMENT_MANIFEST}"
echo "tokenized_path=${TOKENIZED}"
echo "output_dir=${RUN_ROOT}/output_aligned_qwen35_4b_base_full_sft_one_epoch"
echo "wandb_entity=${WANDB_ENTITY}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_run_id=${WANDB_RUN_ID}"
echo "wandb_name=${WANDB_NAME}"
echo "gpu_log=${GPU_LOG}"
nvidia-smi
nvidia-smi topo -m

(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 30
  done
) > "$GPU_LOG" 2>&1 &
MONITOR_PID=$!

python "$WATCHER" \
  --manifest "$MANIFEST" \
  --tokenized-path "$TOKENIZED" \
  --output "$TOKENIZED_MANIFEST" &
WATCHER_PID=$!

on_requeue() {
  echo "requeue_signal_received_at=$(date -Is)"
  scontrol requeue "$SLURM_JOB_ID"
}
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap on_requeue USR1
trap cleanup EXIT

echo "distributed_probe_started_at=$(date -Is)"
torchrun --standalone --nnodes=1 --nproc-per-node=4 "$PROBE"
echo "distributed_probe_finished_at=$(date -Is)"

set +e
llamafactory-cli train "$CONFIG"
STATUS=$?
set -e
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
if [[ "$STATUS" -eq 0 ]]; then
  wait "$WATCHER_PID"
fi
exit "$STATUS"

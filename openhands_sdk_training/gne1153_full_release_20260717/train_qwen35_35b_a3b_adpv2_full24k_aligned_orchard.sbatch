#!/usr/bin/env bash
#SBATCH --job-name=adpv2full24k-q35-35b-orchard-1ep
#SBATCH --partition=flame-earlybirds
#SBATCH --qos=earlybird_qos
#SBATCH --account=gneubig
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo/openhands_sdk_training/gne1153_full_release_20260717/%x-%j.out
#SBATCH --error=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo/openhands_sdk_training/gne1153_full_release_20260717/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
CODE_DIR=/home/gneubig/work/adp/adp-experiments/openhands_sdk_training
VENV_DIR=/home/gneubig/exp/adp-smoke/.venv
DATA_RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
DATASET_ROOT=${DATA_RUN_ROOT}/dataset
MANIFEST=${DATASET_ROOT}/manifest.json
ALIGNMENT_MANIFEST=${DATASET_ROOT}/alignment_manifest.json
TOKENIZED_MANIFEST=${DATASET_ROOT}/tokenized_aligned_manifest.json
TOKENIZED=${DATA_RUN_ROOT}/tokenized_aligned_qwen35_4b_base_qwen3_5_nothink_seq32768
RUN_ROOT=/project/flame/gneubig/adp/runs/openhands_sdk_training/adpv2_full_24k_eb95b66_20260717
MODEL_DIR=/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B
CONFIG_SOURCE=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/qwen35_35b_a3b_adpv2_full24k_aligned_orchard_one_epoch.yaml
DEEPSPEED_CONFIG=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/ds_z3_config_qwen35_hpz8.json
PREPARE_CONFIG=${SOURCE_ROOT}/openhands_sdk_training/gne1153_full_release_20260717/prepare_aligned_restartable_config.py
PATCH_HELPER=${CODE_DIR}/scripts/patch_llamafactory_liger_eval_skip_logits.py
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
CONFIG=${RUN_ROOT}/evidence/qwen35_35b_a3b_adpv2_full24k_aligned_orchard_one_epoch.runtime.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.yaml
CHECKPOINT_EVIDENCE=${RUN_ROOT}/evidence/checkpoint_selection.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.json
GPU_LOG=${RUN_ROOT}/evidence/gpu_${SLURM_JOB_ID}_restart_${RESTART_COUNT}.log

python3 - "$MANIFEST" "$ALIGNMENT_MANIFEST" "$TOKENIZED_MANIFEST" "$CONFIG_SOURCE" "$DEEPSPEED_CONFIG" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
a = json.load(open(sys.argv[2]))
t = json.load(open(sys.argv[3]))
cfg = open(sys.argv[4], encoding="utf-8").read()
ds = json.load(open(sys.argv[5]))
assert m["release"]["revision"] == "eb95b66ca30ef071d5e49495d5d780c29f9aef7b"
assert m["source_rows"] == 9_640_563
assert m["adapted_train_rows"] == 9_196_689
assert m["observed_config_count"] == 52
assert a["status"] == "alignment_complete"
assert a["source_adapted_rows"] == m["adapted_train_rows"]
assert a["source_config_count"] == 52
assert a["nonempty_config_count"] == 51
assert a["empty_dataset_names"] == ["adpv2_full24k_37"]
assert t["splits"] == {"train": 9_196_689, "validation": 500}
assert t["tokenizer_skipped_train_rows"] == 0
assert "num_train_epochs: 1.0" in cfg
assert "max_steps:" not in cfg
dataset_line = next(line for line in cfg.splitlines() if line.startswith("dataset: "))
datasets = dataset_line.removeprefix("dataset: ").split(",")
assert datasets == a["train_dataset_names"]
assert ds["zero_optimization"]["stage"] == 3
assert ds["zero_optimization"]["zero_hpz_partition_size"] == 8
print("full_release_preflight=ok")
print(f"aligned_rows={t['splits']['train']} tokenizer_skips=0 configs=52 nonempty_configs=51")
PY

mkdir -p "${RUN_ROOT}/evidence" "${RUN_ROOT}/wandb" "${RUN_ROOT}/wandb_cache" "${RUN_ROOT}/wandb_artifacts"
test -f "${MODEL_DIR}/config.json"
test -f "${TOKENIZED}/dataset_dict.json"
test -x "${VENV_DIR}/bin/llamafactory-cli"
test -d "${CODE_DIR}"

EXECUTED_SCRIPT=${RUN_ROOT}/evidence/train_qwen35_35b_a3b_orchard.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.sbatch
CONFIG_SNAPSHOT=${RUN_ROOT}/evidence/qwen35_35b_a3b_adpv2_full24k_aligned_orchard_one_epoch.source.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.yaml
DS_SNAPSHOT=${RUN_ROOT}/evidence/ds_z3_config_qwen35_hpz8.${SLURM_JOB_ID}.restart_${RESTART_COUNT}.json
cp -a "$0" "$EXECUTED_SCRIPT"
cp -a "$CONFIG_SOURCE" "$CONFIG_SNAPSHOT"
cp -a "$DEEPSPEED_CONFIG" "$DS_SNAPSHOT"
sha256sum "$EXECUTED_SCRIPT" "$CONFIG_SNAPSHOT" "$DS_SNAPSHOT" "$MANIFEST" "$ALIGNMENT_MANIFEST" "$TOKENIZED_MANIFEST"

source "${VENV_DIR}/bin/activate"
python "$PATCH_HELPER"
python "$PREPARE_CONFIG" \
  --source "$CONFIG_SOURCE" \
  --destination "$CONFIG" \
  --evidence "$CHECKPOINT_EVIDENCE" \
  --world-size 16 \
  --minimum-age-seconds 120
cd "$CODE_DIR"

export HF_HOME=/home/gneubig/exp/adp-smoke/hf_cache
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=gneubig
export WANDB_PROJECT=adp-experiments
export WANDB_RUN_ID=adpv2-full24k-aligned-qwen35-35b-orchard-1ep-eb95b66-20260717
export WANDB_NAME=adpv2-full-24k-aligned-qwen35-35b-a3b-orchard-one-epoch-eb95b66-20260717
export WANDB_RESUME=allow
export WANDB_DIR=${RUN_ROOT}/wandb
export WANDB_CACHE_DIR=${RUN_ROOT}/wandb_cache
export WANDB_ARTIFACT_DIR=${RUN_ROOT}/wandb_artifacts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/adpv2-qwen35-35b-triton-${SLURM_JOB_ID}-${RESTART_COUNT}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=1
unset CC CXX
mkdir -p "$TRITON_CACHE_DIR"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=$((30500 + SLURM_JOB_ID % 1000))
export NNODES=${SLURM_NNODES}
export NPROC_PER_NODE=8
export CONFIG CODE_DIR VENV_DIR GPU_LOG MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE

echo "job_id=${SLURM_JOB_ID}"
echo "restart_count=${RESTART_COUNT}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "nodes=${nodes[*]}"
echo "started_at=$(date -Is)"
echo "config=${CONFIG}"
echo "model_dir=${MODEL_DIR}"
echo "tokenized_path=${TOKENIZED}"
echo "output_dir=${RUN_ROOT}/output_qwen35_35b_a3b_hpz8_fa3_one_epoch"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "wandb_entity=${WANDB_ENTITY}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_run_id=${WANDB_RUN_ID}"
echo "wandb_name=${WANDB_NAME}"

on_requeue() {
  echo "requeue_signal_received_at=$(date -Is)"
  scontrol requeue "$SLURM_JOB_ID"
}
trap on_requeue USR1

set +e
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --gres=gpu:H100:8 --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  cd "${CODE_DIR}"
  source "${VENV_DIR}/bin/activate"
  echo "node=$(hostname) node_rank=${SLURM_NODEID} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi
  (
    while true; do
      date -Is
      echo "node=$(hostname) node_rank=${SLURM_NODEID}"
      nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits
      sleep 30
    done
  ) >> "${GPU_LOG}" 2>&1 &
  MONITOR_PID=$!
  cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; }
  trap cleanup EXIT
  export FORCE_TORCHRUN=1
  export NODE_RANK=${SLURM_NODEID}
  llamafactory-cli train "${CONFIG}"
'
STATUS=$?
set -e
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
exit "$STATUS"

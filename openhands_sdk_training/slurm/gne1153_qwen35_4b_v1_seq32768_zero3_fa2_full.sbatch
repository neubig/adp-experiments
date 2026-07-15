#!/usr/bin/env bash
#SBATCH --job-name=gne1153-qwen35-4b-full
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUNTIME_ROOT=/home/gneubig/work/adp/adp-experiments/openhands_sdk_training
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356
CONFIG_SOURCE=${SOURCE_ROOT}/openhands_sdk_training/configs/gne1153_qwen35_4b_20260715/qwen35_4b_v1_seq32768_zero3_fa2_full.yaml
CONFIG=${RUN_ROOT}/qwen35_4b_v1_seq32768_zero3_fa2_full.runtime.yaml
PREPARE_CONFIG=${SOURCE_ROOT}/openhands_sdk_training/scripts/gne1153_prepare_restartable_config.py
PROBE=${SOURCE_ROOT}/openhands_sdk_training/scripts/gne1153_torch_distributed_probe.py
TOKENIZED=${RUN_ROOT}/tokenized_full_qwen35_4b_seq32768_v1
LOG_DIR=${RUN_ROOT}/logs
GPU_LOG=${LOG_DIR}/gpu_monitor_full_${SLURM_JOB_ID}.csv

mkdir -p "$RUN_ROOT" "$LOG_DIR" "${RUN_ROOT}/evidence"
cp -a "$CONFIG_SOURCE" "${RUN_ROOT}/qwen35_4b_v1_seq32768_zero3_fa2_full.source.yaml"
cp -a "${SOURCE_ROOT}/openhands_sdk_training/configs/gne1153_qwen35_4b_20260715/ds_z3_config.json" "$RUN_ROOT/ds_z3_config.json"
cp -a "$0" "$RUN_ROOT/$(basename "$0")"

# LLaMA-Factory recognizes a saved DatasetDict by its metadata. Preserve any
# interrupted partial write under a timestamped name instead of loading it.
if [[ -d "$TOKENIZED" && ! -f "$TOKENIZED/dataset_dict.json" ]]; then
  mv "$TOKENIZED" "${RUN_ROOT}/evidence/$(basename "$TOKENIZED").incomplete.$(date +%Y%m%dT%H%M%S)"
fi

export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH
python "$PREPARE_CONFIG" --source "$CONFIG_SOURCE" --destination "$CONFIG"

cd "$RUNTIME_ROOT"

export HF_HOME=/home/gneubig/exp/adp-smoke/hf_cache
export WANDB_ENTITY=gneubig
export WANDB_PROJECT=adp-experiments
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 20000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/gne1153-triton-${SLURM_JOB_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
mkdir -p "$TRITON_CACHE_DIR"
unset CC CXX

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "runtime_commit=$(git -C /home/gneubig/work/adp/adp-experiments rev-parse HEAD)"
echo "source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
echo "source_dirty=$(git -C "$SOURCE_ROOT" status --short | wc -l)"
echo "config=${CONFIG}"
echo "tokenized_path=${TOKENIZED}"
echo "gpu_log=${GPU_LOG}"
echo "master_port=${MASTER_PORT}"
echo "nccl_p2p_disable=${NCCL_P2P_DISABLE}"
nvidia-smi
nvidia-smi topo -m
python - <<'PY'
import accelerate, datasets, deepspeed, flash_attn, torch, transformers
from transformers.utils import is_flash_attn_2_available
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
print("is_flash_attn_2_available", is_flash_attn_2_available())
PY

(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
    sleep 30
  done
) > "$GPU_LOG" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
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
exit "$STATUS"

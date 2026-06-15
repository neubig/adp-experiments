#!/bin/bash
#SBATCH --job-name=adp-qwen35-hpz8-nogc
#SBATCH --time=03:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=gneubig
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --exclude=orchard-flame-23
#SBATCH --output=/project/flame/gneubig/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_v2_adapted/logs/qwen35_35b_a3b_hpz8_no_gc_smoke_%j.out
#SBATCH --error=/project/flame/gneubig/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_v2_adapted/logs/qwen35_35b_a3b_hpz8_no_gc_smoke_%j.out

set -euo pipefail

CODE_DIR=/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/adp-experiments/openhands_sdk_training
VENV_DIR="${CODE_DIR}/.venv"
RUN_DIR=/project/flame/gneubig/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_v2_adapted
MODEL_DIR=/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B
CONFIG_YAML="${CODE_DIR}/configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_hpz8_no_gc_smoke.yaml"
OUTPUT_DIR="${RUN_DIR}/output_bench_qwen35_35b_a3b_hpz8_no_gc_seq32768_smoke"
LOG_DIR="${RUN_DIR}/logs"
GPU_LOG="${LOG_DIR}/gpu_monitor_qwen35_35b_a3b_hpz8_no_gc_smoke_${SLURM_JOB_ID}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${CODE_DIR}"
source "${VENV_DIR}/bin/activate"
python "${CODE_DIR}/scripts/patch_llamafactory_liger_eval_skip_logits.py"

export HF_HOME=${HF_HOME:-/project/flame/gneubig/adp/cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=${WANDB_ENTITY:-gneubig}
export WANDB_PROJECT=${WANDB_PROJECT:-adp-experiments}
export WANDB_NAME=${WANDB_NAME:-adp-bench-qwen35-35b-a3b-hpz8-no-gc-seq32768-smoke}
export WANDB_DIR=${WANDB_DIR:-${RUN_DIR}/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${RUN_DIR}/wandb_cache}
export WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-${RUN_DIR}/wandb_artifacts}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-8}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache-${SLURM_JOB_ID}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-4}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-1}
export ADP_LF_SKIP_FINAL_SAVE=1
unset CC CXX
mkdir -p "${TRITON_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT=${MASTER_PORT:-$((30500 + SLURM_JOB_ID % 1000))}
export NNODES="${SLURM_NNODES}"
export NPROC_PER_NODE=8
export CONFIG_YAML CODE_DIR VENV_DIR GPU_LOG

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "nodes=${nodes[*]}"
echo "started_at=$(date -Is)"
echo "config=${CONFIG_YAML}"
echo "model_dir=${MODEL_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpu_log=${GPU_LOG}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "nnodes=${NNODES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "wandb_entity=${WANDB_ENTITY}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_name=${WANDB_NAME}"
echo

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "prefetch_model_started_at=$(date -Is)"
  hf download Qwen/Qwen3.5-35B-A3B --local-dir "${MODEL_DIR}"
  echo "prefetch_model_finished_at=$(date -Is)"
fi

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --gres=gpu:8 --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  cd "${CODE_DIR}"
  source "${VENV_DIR}/bin/activate"
  echo "node=$(hostname) node_rank=${SLURM_NODEID} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi
  (
    while true; do
      date -Is
      echo "node=$(hostname) node_rank=${SLURM_NODEID}"
      nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
      sleep 30
    done
  ) >> "${GPU_LOG}" 2>&1 &
  MONITOR_PID=$!
  cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; }
  trap cleanup EXIT

  export FORCE_TORCHRUN=1
  export NNODES="${NNODES}"
  export NPROC_PER_NODE="${NPROC_PER_NODE}"
  export NODE_RANK="${SLURM_NODEID}"
  export MASTER_ADDR="${MASTER_ADDR}"
  export MASTER_PORT="${MASTER_PORT}"

  llamafactory-cli train "${CONFIG_YAML}"
'

STATUS=$?
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
exit "${STATUS}"

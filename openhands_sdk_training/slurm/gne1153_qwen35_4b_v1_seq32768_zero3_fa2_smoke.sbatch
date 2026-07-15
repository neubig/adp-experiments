#!/usr/bin/env bash
#SBATCH --job-name=gne1153-qwen35-4b-smoke
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356/%x-%j.err

set -euo pipefail

SOURCE_ROOT=/home/gneubig/homework/delegated/babel-gne1153-agentserver-xC2x1IyV/repo
RUNTIME_ROOT=/home/gneubig/work/adp/adp-experiments/openhands_sdk_training
RUN_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356
CONFIG_SOURCE=${SOURCE_ROOT}/openhands_sdk_training/configs/gne1153_qwen35_4b_20260715/qwen35_4b_v1_seq32768_zero3_fa2_smoke.yaml
CONFIG=${RUN_ROOT}/qwen35_4b_v1_seq32768_zero3_fa2_smoke.yaml
PROBE=${SOURCE_ROOT}/openhands_sdk_training/scripts/gne1153_torch_distributed_probe.py
LOG_DIR=${RUN_ROOT}/logs
GPU_LOG=${LOG_DIR}/gpu_monitor_smoke_${SLURM_JOB_ID}.csv

mkdir -p "$RUN_ROOT" "$LOG_DIR" "${RUN_ROOT}/evidence/root_cause_probes"
cp -a "$CONFIG_SOURCE" "$CONFIG"
cp -a "${SOURCE_ROOT}/openhands_sdk_training/configs/gne1153_qwen35_4b_20260715/ds_z3_config.json" "$RUN_ROOT/ds_z3_config.json"
cp -a "$0" "$RUN_ROOT/$(basename "$0")"
cp -a /home/gneubig/gne1126-nccl-barrier-probe-9308956.{out,err} \
  /home/gneubig/gne1126-nccl-nocumem-9309062.{out,err} \
  /home/gneubig/gne1126-nccl-nop2p-9309063.{out,err} \
  "${RUN_ROOT}/evidence/root_cause_probes/" 2>/dev/null || true

cd "$RUNTIME_ROOT"

export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH
export HF_HOME=/home/gneubig/exp/adp-smoke/hf_cache
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export MASTER_PORT=$((35000 + SLURM_JOB_ID % 20000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/gne1153-triton-${SLURM_JOB_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# babel-q5-20's NCCL 2.26.5 P2P path times out even with explicit device mapping.
# The same four-rank probe succeeds via NCCL shared-memory transport.
export NCCL_P2P_DISABLE=1
mkdir -p "$TRITON_CACHE_DIR"
unset CC CXX

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "runtime_commit=$(git -C /home/gneubig/work/adp/adp-experiments rev-parse HEAD)"
echo "config=${CONFIG}"
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
    sleep 15
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
if [[ "$STATUS" -eq 0 ]]; then
  CHECKPOINT=${RUN_ROOT}/output_smoke_qwen35_4b_seq32768_zero3_fa2/checkpoint-3
  echo "checkpoint_validation_started_at=$(date -Is)"
  CHECKPOINT="$CHECKPOINT" python - <<'PY'
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

checkpoint = Path(os.environ["CHECKPOINT"])
required = [checkpoint / "config.json", checkpoint / "trainer_state.json"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"missing checkpoint metadata: {missing}")
if not list(checkpoint.glob("*.safetensors")) and not (checkpoint / "model.safetensors.index.json").is_file():
    raise RuntimeError("checkpoint has no safetensors weights")
state = json.loads((checkpoint / "trainer_state.json").read_text())
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(
    "checkpoint_load_ok",
    f"path={checkpoint}",
    f"global_step={state.get('global_step')}",
    f"model_type={config.model_type}",
    f"processor={type(processor).__name__}",
    f"parameters={parameter_count}",
)
PY
  echo "checkpoint_validation_finished_at=$(date -Is)"
fi
exit "$STATUS"

#!/bin/bash
#SBATCH --job-name=adp-qwen35-mca-tp4pp2g
#SBATCH --time=2:00:00
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
#SBATCH --output=/project/flame/gneubig/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_v2_adapted/logs/qwen35_35b_a3b_mca_tp4_pp2_ep2_sp_smoke20_gatefix_fa3_12step_%j.out
#SBATCH --error=/project/flame/gneubig/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_v2_adapted/logs/qwen35_35b_a3b_mca_tp4_pp2_ep2_sp_smoke20_gatefix_fa3_12step_%j.out

set -euo pipefail

CODE_DIR=/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/adp-experiments/openhands_sdk_training
RUN_DIR=/project/flame/gneubig/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_v2_adapted
LOG_DIR="${RUN_DIR}/logs"

export CONFIG_YAML="${CODE_DIR}/configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp4_pp2_ep2_sp_smoke20_gatefix_fa3_12step.yaml"
export OUTPUT_DIR="${RUN_DIR}/output_bench_qwen35_35b_a3b_mca_tp4_pp2_ep2_sp_seq32768_smoke20_gatefix_fa3_12step"
export LOG_DIR
export GPU_LOG="${LOG_DIR}/gpu_monitor_qwen35_35b_a3b_mca_tp4_pp2_ep2_sp_smoke20_gatefix_fa3_12step_${SLURM_JOB_ID}.log"
export WANDB_NAME=adp-bench-qwen35-35b-a3b-mca-tp4-pp2-ep2-sp-seq32768-smoke20-gatefix-fa3-12step

exec bash "${CODE_DIR}/scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke7_fa3_50step.sbatch"

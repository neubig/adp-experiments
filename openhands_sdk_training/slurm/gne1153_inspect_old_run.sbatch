#!/usr/bin/env bash
#SBATCH --job-name=gne1153-inspect-old
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/gneubig/gne1153-inspect-old-%j.out
#SBATCH --error=/home/gneubig/gne1153-inspect-old-%j.err

set -u

OLD_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_v1_20260705_restartable_adapted
OLD_OUTPUT=${OLD_ROOT}/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn
OLD_REPO=/home/gneubig/work/adp/adp-experiments
NEW_ROOT=/home/gneubig/exp/adp/runs/openhands_sdk_training/gne1153_qwen35_4b_20260715_194356
EVIDENCE=${NEW_ROOT}/evidence/old_job_9285833

mkdir -p "$EVIDENCE"

snapshot() {
  date -Is
  hostname
  echo "old_root=$OLD_ROOT"
  echo "old_output=$OLD_OUTPUT"
  echo "old_repo=$OLD_REPO"
  echo "new_root=$NEW_ROOT"

  echo "== accounting =="
  sacct -j 9285833 -X --starttime 2026-07-14 \
    --format=JobID,JobName%45,User,Account,Partition,State,Elapsed,Start,End,ExitCode,AllocTRES%80,ReqTRES%80,NodeList%20 -P
  sacct -j 9285833 --starttime 2026-07-14 --json

  echo "== old root listing =="
  ls -la "$OLD_ROOT"
  find "$OLD_ROOT" -maxdepth 3 -type f \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort | tail -200

  echo "== stdout tail =="
  tail -300 "$OLD_ROOT/adp-qwen35-4b-v1-32k-fa2-l40s-9285833.out"
  echo "== stderr tail =="
  tail -300 "$OLD_ROOT/adp-qwen35-4b-v1-32k-fa2-l40s-9285833.err"

  echo "== progress/error scan =="
  grep -Ein "loss|step|checkpoint|main_process_first|barrier|NCCL|CUDA|OOM|nan|Traceback|Error" \
    "$OLD_ROOT/adp-qwen35-4b-v1-32k-fa2-l40s-9285833.out" \
    "$OLD_ROOT/adp-qwen35-4b-v1-32k-fa2-l40s-9285833.err" | tail -500

  echo "== gpu monitor tails =="
  find "$OLD_ROOT" -type f \( -iname '*gpu*9285833*' -o -iname '*9285833*gpu*' \) -print0 |
    xargs -0 -r -n1 sh -c 'echo "FILE $1"; tail -120 "$1"' sh

  echo "== output and checkpoints =="
  ls -la "$OLD_OUTPUT"
  find "$OLD_OUTPUT" -maxdepth 2 -printf '%y %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort | tail -300

  echo "== config and launcher candidates =="
  find "$OLD_ROOT" /home/gneubig/exp/adp/runs/openhands_sdk_training/slurm \
    -maxdepth 2 -type f \( -name '*.yaml' -o -name '*.json' -o -name '*.sbatch' \) \
    -printf '%p\n' 2>/dev/null | sort

  echo "== old runtime repository =="
  git -C "$OLD_REPO" status --short --branch
  git -C "$OLD_REPO" rev-parse HEAD
  git -C "$OLD_REPO" log -1 --decorate --format=fuller

  echo "== environment versions =="
  export PATH=/home/gneubig/exp/adp-smoke/.venv/bin:$PATH
  command -v python
  command -v llamafactory-cli
  python - <<'PY'
import importlib
for name in ("torch", "transformers", "datasets", "accelerate", "deepspeed", "flash_attn", "liger_kernel", "llamafactory"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", "unknown"))
    except Exception as exc:
        print(name, "IMPORT_ERROR", repr(exc))
PY

  echo "== dataset metadata and reusable tokenized data =="
  find "$OLD_ROOT" -maxdepth 2 \( -name dataset_info.json -o -name 'dataset_dict.json' -o -name 'state.json' -o -name '*.arrow' \) \
    -printf '%y %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -300
  du -sh "$OLD_ROOT"/* 2>/dev/null | sort -h
}

snapshot 2>&1 | tee "$EVIDENCE/inspection.txt"

cp -a "$OLD_ROOT"/*.yaml "$OLD_ROOT"/*.json "$EVIDENCE"/ 2>/dev/null || true
cp -a /home/gneubig/exp/adp/runs/openhands_sdk_training/slurm/run_qwen35_4b_v1_restartable_seq32768_flashattn_l40s.sbatch "$EVIDENCE"/ 2>/dev/null || true
cp -a "$OLD_ROOT"/adp-qwen35-4b-v1-32k-fa2-l40s-9285833.out "$EVIDENCE"/ 2>/dev/null || true
cp -a "$OLD_ROOT"/adp-qwen35-4b-v1-32k-fa2-l40s-9285833.err "$EVIDENCE"/ 2>/dev/null || true

echo "evidence_dir=$EVIDENCE"
echo "finished_at=$(date -Is)"

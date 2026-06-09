#!/usr/bin/env bash
set -euo pipefail

# Run from the OpenHands benchmarks checkout used for the experiment.
# Original checkout:
# /home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/benchmarks

export EXP_ROOT=/home/gneubig/exp/adp/evals/2000/swe-bench-smoke
export BENCHMARKS=/home/gneubig/workspace/project/b0ec6769629643e9b4eb723ca0e440cf/benchmarks
export VENV=/home/gneubig/work/openhands-benchmarks-venv
export CHECKPOINT=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn/checkpoint-2000
export VLLM_SANDBOX=/home/gneubig/work/vllm/vllm-openai-v0.22.1-cu129.sandbox

mkdir -p "$EXP_ROOT/logs"

# Serve checkpoint-2000 with vLLM through Apptainer on two L40S GPUs.
CUDA_VISIBLE_DEVICES=0,1 \
APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1 \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
apptainer exec --nv "$VLLM_SANDBOX" \
  /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8012 \
  --model "$CHECKPOINT" \
  --served-model-name local-qwen35-4b \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.70 \
  --language-model-only \
  --skip-mm-profiling \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  > "$EXP_ROOT/logs/vllm_server_32768_native_tools_tp2_continuation.log" 2>&1 &

curl -f http://127.0.0.1:8012/health

# Initial five-instance smoke run. This selected the first five instances from
# princeton-nlp/SWE-bench_Verified test split.
cd "$BENCHMARKS"
IMAGE_TAG_PREFIX=main \
APPTAINER_CACHEDIR=/home/gneubig/work/apptainer-cache \
PYTHONPATH=/openhands-compat \
OPENHANDS_APPTAINER_FORWARD_ENV=PYTHONPATH \
OPENHANDS_APPTAINER_EXTRA_BIND="$BENCHMARKS/benchmarks/utils/apptainer_compat:/openhands-compat:ro" \
BENCHMARKS_DISABLE_PUBLIC_SKILLS=1 \
"$VENV/bin/swebench-infer" \
  "$EXP_ROOT/llm_config.local-vllm-native-8012-2048.json" \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --workspace apptainer \
  --max-iterations 12 \
  --num-workers 1 \
  --n-limit 5 \
  --n-critic-runs 1 \
  --max-retries 0 \
  --tool-preset default \
  --enable-condenser \
  --prompt-path smoke_native_repo_root.j2 \
  --output-dir "$EXP_ROOT" \
  --note smoke5_native_tools_32k_tp2_repo_root_2048_max12 \
  2>&1 | tee "$EXP_ROOT/logs/swebench_infer_smoke5_native_tools_32k_tp2_repo_root_2048_max12.log"

SMOKE5_RUN="$EXP_ROOT/princeton-nlp__SWE-bench_Verified-test/openai/local-qwen35-4b_sdk_c950fdb_maxiter_12_N_smoke5_native_tools_32k_tp2_repo_root_2048_max12"

"$VENV/bin/swebench-eval" \
  "$SMOKE5_RUN/output.jsonl" \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --run-id smoke5_native_tools_32k_tp2_repo_root_2048_max12 \
  --skip-evaluation \
  2>&1 | tee "$EXP_ROOT/logs/swebench_eval_convert_smoke5_native_tools_32k_tp2_repo_root_2048_max12.log"

"$VENV/bin/python" "$EXP_ROOT/apptainer_score_smoke.py" \
  --predictions "$SMOKE5_RUN/output.swebench.jsonl" \
  --score-dir "$EXP_ROOT/apptainer_official_score_32k_tp2" \
  2>&1 | tee "$EXP_ROOT/logs/apptainer_score_smoke_32k_tp2.log"

# Attempted continuation over the next 15 dataset-order instances. These were
# Astropy instances and failed before inference because their OpenHands
# source-minimal image tags were missing in GHCR.
IMAGE_TAG_PREFIX=main \
APPTAINER_CACHEDIR=/home/gneubig/work/apptainer-cache \
PYTHONPATH=/openhands-compat \
OPENHANDS_APPTAINER_FORWARD_ENV=PYTHONPATH \
OPENHANDS_APPTAINER_EXTRA_BIND="$BENCHMARKS/benchmarks/utils/apptainer_compat:/openhands-compat:ro" \
BENCHMARKS_DISABLE_PUBLIC_SKILLS=1 \
"$VENV/bin/swebench-infer" \
  "$EXP_ROOT/llm_config.local-vllm-native-8012-2048.json" \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --workspace apptainer \
  --max-iterations 12 \
  --num-workers 1 \
  --select "$EXP_ROOT/selected_next15_after_smoke5.txt" \
  --n-critic-runs 1 \
  --max-retries 0 \
  --tool-preset default \
  --enable-condenser \
  --prompt-path smoke_native_repo_root.j2 \
  --output-dir "$EXP_ROOT" \
  --note next15_native_tools_32k_tp2_repo_root_2048_max12 \
  2>&1 | tee "$EXP_ROOT/logs/swebench_infer_next15_native_tools_32k_tp2_repo_root_2048_max12.log" || true

# Continuation over five later instances with available OpenHands images.
IMAGE_TAG_PREFIX=main \
APPTAINER_CACHEDIR=/home/gneubig/work/apptainer-cache \
PYTHONPATH=/openhands-compat \
OPENHANDS_APPTAINER_FORWARD_ENV=PYTHONPATH \
OPENHANDS_APPTAINER_EXTRA_BIND="$BENCHMARKS/benchmarks/utils/apptainer_compat:/openhands-compat:ro" \
BENCHMARKS_DISABLE_PUBLIC_SKILLS=1 \
"$VENV/bin/swebench-infer" \
  "$EXP_ROOT/llm_config.local-vllm-native-8012-2048.json" \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --workspace apptainer \
  --max-iterations 12 \
  --num-workers 1 \
  --select "$EXP_ROOT/selected_available_next5_after_smoke5.txt" \
  --n-critic-runs 1 \
  --max-retries 0 \
  --tool-preset default \
  --enable-condenser \
  --prompt-path smoke_native_repo_root.j2 \
  --output-dir "$EXP_ROOT" \
  --note available_next5_native_tools_32k_tp2_repo_root_2048_max12 \
  2>&1 | tee "$EXP_ROOT/logs/swebench_infer_available_next5_native_tools_32k_tp2_repo_root_2048_max12.log"

NEXT5_RUN="$EXP_ROOT/princeton-nlp__SWE-bench_Verified-test/openai/local-qwen35-4b_sdk_c950fdb_maxiter_12_N_available_next5_native_tools_32k_tp2_repo_root_2048_max12"

"$VENV/bin/swebench-eval" \
  "$NEXT5_RUN/output.jsonl" \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --run-id available_next5_native_tools_32k_tp2_repo_root_2048_max12 \
  --skip-evaluation \
  2>&1 | tee "$EXP_ROOT/logs/swebench_eval_convert_available_next5_native_tools_32k_tp2_repo_root_2048_max12.log"

"$VENV/bin/python" "$EXP_ROOT/apptainer_score_smoke.py" \
  --predictions "$NEXT5_RUN/output.swebench.jsonl" \
  --score-dir "$EXP_ROOT/apptainer_official_score_available_next5_32k_tp2" \
  2>&1 | tee "$EXP_ROOT/logs/apptainer_score_available_next5_32k_tp2.log"


# Full SWE-bench 4B Evaluation Runbook

This documents the June 2026 full SWE-bench Verified evaluations for the ADP
4B OpenHands experiments. The target comparison is the base Qwen3.5 4B model
against the checkpoint-2000 fine-tuned model.

## Models

Base model:

```text
Qwen/Qwen3.5-4B-Base
```

Fine-tuned checkpoint:

```text
/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn/checkpoint-2000
```

The base run uses the fine-tuned checkpoint tokenizer for both SDK token
counting and vLLM serving so that the Hugging Face chat template and special
tokens match the fine-tuning/evaluation format.

## Code changes used

The full runs rely on draft PRs rather than untracked local edits:

- OpenHands/benchmarks#745: Apptainer agent-server image builds, SIF reuse,
  tokenizer/condenser CLI wiring, lower `uv` build concurrency, and
  per-instance writable `/workspace` binds for Apptainer inference.
- OpenHands/benchmarks#743: Apptainer SWE-bench scoring and support for the
  Epoch GHCR SWE-bench image mirror.
- OpenHands/benchmarks#751: generic failed-run patch capture for SWE-bench.
  This preserves `test_result.git_patch` for failed, timed-out, or stuck rows
  and uses the staged index diff so generated patches can still be scored. It
  targets `main` independently of the Apptainer image-build PR.
- OpenHands/software-agent-sdk#3641: Apptainer tokenizer binds and
  chat-template token counting for condenser thresholds.

## Token and decoding settings

The model was trained with a 32768 total-token length and 28000 max input
tokens, so the full runs use the same boundary:

```text
MAX_MODEL_LEN=32768
MAX_INPUT_TOKENS=28000
CONDENSER_MAX_TOKENS=28000
MAX_OUTPUT_TOKENS=2047
CONDENSER_MAX_OUTPUT_TOKENS=1024
CONDENSER_MAX_SIZE=240
CONDENSER_KEEP_FIRST=2
```

The condenser threshold is 28000, not 32000, because the model still needs room
for generated output and tool-call framing after condensation. This also matches
the SDK-side max input token guard.

Thinking is disabled for both runs:

```json
{
  "litellm_extra_body": {
    "stop_token_ids": [248046],
    "chat_template_kwargs": {"enable_thinking": false}
  },
  "reasoning_effort": "none"
}
```

Both runs use native tool calling with vLLM's `qwen3_coder` tool parser,
temperature 0.0, and a conversation timeout of 7200 seconds per instance.

## Apptainer image strategy

The full run uses the Epoch SWE-bench image mirror:

```text
OPENHANDS_SWEBENCH_IMAGE_TEMPLATE=ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest
```

The agent-server SIF build cache is shared across prebuild, inference, and
scoring:

```text
OPENHANDS_APPTAINER_BUILD_ROOT=/data/user_data/gneubig/openhands/oh-apptainer-swebench-epoch-sdk43376f1
APPTAINER_CACHEDIR=/data/user_data/gneubig/openhands/apptainer-cache-swebench-epoch
```

The full prebuild job built 500/500 SWE-bench Verified images successfully.
Each prebuild array task used about 59 GB peak RSS, so the successful run used
64 GB memory jobs and serialized `uv` builds:

```text
OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS=4
OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS=1
OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS=1
```

Public skills are enabled (`OPENHANDS_DISABLE_PUBLIC_SKILLS=0`) because public
skills are part of the standard OpenHands harness.

Apptainer inference binds a per-instance host directory onto `/workspace`.
This avoids intermittent `Permission denied` failures when repo setup tries to
copy `/testbed` into `/workspace/<repo>` under Apptainer fakeroot/compat mode:

```text
OPENHANDS_APPTAINER_WORKSPACE_ROOT=/scratch/${USER}/openhands-apptainer-workspaces-${SLURM_JOB_ID}
```

## Slurm jobs

Prebuild:

```text
8325026: adp-swe-prebuild, array 0-7, completed, 500/500 images built
```

Inference:

```text
8331642: adp-swe-base4b, running on babel-n9-32 with 2x L40S
8331643: adp-swe-ft4b, running on babel-p9-24 with 2x L40S
```

Scoring:

```text
8331644: adp-swe-score for base run, dependency afterany:8331642
8331645: adp-swe-score for fine-tuned run, dependency afterany:8331643
8331751: adp-swe-score partial base-run scoring smoke, completed
```

The live inference jobs request the `general` partition, two GPUs constrained
to `L40S|A6000`, `NUM_WORKERS=4`, `TENSOR_PARALLEL_SIZE=2`, 32 CPUs, 256 GB
memory, and a 2 day walltime. The first allocation after this change used
2x A6000 and a four-instance smoke reached `run() triggered successfully` for
all four workers without `/workspace` permission errors.

## Run directories

Base output:

```text
/home/gneubig/exp/adp/evals/full-swebench/q35_base_swe_epoch_tp1_cond28k_thinkoff_errpatch_r1
/home/gneubig/exp/adp/evals/full-swebench/q35_base_swe_epoch_tp2_cond28k_thinkoff_cachedpatch_bind_2gpu4w_r4
```

Fine-tuned output:

```text
/home/gneubig/exp/adp/evals/full-swebench/q35_ft_ckpt2000_swe_epoch_tp1_cond28k_in28k_thinkoff_errpatch_r1
/home/gneubig/exp/adp/evals/full-swebench/q35_ft_ckpt2000_swe_epoch_tp2_cond28k_in28k_thinkoff_cachedpatch_bind_2gpu4w_r4
```

The scorer writes per-run outputs under:

```text
<run_dir>/apptainer_patch_eval/patch_candidates.jsonl
<run_dir>/apptainer_patch_eval/patch_eval_results.jsonl
<run_dir>/apptainer_patch_eval/patch_eval_summary.json
```

The scorer script checked into this experiment directory accepts either the
top-level vLLM run directory or the nested benchmark output directory. If
`output.jsonl` and `output_errors.jsonl` are not present directly under
`RUN_DIR`, it searches recursively for them.

Scoring includes both `output.jsonl` and `output_errors.jsonl`, so error rows
with captured patches are evaluated instead of being silently dropped.

## Submission commands

The prebuild was launched before inference:

```bash
sbatch --array=0-7%8 /home/gneubig/exp/adp/evals/slurm/swebench_prebuild_apptainer_epoch.sbatch
```

The current base and fine-tuned jobs were submitted after the patch-capture,
SIF-image, and Apptainer workspace-bind fixes. The original local integration
branch was later split into independent review PRs: #745 now owns the
Apptainer-specific runtime/image changes, while #751 owns the generic
failed-run patch capture.

```bash
sbatch --export=ALL,RUN_NAME=q35_base_swe_epoch_tp2_cond28k_thinkoff_cachedpatch_bind_2gpu4w_r4,NOTE=q35_base_swe_epoch_tp2_cond28k_thinkoff_cachedpatch_bind_2gpu4w_r4,N_LIMIT=0,NUM_WORKERS=4,TENSOR_PARALLEL_SIZE=2 \
  /home/gneubig/exp/adp/evals/slurm/swebench_full_qwen35_4b_base.sbatch

sbatch --export=ALL,RUN_NAME=q35_ft_ckpt2000_swe_epoch_tp2_cond28k_in28k_thinkoff_cachedpatch_bind_2gpu4w_r4,NOTE=q35_ft_ckpt2000_swe_epoch_tp2_cond28k_in28k_thinkoff_cachedpatch_bind_2gpu4w_r4,N_LIMIT=0,NUM_WORKERS=4,TENSOR_PARALLEL_SIZE=2 \
  /home/gneubig/exp/adp/evals/slurm/swebench_full_qwen35_4b_ft_ckpt2000.sbatch
```

Scoring was queued with `afterany` dependencies so it still runs if inference
exits nonzero after writing partial outputs:

```bash
sbatch --dependency=afterany:8331642 --export=ALL,RUN_DIR=/home/gneubig/exp/adp/evals/full-swebench/q35_base_swe_epoch_tp2_cond28k_thinkoff_cachedpatch_bind_2gpu4w_r4 \
  /home/gneubig/exp/adp/evals/slurm/swebench_score_patches_apptainer.sbatch

sbatch --dependency=afterany:8331643 --export=ALL,RUN_DIR=/home/gneubig/exp/adp/evals/full-swebench/q35_ft_ckpt2000_swe_epoch_tp2_cond28k_in28k_thinkoff_cachedpatch_bind_2gpu4w_r4 \
  /home/gneubig/exp/adp/evals/slurm/swebench_score_patches_apptainer.sbatch
```

## Monitoring commands

```bash
squeue -j 8331642,8331643,8331644,8331645 \
  -o '%.18i %.24j %.9P %.8T %.10M %.10L %.20R'

sacct -j 8331642,8331643,8331644,8331645 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList -P

tail -100 /home/gneubig/exp/adp/evals/full-swebench/slurm-adp-swe-base4b-8331642.out
tail -100 /home/gneubig/exp/adp/evals/full-swebench/slurm-adp-swe-ft4b-8331643.out
```

After scoring completes, summarize:

```bash
cat /home/gneubig/exp/adp/evals/full-swebench/q35_base_swe_epoch_tp2_cond28k_thinkoff_cachedpatch_bind_2gpu4w_r4/apptainer_patch_eval/patch_eval_summary.json
cat /home/gneubig/exp/adp/evals/full-swebench/q35_ft_ckpt2000_swe_epoch_tp2_cond28k_in28k_thinkoff_cachedpatch_bind_2gpu4w_r4/apptainer_patch_eval/patch_eval_summary.json
```

## Status

At the latest monitoring point, the Apptainer prebuild was healthy and the
current full base and fine-tuned inference jobs were running on the `general`
GPU partition with two L40S GPUs each. The logs confirmed `NUM_WORKERS=4`,
`TENSOR_PARALLEL_SIZE=2`, `CUDA_VISIBLE_DEVICES=0,1`, and vLLM serving multiple
concurrent requests.

The current runs no longer show the earlier Apptainer `/workspace` permission
failure or missing-image build failures. Early inference results are still poor:
completed rows fail with `Remote conversation got stuck`, but failed-run patch
capture is active for every completed row. At 08:04 EDT on 2026-06-12, the base
run had 12 completed rows, 2 non-empty captured patches, and 12 rows marked
`git_patch_captured_on_error`; the fine-tuned run had 7 completed rows, 0
non-empty captured patches, and 7 rows marked `git_patch_captured_on_error`.
Inspection of a zero-patch fine-tuned trajectory showed repeated
inspect/search/view actions and stuck-detector termination before any repo edit,
so those empty diffs look like model behavior rather than another
patch-capture failure.

A partial base-run scoring smoke was launched against the top-level base run
directory to validate the scorer before full inference finishes:

```text
8331751: completed in 1m34s, 3 patch candidates, 3 patches applied,
2 resolved, 0 score errors
```

This smoke verified that recursive source discovery finds the nested
`output.jsonl` and `output_errors.jsonl` files and that both completed-row
patches and failed-run captured patches are evaluated by the Apptainer scorer.

Earlier `errpatch_r1` and `cachedpatch_r2/r3` attempts were cancelled after
finding the failed-run patch capture bug and the Apptainer `/workspace`
permission issue. Final patch counts and resolved/unresolved SWE-bench results
should be added here after jobs `8331642`, `8331643`, `8331644`, and `8331645`
finish.

# Codescout RL Smoke Test

This directory logs the first small Codescout RL smoke test run through Platoon
PR 14 and AReaL. The goal was to verify that the code path works end to end on
a small model, not to measure task-solving performance.

## Run Summary

- Platoon source: `https://github.com/ApGa/platoon/pull/14`
- Date run: 2026-06-05
- Local checkout: `~/work/platoon`, branch `pr-14`, commit `3d4cf0a`
- Experiment workspace: `~/exp/codescout`
- Smoke script: `~/exp/codescout/train_codescout_smoke.py`
- Config: `~/exp/codescout/train_codescout_smoke.yaml`
- Slurm wrapper: `~/exp/codescout/run_codescout_smoke.slurm`
- Model: `Qwen/Qwen3-0.6B`
- Trainer stack: Platoon Codescout plugin, AReaL local launcher, SGLang rollout
  server, OpenHands Apptainer environment
- Dataset loader: `adityasoni17/SWE-smith-py-code-search`
- Smoke data slice: 2 train tasks and 2 validation tasks
- Rollout settings: 2 samples, 6 maximum agent steps, 512 maximum new tokens,
  16k SGLang context
- Hardware request: 1 debug Slurm job with 2 GPUs, 12 CPUs, and 96 GiB memory

The working job was:

```bash
sbatch ~/exp/codescout/run_codescout_smoke.slurm
```

## Result

The successful functional run was Slurm job `8268037`. It launched SGLang,
loaded the Codescout dataset, started the OpenHands Apptainer environment,
generated rollouts, converted event traces into training data, logged AReaL
statistics, and saved a checkpoint.

Checkpoint:

```text
~/exp/codescout/areal/experiments/checkpoints/gneubig/codescout-smoke/qwen3-0_6b-two-task-16k-nofilter/default/epoch0epochstep0globalstep0/
```

The checkpoint contains `model.safetensors`, about 1.5 GiB. Rollout event JSONL
files were written under:

```text
~/exp/codescout/areal/experiments/codescout-smoke-qwen3-0_6b-16k-nofilter/train_rollout/0/events/
```

The trainer log reported `Training completes! Total time elapsed 63.10.` The
observed task rewards were all zero, which is expected for a tiny
`Qwen3-0.6B` smoke run and should not be interpreted as a performance result.

## Follow-Up Notes

Earlier attempts exposed two smoke-test configuration requirements:

- 4k and 8k SGLang context windows were too short once multi-step Codescout
  trajectories accumulated history. The smoke run completed with 16k context.
- A tiny model with a very small sample count produced zero-variance reward
  groups. For smoke testing, `workflow_config.filter_zero_variance_groups` was
  set to `false` so the run can complete and save a checkpoint.

Slurm marked job `8268037` as failed because AReaL's local launcher raised a
post-training `JobException` after the trainer completed:

```text
areal.utils.launcher.JobException:
Job codescout-smoke_qwen3-0_6b-two-task-16k-nofilter:trainer
JobState.COMPLETED at node local
```

This appears to be a launcher teardown/state-handling issue rather than a
training failure, because the checkpoint and rollout artifacts were written.

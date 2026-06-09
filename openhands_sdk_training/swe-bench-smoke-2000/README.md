# SWE-bench Smoke Evaluation for Checkpoint 2000

This directory documents the June 9, 2026 smoke evaluation of the 4B ADP
OpenHands checkpoint on SWE-bench Verified.

## Model

Checkpoint:

```text
/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn/checkpoint-2000
```

The model was served with vLLM through Apptainer on two L40S GPUs using tensor
parallel size 2 and a 32768-token model length. The benchmark used OpenHands
SDK commit `c950fdb08abea040eebd0bb3d5ff63db293b9125`, matching the benchmarks
submodule checked out for this run.

## Results

Two groups were scored with the same model and serving configuration:

- Initial smoke set: `2/5` resolved.
- Additional available-image set: `0/5` resolved.
- Combined scored result: `2/10` resolved.

The next 15 dataset-order instances after the initial smoke set were Astropy
instances whose OpenHands Apptainer image tags were not available in GHCR, so
they failed before inference. The second scored group therefore used the next
five instances for which source-minimal benchmark images were available.

See `run_summary.md` for details, per-instance outcomes, and log paths on the
original evaluation machine.

## Files

- `commands.sh`: command transcript for serving, inference, conversion, and
  scoring.
- `llm_config.local-vllm-native-8012-2048.json`: LLM config passed to
  `swebench-infer`.
- `smoke_native_repo_root.j2`: prompt used for the completed 32k TP2 native-tool
  runs.
- `selected_available_next5_after_smoke5.txt`: additional scored instances.
- `selected_next15_after_smoke5.txt`: attempted Astropy continuation that
  failed due to missing images.
- `score_smoke5_summary.json`: Apptainer scoring summary for the initial five.
- `score_available_next5_summary.json`: Apptainer scoring summary for the
  additional five.
- `apptainer_score_smoke.py`: local Apptainer scorer used because Modal auth and
  Docker were unavailable on the evaluation host.
- `run_summary.md`: full run summary generated during the experiment.


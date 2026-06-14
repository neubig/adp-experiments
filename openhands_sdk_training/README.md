# OpenHands SDK SFT Training With LLaMA-Factory

These scripts reproduce the data extraction, cleaning, and training setup used
for ADP OpenHands-compatible SFT data.

The original reproducible script path below is the small 0.8B ROCm run. The
current 32k Qwen3.5 4B/9B condenser experiments are tracked in
[`CURRENT_EXPERIMENTS.md`](CURRENT_EXPERIMENTS.md).

The original small run documented here used:

- Model: `Qwen/Qwen3.5-0.8B-Base`
- Trainer: LLaMA-Factory SFT
- Training type: full-parameter fine-tuning
- Data mixture: ADP paper-style OpenHands/SWE-Agent non-web mixture
- Sequence length: `2048`
- Per-device batch size: `1`
- Max steps: `10000`
- Eval every: `500` steps
- Eval split size: `290` trajectories
- Hardware observed: AMD ROCm on Radeon 8060S, 32 GiB VRAM allocation from a
  64 GiB unified-memory machine

The local run reached training successfully with 39,956 examples after cleaning
and LLaMA-Factory filtering. It used about 20.2 GiB VRAM and ran at roughly
3.36 seconds per step after ROCm warmup.

## Local Layout

Use the repository-level workspace convention:

- Code and scripts: `~/work/adp/adp-experiments/openhands_sdk_training`
- Downloaded ADP release files: `~/exp/adp/datasets/hf_release`
- Standardized/full source files used by condenser experiments:
  `~/exp/adp/datasets/hf_std`
- Paper-style train/eval splits:
  `~/exp/adp/datasets/paper_openhands_nonweb_v1`
- Training logs and model outputs: `~/exp/adp/runs/openhands_sdk_training`
- Shared Hugging Face cache: `~/exp/adp/cache/hf`

## Script Layout

- `scripts/setup_rocm_uv_env.sh`: create a uv virtual environment and install
  the training dependencies.
- `scripts/download_adp_hf_release.py`: download the full ADP SFT JSONL files
  from Hugging Face.
- `scripts/make_paper_splits.py`: create deterministic paper-style train/eval
  mixtures.
- `scripts/clean_for_llamafactory.py`: remove malformed records and escape
  literal multimodal XML tags that LLaMA-Factory interprets as media markers.
- `scripts/write_llamafactory_config.py`: write `dataset_info.json` entries and
  the Qwen3.5 0.8B training YAML.
- `scripts/run_training.sh`: launch the 10k-step run in the background.
- `scripts/monitor_training.sh`: inspect the running job, logs, and ROCm memory.
- `scripts/analyze_debug_monitors.py`: summarize Slurm debug monitor logs from
  `nvidia-smi dmon`, `sar`, and `pidstat` for GPU, network, and CPU bottleneck
  analysis.

## Babel Slurm Run History

The Babel Slurm jobs used for the condenser and Qwen3.5 training experiments
are archived in this repo:

- `slurm/`: submitted Slurm job scripts copied from
  `~/exp/adp/runs/openhands_sdk_training/slurm`, including the Qwen3.5 9B
  32k full-condenser run.
- `configs/`: LLaMA-Factory YAML configs copied from
  `~/exp/adp/runs/openhands_sdk_training/*/*.yaml`.
- `run_history/slurm_adp_jobs_2026-06-01.tsv`: `sacct` history for ADP Slurm
  jobs since 2026-06-01, including failed/cancelled smoke attempts and the
  active production runs.

The most recent 9B production run is:

```text
Job:    adp-qwen35-9b-32k-cond24k-fa2-a100
Script: slurm/run_qwen35_9b_liger_seq32768_full_condenser_24k_all_records_one_epoch_large80_192g.sbatch
Config: configs/full_condenser_24k_all_records_adapted/qwen35_9b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_a100.yaml
State:  running as Slurm job 8263041 on babel-v5-20
```

## Reproduce

Run from the repository root:

```bash
cd openhands_sdk_training

# Optional but recommended on the machine where training will run.
bash scripts/setup_rocm_uv_env.sh
source .venv/bin/activate

# Authenticate before training if you want W&B logging.
wandb login

# Download the full ADP SFT release. This is large: about 33 GiB for all subsets.
python scripts/download_adp_hf_release.py \
  --repo-id neulab/agent-data-collection \
  --out ~/exp/adp/datasets/hf_release \
  --preset all

# Build the OpenHands/SWE-Agent non-web train/eval split used in this run.
python scripts/make_paper_splits.py \
  --input-root ~/exp/adp/datasets/hf_release \
  --output-dir ~/exp/adp/datasets/paper_openhands_nonweb_v1 \
  --mixture openhands_nonweb

# Validate/clean JSONL and neutralize literal <image>/<video>/<audio> tags.
python scripts/clean_for_llamafactory.py \
  --train ~/exp/adp/datasets/paper_openhands_nonweb_v1/paper_openhands_nonweb_train.jsonl \
  --eval ~/exp/adp/datasets/paper_openhands_nonweb_v1/paper_openhands_nonweb_eval.jsonl

# Write LLaMA-Factory dataset metadata and the training YAML.
python scripts/write_llamafactory_config.py \
  --dataset-dir ~/exp/adp/datasets/paper_openhands_nonweb_v1 \
  --output-dir ~/exp/adp/runs/openhands_sdk_training/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe/output \
  --run-name adp-openhands-nonweb-qwen35-0.8b-full-10k-bs1-seq2048-mm-safe

# Launch in the background.
bash scripts/run_training.sh \
  ~/exp/adp/datasets/paper_openhands_nonweb_v1/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe.yaml \
  ~/exp/adp/runs/openhands_sdk_training/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe/logs/train_10k_mm_safe
```

Monitor the run:

```bash
bash scripts/monitor_training.sh ~/exp/adp/runs/openhands_sdk_training/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe/logs/train_10k_mm_safe
tail -f ~/exp/adp/runs/openhands_sdk_training/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe/logs/train_10k_mm_safe.log
```

For Slurm jobs that write per-node debug monitor directories, summarize the raw
system metrics with:

```bash
python scripts/analyze_debug_monitors.py \
  ~/exp/adp/runs/openhands_sdk_training/<run>/logs/debug_<variant>_<job_id> \
  --trainer-log ~/exp/adp/runs/openhands_sdk_training/<run>/output/trainer_log.jsonl
```

## Data Mixtures

The ADP paper-style weighted mixtures implemented by `make_paper_splits.py` are:

- `openhands_nonweb`: AgentTuning, CodeActInstruct, SWE-Agent/OpenHands, SWE-Gym,
  SWE-Smith, Code Feedback, and Orca AgentInstruct subsets.
- `agentlab_web`: Go-Browse-WA, Mind2Web, NNetNav, and Synatra web subsets.
- `all_weighted_union`: union of the two groups above.

The eval policy used here is a deterministic half-size holdout per source:

```text
previous_eval_count = min(100, max(10, ceil(0.02 * raw_count)))
eval_count = max(1, floor(previous_eval_count / 2))
```

Training rows are sampled from the remaining rows using the ADP appendix
multipliers encoded in the script. Eval rows are never upsampled.

## Notes From The First Run

The initial uncleaned split had two issues:

- 5 malformed rows had null or invalid `conversations` fields and failed
  LLaMA-Factory conversion.
- 14 rows contained literal `<video>`/`<image>`/`<audio>` text. Qwen VL
  processors interpreted these as media placeholders, so the cleaner escapes
  those literal strings before training.

LLaMA-Factory also skipped one abnormal role-pattern example during conversion.
That was non-fatal.

Sequences longer than `cutoff_len: 2048` are truncated by LLaMA-Factory. This
means only the prefix up to the cutoff participates in the loss; labels beyond
that point are dropped for that training example.

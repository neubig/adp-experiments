# Current OpenHands SDK Experiments

This file tracks the current experiments under `~/exp/adp` as of 2026-06-05.
The original 0.8B reproduction flow remains in `README.md`; the current main
work is the 32k Qwen3.5 full-parameter SFT line on OpenHands SDK condenser data.

## Data

Current all-records adapted split:

```text
root: ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted
manifest: ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/manifest.json
train: full_condenser_24k_all_records_train.openai.jsonl
eval: full_condenser_24k_all_records_eval.openai.jsonl
llamafactory train: full_condenser_24k_all_records_train.llamafactory.jsonl
llamafactory eval: full_condenser_24k_all_records_eval.llamafactory.jsonl
tokenized 4B path: tokenized_qwen35_4b_seq32768_all_records
```

### LLaMA-Factory normalization requirement

For OpenHands SDK SFT records, do not point LLaMA-Factory at the canonical
OpenAI JSONL directly and do not hand-normalize fields in an experiment-local
script. Regenerate LLaMA-Factory train/eval files with the ADP adapter:

```bash
python -m agents.openhands_sdk.sft_to_llamafactory \
  --input INPUT.openai.jsonl \
  --output OUTPUT.llamafactory.jsonl \
  --dataset-info dataset_info.json \
  --dataset-name DATASET_NAME \
  --trim-to-trainable \
  --skip-untrainable
```

This adapter converts tool calls into LLaMA-Factory `function_call` messages,
merges adjacent prompt-side messages, trims to trainable prefixes, and
stringifies the top-level `tools` field. The `tools` stringification is required
before LLaMA-Factory invokes Hugging Face `datasets.load_dataset`; otherwise
heterogeneous nested tool schemas can be inferred as incompatible Arrow structs
and fail before LLaMA-Factory's own converter runs.

### Qwen3.5 32k Liger logits OOM

For Qwen3.5 long-context runs with `enable_liger_kernel: true`, training can fit
while evaluation OOMs. This is counterintuitive but expected from the current
Liger Qwen3.5 forward path: training with labels uses Liger's fused causal-LM
loss and skips materializing full `batch x sequence x vocab` logits, while eval
defaults to computing logits before loss because `model.training` is false.
With Qwen3.5-9B at `cutoff_len: 32768` and vocab size 248320, one bf16 logits
tensor is about 15 GiB before cross-entropy workspace, padding, or distributed
gathering.

Use loss-only eval and patch LLaMA-Factory SFT eval to pass Liger's
`skip_logits=True` forward argument:

```bash
source .venv/bin/activate
python scripts/patch_llamafactory_liger_eval_skip_logits.py
```

Set the training YAML eval options to:

```yaml
prediction_loss_only: true
per_device_eval_batch_size: 1
eval_strategy: steps
```

`prediction_loss_only: true` prevents Hugging Face Trainer from retaining and
gathering logits, but it is not sufficient by itself for Liger Qwen3.5 because
the model forward would still compute logits internally. The patch makes the
forward loss-only as well. Do not use this patched path for generated-prediction
metrics; it is intended for scalar eval loss/perplexity during SFT.

For Qwen3.5-MoE models such as `Qwen3.5-35B-A3B`, training can OOM for the same
underlying reason if LLaMA-Factory does not dispatch Liger for
`model_type: qwen3_5_moe`. Recent Liger releases include
`apply_liger_kernel_to_qwen3_5_moe`, whose training forward also avoids full
logit materialization. The same patch script adds this LLaMA-Factory dispatch;
without it, 32k full SFT materializes logits during training and can OOM before
the first step.

### Qwen3.5-35B-A3B multi-node efficiency

On 2x8 H100 nodes, Qwen3.5-35B-A3B full SFT is communication-bound under
ordinary ZeRO-3. The model has roughly 35B trainable parameters, but most are
MoE expert weights. With 256 experts and 8 selected experts per token, the
active-parameter estimate is only about 4B parameters per token, while ZeRO-3
still shards and gathers the near-35B trainable parameter set. Padding/packing
can still waste useful token work, but it is not enough to explain the observed
low FLOP utilization by itself.

The profile run showed `nccl:_all_gather_base` and `nccl:_reduce_scatter_base`
as dominant costs, and NCCL initialized with socket networking rather than
IB/RDMA:

```text
NCCL INFO NET/Plugin: Could not find: libnccl-net.so
NCCL INFO NET/IB : No device found.
NCCL INFO Using network Socket
```

The best reproducible LLaMA-Factory/DeepSpeed fix found so far is hierarchical
ZeRO partitioning with one ZeRO parameter partition group per 8-GPU node:

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": false,
    "zero_hpz_partition_size": 8
  }
}
```

For a 16-rank run with ranks assigned node-locally, `zero_hpz_partition_size: 8`
forms groups `[0..7]` and `[8..15]`, keeping ZeRO parameter all-gathers within
each NVLink-connected node and avoiding the slow socket path for that traffic.
This raised memory to roughly 67-69 GiB/GPU but improved post-warmup step time
from about 75s to about 51s in the 35B-A3B 32k benchmark. Do not combine this
with manual large ZeRO bucket tuning without re-testing; larger bucket variants
OOMed or hit CUDA errors in smoke runs.

### 2026-06-14 FSDP2 and WSD notes

An upstream LLaMA-Factory `main` FSDP2 smoke run was attempted for
`Qwen3.5-35B-A3B` full SFT at the same 32k context length and global batch as
the hpZ8 run:

```text
job: 123812
run: adp-bench-qwen35-35b-a3b-fsdp2-full-seq32768-2node-h100
nodes: orchard-flame-[5,0]
config: qwen35_35b_a3b_bench_fsdp2_full_seq32768_2node_h100.yaml
accelerate: accelerate_fsdp2_qwen35_moe_2node_h100.yaml
```

This is a negative result for now. The job reached training/backward, then OOMed
inside Liger fused MoE backward at microbatch size 1:

```text
liger_kernel/ops/fused_moe.py backward
torch.OutOfMemoryError: Tried to allocate 724 MiB to 1024 MiB
process has roughly 78.7-79.2 GiB in use on 80 GiB H100
```

Other ranks then reported NCCL `_ALLGATHER_BASE` remote-close errors, but those
were a consequence of the OOM ranks exiting. This FSDP2 recipe is therefore not
yet a replacement for hpZ8 at 32k unless memory is reduced elsewhere, for
example by changing the MoE kernel, reducing context length, or adding a
working sequence/context-parallel implementation.

The optional HyperParallel backend is not required for the current stable
DeepSpeed hpZ8 run. If it is tested later, install from the GitCode
`mindspore/hyper-parallel` source rather than PyPI: the advertised
`hyper_parallel` package was not present on PyPI, and the GitHub mirror lagged
the LLaMA-Factory integration API at the time of testing. The patch helper now
treats LLaMA-Factory HyperParallel modules as optional so missing or mismatched
HyperParallel does not block ordinary SFT patching.

### 2026-06-14 MCA / Megatron-Core notes

LLaMA-Factory has a Megatron-Core Adapter path gated by `USE_MCA=1`. Do not use
the PyPI `mcore-adapter==0.0.1` package; it installs no usable
`mcore_adapter` module. The working adapter source is the ROLL subdirectory:

```bash
python -m pip install --force-reinstall --no-deps \
  "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"
python -m pip install --no-deps "megatron-core>=0.13.0,<0.14.0"
```

This produced `mcore_adapter==0.9.0` and `megatron-core==0.13.1` in the
isolated MCA venv. LLaMA-Factory's MCA parser exposes the relevant MoE training
knobs: `expert_model_parallel_size`, `pipeline_model_parallel_size`,
`context_parallel_size`, `sequence_parallel`, `moe_token_dispatcher_type`,
`moe_grouped_gemm`, `moe_shared_expert_overlap`, distributed optimizer, and
gradient/parameter overlap.

Transformer Engine did not install cleanly in the current Torch 2.12 / CUDA 13
venv. `transformer-engine[pytorch]==2.16.0` downloaded the CUDA 13 support
wheel but had no prebuilt `transformer_engine_torch` wheel for the exact
`torch2.12.0+cu130` ABI, then failed source compilation because `cudnn.h` was
not available on the build host. As a result, the first MCA smoke uses
`transformer_impl: local`; Megatron warns that it is falling back to Torch Norm
and Torch optimizer helpers. A fully optimized Megatron recipe likely needs a
matching NGC-style container or CUDA/cuDNN headers plus a compatible TE wheel.

The first two-node MCA smoke is:

```text
job: 123821
run: adp-bench-qwen35-35b-a3b-mca-pp4-ep4-seq32768-smoke2
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_pp4_ep4_smoke.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_pp4_ep4_smoke.sbatch
parallelism: TP=1, PP=4, EP=4, CP=1, GAS=4
```

The raw copied venv console scripts still point at the original venv. Use
`scripts/mca_bin/torchrun` with `MCA_PYTHON=/path/to/.venv_mca/bin/python` so
LLaMA-Factory's MCA launcher calls the MCA interpreter via
`python -m torch.distributed.run`.

Two early MCA smoke attempts failed before training:

```text
job 123820: MCA parser rejected overwrite_output_dir; removed that key from the smoke config.
job 123821: mcore_adapter converted Qwen3.5-MoE HF config keys into MCA keys,
           but Qwen3_5Config did not declare Qwen3.5 linear-attention fields,
           causing TypeError on linear_conv_kernel_dim.
```

The ADP patch helper now also patches
`mcore_adapter.models.qwen3_5.config_qwen3_5` to declare the Qwen3.5
linear-attention and Qwen3.5-MoE template fields that the adapter itself maps:
`linear_conv_kernel_dim`, `linear_key_head_dim`, `linear_value_head_dim`,
`linear_num_key_heads`, `linear_num_value_heads`, `linear_attention_freq`,
`attention_output_gate`, `experimental_attention_variant`, and
`moe_shared_expert_gate`. After this patch, a local
`AutoConfig.from_pretrained("/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B")`
probe succeeded and reported `num_moe_experts=256`, `moe_router_topk=8`, and
`experimental_attention_variant=gated_delta_net`.

The next MCA attempts surfaced two more compatibility constraints:

```text
job 123822: got past Qwen3.5 config conversion but failed because
           apply_rope_fusion requires Transformer Engine >= 1.4 or Apex.
           The smoke config now sets apply_rope_fusion: false.
job 123823: got past rope fusion but failed because megatron-core==0.13.1
           does not include
           megatron.core.models.gpt.experimental_attention_variant_module_specs,
           which ROLL's Qwen3.5 adapter imports for gated-delta attention.
```

Upgrading only the isolated MCA venv to `megatron-core==0.16.1` fixed the
missing experimental-attention module in a local import/config probe:

```text
AutoConfig.from_pretrained(...Qwen3.5-35B-A3B) -> Qwen3_5Config
transformer_impl: transformer_engine
experimental_attention_variant: gated_delta_net
```

The current patched MCA smoke is queued as:

```text
job: 123824
run: adp-bench-qwen35-35b-a3b-mca-pp4-ep4-seq32768-smoke5
parallelism: TP=1, PP=4, EP=4, CP=1, GAS=4
config changes since smoke3: megatron-core==0.16.1, transformer_impl=transformer_engine,
                            apply_rope_fusion=false
```

This path requires a matching Transformer Engine runtime. Smoke `123824` failed
before model construction with:

```text
NameError: name 'TESpecProvider' is not defined
```

This comes from Megatron's experimental gated-delta attention spec. The module
is present in `megatron-core==0.16.1`, but it only defines `TESpecProvider` when
Transformer Engine imports successfully. ROLL's Qwen3.5 model also asserts
`transformer_impl == "transformer_engine"` when the gated-delta attention
variant is present, so MCA is blocked in this venv until a compatible
Transformer Engine stack is available.

For the next full run, use the best stable hpZ8 DeepSpeed recipe and switch only
the scheduler to WSD:

```yaml
deepspeed: ds_z3_config_qwen35_hpz8.json
use_v1_kernels: cuda_fused_moe
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
lr_scheduler_type: warmup_stable_decay
warmup_ratio: 0.03
eval_steps: 100
save_only_model: false
```

The prepared no-gradient-checkpointing fallback keeps the same hpZ8 and
`cuda_fused_moe` settings, changing only `gradient_checkpointing: false` for a
12-step smoke. This isolates activation recompute overhead from the fused-MoE
kernel choice. The first queued job for this fallback is:

```text
job: 123825
run: adp-bench-qwen35-35b-a3b-hpz8-cuda-fused-moe-no-gc-seq32768-smoke
```

Job `123825` failed before launch because the patch helper tried to import the
optional MCA workflow in the base venv, where `mcore_adapter` is intentionally
not installed. The helper now skips optional MCA patches on `ImportError`; the
same smoke was resubmitted as job `123826`.

Both installed LLaMA-Factory `0.9.5` and the upstream `main` overlay include the
WSD scheduler hook. Without explicit `lr_scheduler_kwargs`, the local
LLaMA-Factory helper defaults to one third of post-warmup steps as stable LR and
the remaining two thirds as decay.

Manifest counts:

```text
total_records: 170958
canonical_train_records: 170458
canonical_eval_records: 500
unique_source_trajectories: 107322
selection: all currently available OpenHands SDK condenser SFT records; eval split is 500 records with lowest sha1(record_id)
```

## Current 32k Run Family

Common settings:

```text
trainer: LLaMA-Factory SFT
finetuning_type: full
template: qwen3_5_nothink
cutoff_len: 32768
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
max_steps: 5322
save_steps: 500
save_total_limit: 3
eval_strategy: steps
eval_steps: 100
bf16: true
gradient_checkpointing: true
save_only_model: true
```

Configs in `~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted`:

| Config | Model | Variant | Output |
| --- | --- | --- | --- |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-3 baseline | `output_qwen35_4b_seq32768_zero3_one_epoch` |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-3 + FlashAttention-2 | `output_qwen35_4b_seq32768_zero3_one_epoch_flashattn` |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_muon.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-3 + Muon | `output_qwen35_4b_seq32768_zero3_one_epoch_muon` |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_z2_one_epoch_muon.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-2 + Muon | `output_qwen35_4b_seq32768_z2_one_epoch_muon` |
| `qwen35_9b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_a100.yaml` | `Qwen/Qwen3.5-9B` | ZeRO-3 + FlashAttention-2 for A100 | `output_qwen35_9b_seq32768_zero3_one_epoch_a100_flashattn` |

Run a config directly:

```bash
cd ~/work/adp/adp-experiments/openhands_sdk_training

llamafactory-cli train \
  ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml
```

For Slurm runs, capture the job metadata, `nvidia-smi`, flash-attn check, and
GPU monitor path in the `.out` file before launching `llamafactory-cli train`.
The current successful flash-attn run did this and used one 4x L40S node.

## FlashAttention-2 Setup

The working flash-attn run used:

```text
flash-attn: 2.8.3
torch CUDA arch list includes sm_80
is_flash_attn_2_available: True
hardware: 4x NVIDIA L40S, 46068 MiB each
```

Verify before launching:

```bash
python - <<'PY'
import flash_attn
from transformers.utils import is_flash_attn_2_available

print("flash_attn", flash_attn.__version__)
print("is_flash_attn_2_available", is_flash_attn_2_available())
PY
```

The LLaMA-Factory log should include:

```text
Using FlashAttention-2 for faster training and inference.
Fine-tuning method: Full
```

## Status Snapshot

Observed status on 2026-06-05:

```text
active Slurm job: 8254382
run: adp-full-condenser-24k-all-records-qwen35-4b-seq32768-zero3-one-epoch-flashattn
config: qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml
latest log step: 1697 / 5322
latest saved checkpoint: checkpoint-1500
checkpoint-1500 eval_loss: 0.2295125424861908
```

Find the latest checkpoint dynamically instead of hard-coding a step:

```bash
find ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn \
  -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' |
  sed 's/checkpoint-//' |
  sort -n |
  tail -1
```

List current checkpoints with timestamps:

```bash
find ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn \
  -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' |
  sort -nr
```

## Loading A Checkpoint

The saved 4B checkpoints are full-model Hugging Face checkpoints, not LoRA
adapters and not ZeRO shards. They contain `model.safetensors`, `config.json`,
tokenizer files, `processor_config.json`, and `generation_config.json`.

Example:

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ckpt = (
    "/home/gneubig/exp/adp/runs/openhands_sdk_training/"
    "full_condenser_24k_all_records_adapted/"
    "output_qwen35_4b_seq32768_zero3_one_epoch_flashattn/"
    "checkpoint-1500"
)

processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

Compatibility note: these checkpoints have `model_type: qwen3_5` and were saved
with a Transformers build that reports `transformers_version: 5.6.0`. Use a
Transformers version or local branch that knows the Qwen3.5 classes.

## Hugging Face Uploads

The private Hub repo `gneubig/adp-qwen35-4b-flashattn-ckpt1000` contains the
older `checkpoint-1000`. The current local latest checkpoint is newer
(`checkpoint-1500` as of 2026-06-05), so upload scripts should discover the
latest checkpoint before publishing.

For checkpoints larger than 5 GiB, enable large-file support and use Git LFS.
The Python `huggingface_hub.upload_folder` path can exceed memory limits on
large `model.safetensors` files.

```bash
hf lfs-enable-largefiles /path/to/local/repo
git lfs track '*.safetensors' 'tokenizer.json' '*.bin'
git add .gitattributes .
git commit -m "Upload flash-attn checkpoint"
git push origin main
```

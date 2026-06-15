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

Follow-up: Transformer Engine can be built in the isolated MCA venv if the pip
NVIDIA headers are exposed during compilation:

```bash
VENV=.venv_mca
SITE="$VENV/lib/python3.11/site-packages"
export CUDNN_INCLUDE_DIR="$SITE/nvidia/cudnn/include"
export CUDNN_LIB_DIR="$SITE/nvidia/cudnn/lib"
export CPLUS_INCLUDE_PATH="$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
python -m pip install -v --no-build-isolation --no-deps \
  transformer-engine==2.16.0 \
  transformer-engine-cu13==2.16.0 \
  transformer-engine-torch==2.16.0
```

Runtime import additionally requires using the venv CUDA 13 libraries before
the system CUDA libraries. `transformer-engine==2.16.0` referenced a
`libcublasLt.so.13` symbol that was not present in the default cuBLAS library,
so the MCA venv was upgraded to `nvidia-cublas==13.5.1.27` with `--no-deps`
after installing TE's Python-only dependencies (`onnxscript` and
`nvdlfw-inspect`). The MCA launcher now exports:

```bash
SITE="$VENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cu13/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
```

With that path, a local import probe succeeded for `transformer_engine.pytorch`
and `mcore_adapter.models.AutoConfig` reported
`experimental_attention_variant=gated_delta_net` for Qwen3.5-35B-A3B. Note
that this makes the venv formally inconsistent with Torch 2.12's declared
`nvidia-cublas<=13.1.1.3` dependency, so keep it isolated to MCA experiments.

After the TE runtime fix, MCA smoke `123828` got past the earlier
`TESpecProvider` failure and began constructing `Qwen3_5MoeConfig`, but failed
before training with:

```text
AssertionError: wsd_decay_steps is required for WSD
```

The MCA/Megatron scheduler path does not apply LLaMA-Factory's default WSD
split, so WSD must be explicit in MCA configs. The 12-step MCA smoke now uses:

```yaml
lr_scheduler_type: warmup_stable_decay
lr_scheduler_kwargs:
  wsd_decay_steps: 8
  lr_wsd_decay_style: cosine
```

This corrected MCA smoke was resubmitted as job `123829` with run name
`adp-bench-qwen35-35b-a3b-mca-pp4-ep4-seq32768-smoke6`.

Job `123829` passed scheduler setup and reached
`mcore_adapter.trainer: ***** Running training *****`, but failed on the first
forward pass inside Transformer Engine fused attention:

```text
RuntimeError: Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13
```

The traceback ended in `transformer_engine.pytorch.cpp_extensions.fused_attn`.
The next smoke keeps `transformer_impl=transformer_engine` for Qwen3.5
gated-delta attention, but forces the attention backend away from TE fused
attention via environment variables:

```bash
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
```

A local `AutoConfig.from_pretrained(...)` probe with these variables reported
`attention_backend=AttnBackend.flash` and
`transformer_impl=transformer_engine`. This MCA smoke is labeled `smoke7`.

Smoke7 was submitted as job `123830`. It reached
`mcore_adapter.trainer: ***** Running training *****` and got into the first
backward pass, but failed before logging loss/W&B training metrics with:

```text
RuntimeError: Triton Error [CUDA]: out of memory
```

The traceback came from FLA's Triton autotuned `l2norm_bwd_kernel` during
Qwen3.5 gated-delta attention backward. The GPU monitor showed local ranks 4-7
on the second node at roughly 80GB used at failure, while the earlier steady
state was around 40-49GB per rank. This makes the next MCA smoke a
per-rank-activation-memory test rather than a launcher/runtime test:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_pp4_ep2_cp2_smoke.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_pp4_ep2_cp2_smoke.sbatch
parallelism: TP=1, PP=4, EP=2, CP=2, GAS=8
reason: use context parallelism to split the 32k sequence across two ranks and
        keep the optimizer-step batch comparable after the data-parallel group
        drops from 4 to 2.
```

Job `123831` failed during model construction before training:

```text
AssertionError: Gated delta net does not support context parallel for now, but got self.context_parallel_size=2.
```

For Qwen3.5 gated-delta MCA, `context_parallel_size > 1` is therefore not a
usable memory-reduction knob yet. The next smoke keeps CP disabled and tries
tensor parallelism instead:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke.sbatch
parallelism: TP=2, PP=4, EP=2, CP=1, GAS=8
reason: shard tensor dimensions and let MCA enable sequence parallelism for
        TP+EP, without using the unsupported gated-delta context-parallel path.
```

Job `123832` (`smoke1`) reached training and logged one loss:

```text
{'loss': '0.8861', 'grad_norm': '9.925', 'learning_rate': '1e-05',
 'skipped_iter': 0, 'num_zeros_in_grad': 0,
 'token_per_sec_per_gpu': '1092', 'epoch': '4.117e-05'}
```

Peak memory stayed well below the smoke7 OOM level, roughly 45-62GB depending
on rank/stage. It then failed on the second forward at embedding
sequence-parallel reduce-scatter:

```text
AssertionError: First dimension of the tensor should be divisible by tensor parallel size
```

The root cause is MCA padding each optimizer step to the local
gradient-accumulation max sequence length, which can be odd even though the
nominal cutoff is 32768. The ADP patch helper now rounds MCA's padded step
length up to a multiple of `tensor_model_parallel_size` before `_pad_batched_inputs`.
The patched rerun is:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke2.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke2.sbatch
parallelism: TP=2, PP=4, EP=2, CP=1, GAS=8
```

Job `123833` (`smoke2`) completed successfully:

```text
state: COMPLETED
elapsed: 00:10:02
exit_code: 0:0
train_runtime: 517.4s
train_steps_per_second: 0.023
train_loss: 0.7112
```

The first two steps were dominated by startup/autotune, but later steps reached
roughly 15-18 seconds per optimizer step in the progress log. The logged
per-step throughput after warmup was typically around 15k-17k tokens/sec/GPU,
substantially above the previous DeepSpeed hpZ8 fused-MoE smoke measurement.
Memory was tight but no longer OOMed: the most loaded late-stage ranks on the
second node reached roughly 79-80GB H100 memory. Transformer Engine and
Megatron both warned that tensor/sequence-parallel overlap is fastest with:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

The follow-up smoke keeps the same TP2/PP4/EP2/CP1/GAS8 geometry and adds only
that environment setting:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke3.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke3.sbatch
```

Job `123834` (`smoke3`) also completed, but it was not a speed improvement:

```text
state: COMPLETED
elapsed: 00:10:08
exit_code: 0:0
train_runtime: 523.6s
train_steps_per_second: 0.023
train_loss: 0.7102
```

The setting did remove the Megatron/Transformer Engine warning about
`CUDA_DEVICE_MAX_CONNECTIONS`, but late-step throughput remained essentially
unchanged: the stable steps were again around 15k-17k tokens/sec/GPU. Peak GPU
memory was also unchanged at roughly 80.36GB, concentrated on the last local
pipeline ranks. This means the connection setting is harmless, but not a
measurable throughput fix in this small smoke. The peak memory also means that
disabling full recomputation is not viable in the current TP2/PP4/EP2 layout
without first reducing activation/model memory elsewhere.

Transformer Engine also warned that FA3/FA4 may improve feature support or
performance. The normal MCA venv remains on `flash_attn==2.8.3`; an isolated
hardlink clone `.venv_mca_fa4` was created for the prerelease FA4 test. A
plain `flash-attn-4[cu13]==4.0.0b11` install downgraded `nvidia-cublas` and
broke Transformer Engine, and the latest prerelease Cutlass DSL caused
`flash_attn.cute` import errors. The working import combination for the clone
is:

```text
flash-attn-4==4.0.0b11
nvidia-cutlass-dsl[cu13]==4.4.2
nvidia-cublas==13.5.1.27  # force-reinstalled with --no-deps for TE runtime
```

The FA4 smoke keeps the same TP2/PP4/EP2/CP1/GAS8 geometry and points only the
launcher venv at `.venv_mca_fa4`:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke4_fa4.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke4_fa4.sbatch
```

Job `123842` (`smoke4_fa4`) completed:

```text
state: COMPLETED
elapsed: 00:11:19
exit_code: 0:0
train_runtime: 542.1s
train_steps_per_second: 0.022
train_loss: 0.7082
```

The 12-step total runtime was worse than smoke2/smoke3 because startup and
early steps were slower. The later per-step `token_per_sec_per_gpu` values were
however higher than smoke3 on the same step range, roughly 17.5k average across
steps 8-12 versus roughly 16.5k for smoke3. FA4 did not eliminate all attention
warnings: the log still emitted a `flash-attn v3` warning. Peak memory rose
slightly to about 80.5GB on the tight late-stage ranks. Because the full-run
startup cost would be amortized, the next FA4 smoke extends the same setup to
50 steps:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke5_fa4_50step.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke5_fa4_50step.sbatch
```

Job `123844` (`smoke5_fa4_50step`) completed:

```text
state: COMPLETED
elapsed: 00:20:16
exit_code: 0:0
train_runtime: 1131.0s
train_steps_per_second: 0.044
train_loss: 0.6549
```

The run stayed healthy and logged losses through 50 steps. After startup, the
progress bar settled mostly around 15-16 seconds per step. To determine whether
this is actually better than the non-FA4 environment, the matched baseline is a
50-step rerun of smoke3 with the same WSD schedule and no FA4 clone:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke6_base_50step.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke6_base_50step.sbatch
```

Job `123845` (`smoke6_base_50step`) completed:

```text
state: COMPLETED
elapsed: 00:20:23
exit_code: 0:0
train_runtime: 1136.0s
train_steps_per_second: 0.044
train_loss: 0.6541
```

This makes FA4 a wash rather than a clear win:

```text
50-step FA4 clone: 1131.0s train_runtime, 0.044 steps/s, peak ~80.44GB
50-step base venv: 1136.0s train_runtime, 0.044 steps/s, peak ~80.50GB
```

The FA4 run had some higher per-step `token_per_sec_per_gpu` readings, but the
end-to-end 50-step runtime differed by only about 0.4%, while the FA4 clone
requires a fragile package combination and still emits a `flash-attn v3`
warning. For now, keep the regular `.venv_mca` TP2/PP4/EP2 recipe as the
practical baseline; FA4 is not worth carrying into full runs unless a newer
Transformer Engine / FA4 stack becomes cleaner.

FA3 source build note: a full Hopper `flash-attn-3` source build tried to
compile 293 objects including SM80 and many unused dtypes/head dimensions. For
the H100-only Qwen3.5-35B-A3B comparison, the `.venv_mca_fa3` environment was
instead built from the FA3 Hopper source with only SM90, bf16, hdim256, and
training backward enabled:

```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
python setup.py install
```

This reduced the build to four objects and installed `flash-attn-3 3.0.0`.
Transformer Engine imports now report `FlashAttentionUtils.v3_is_installed ==
True` and `fa3_version == 3.0.0` in `.venv_mca_fa3`. The matched 50-step smoke
keeps the same TP2/PP4/EP2/CP1/GAS8 geometry as smoke6 but runs from the FA3
venv:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke7_fa3_50step.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke7_fa3_50step.sbatch
```

Job `123848` (`smoke7_fa3_50step`) completed:

```text
state: COMPLETED
elapsed: 00:19:49
exit_code: 0:0
train_runtime: 1091.0s
train_steps_per_second: 0.046
train_loss: 0.6538
```

FA3 is a modest improvement over the matched 50-step baseline, but not a
fundamental fix:

```text
50-step FA3 clone: 1091.0s train_runtime, 0.046 steps/s
50-step FA4 clone: 1131.0s train_runtime, 0.044 steps/s
50-step base venv: 1136.0s train_runtime, 0.044 steps/s
```

The FA3 run removed the previous Transformer Engine warning about missing
flash-attn v3, replacing it with a flash-attn v4 recommendation. Post-warmup
steps were usually around 14.5-17s with per-step throughput roughly
16k-19k tokens/sec/GPU. This is worth keeping as the current best MCA
environment, but the roughly 4% end-to-end speedup means the main remaining
bottlenecks are still parallelism, communication, and Qwen3.5 gated-delta/FLA
kernel behavior rather than the FlashAttention package alone.

Selective recompute was tested to see whether full-layer activation
checkpointing could be reduced. Job `123849` (`smoke8_fa3_selective_50step`)
used the same FA3 TP2/PP4/EP2 geometry but changed:

```yaml
recompute_granularity: selective
recompute_method: null
recompute_modules: core_attn
recompute_num_layers: null
```

This failed before logging loss:

```text
state: FAILED
elapsed: 00:02:23
exit_code: 143:0
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00-1.09 GiB.
GPU 0/1 had only about 600-655 MiB free with roughly 78.5 GiB in use.
```

The traceback reached the first forward pass through the MoE path:

```text
MoELayer.forward
token_dispatcher.combine_preprocess
reduce_scatter_to_sequence_parallel_region
torch.empty_like(input_tensor_list[rank])
```

Megatron also warned that `core_attn` recompute is usually unnecessary with a
Transformer Engine fused attention backend. This negative result means the
current run is not mainly limited by attention activation memory; removing
full-layer recompute immediately exposes MoE/token-dispatch memory. The next
selective recompute smoke therefore targets routed MoE internals instead:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke9_fa3_selective_moe_50step.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke9_fa3_selective_moe_50step.sbatch
recompute_modules: moe,moe_act
```

Job `123850` (`smoke9_fa3_selective_moe_50step`) also failed before logging
loss:

```text
state: FAILED
elapsed: 00:02:38
exit_code: 143:0
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.24 GiB.
GPU 3 had 1.15 GiB free with roughly 78.02 GiB in use.
```

This time the traceback confirmed that MoE-layer checkpointing was active:

```text
MoELayer.forward
tensor_parallel.checkpoint
routed_experts_compute
token_dispatcher.combine_preprocess
reduce_scatter_to_sequence_parallel_region
torch.empty_like(input_tensor_list[rank])
```

So selective MoE recompute reduced live memory compared with `core_attn` only,
but not enough to fit the first forward. Because the failed allocation missed
by about 90 MiB, the next smoke adds `layernorm` recompute to drop the
pre-MLP layernorm state around the MoE path while still avoiding full
gated-delta attention recompute:

```text
config: configs/full_condenser_24k_all_records_v2_adapted/qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke10_fa3_selective_moe_lnorm_50step.yaml
launcher: scripts/run_qwen35_35b_a3b_mca_tp2_pp4_ep2_smoke10_fa3_selective_moe_lnorm_50step.sbatch
recompute_modules: moe,moe_act,layernorm
```

Open MCA memory/speed candidates after the TP2 smoke:

- Implement context-parallel gated-delta attention for Qwen3.5/MCA so the 32k
  sequence can be split across ranks without hitting the current assertion.
  This is likely the cleanest way to make CP a valid memory-reduction path, but
  it requires changing the Qwen3.5 gated-delta/FLA integration rather than only
  changing YAML. The current blocker is in Megatron Core, not just
  LLaMA-Factory: `TransformerConfig` asserts `context_parallel_size == 1` for
  `experimental_attention_variant: gated_delta_net`, and
  `megatron/core/ssm/gated_delta_net.py` has a TODO for
  `GatedDeltaNetContextParallel`. The closest upstream template is
  `megatron/core/ssm/mamba_context_parallel.py`, but gated delta will need
  correct propagation of recurrent boundary state across CP shards rather than
  naive sequence slicing.
- If TP2 still OOMs in `l2norm_bwd_kernel`, add a reproducible ADP patch to
  constrain or pre-warm FLA's Triton l2norm backward autotuning, because job
  `123830` failed during autotune/first backward at near-full H100 memory.
- Try a different pipeline split such as PP8/EP2 if tensor parallelism works
  but leaves uneven late-stage memory pressure.

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

The MCA smoke at this point was:

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
variant is present, so MCA was blocked until a compatible Transformer Engine
stack was available. The TE build/runtime notes above describe the fix; the MCA
launcher now exports the isolated venv's CUDA 13, cuDNN, and NCCL libraries so
future MCA submissions load that working stack.

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
`cuda_fused_moe` settings for a 12-step smoke. This isolates activation
recompute overhead from the fused-MoE kernel choice. LLaMA-Factory's model
preparation path does **not** use only Hugging Face's
`gradient_checkpointing: false` for this; it checks the model argument
`disable_gradient_checkpointing`. Correct no-GC configs should therefore set
both:

```yaml
gradient_checkpointing: false
disable_gradient_checkpointing: true
```

The first queued job for this fallback was:

```text
job: 123825
run: adp-bench-qwen35-35b-a3b-hpz8-cuda-fused-moe-no-gc-seq32768-smoke
```

Job `123825` failed before launch because the patch helper tried to import the
optional MCA workflow in the base venv, where `mcore_adapter` is intentionally
not installed. The helper now skips optional MCA patches on `ImportError`; the
same smoke was resubmitted as job `123826`.

Job `123826` completed, but it was **not** a valid no-gradient-checkpointing
measurement: the log still included `Gradient checkpointing enabled.` twice.
The resulting timing closely matched the checkpointed hpZ8 fused-MoE smoke
because checkpointing was still active. The corrected smoke adds
`disable_gradient_checkpointing: true`, uses run name
`adp-bench-qwen35-35b-a3b-hpz8-cuda-fused-moe-disable-gc-seq32768-smoke`, and
was submitted as job `123827`.

Job `123827` did honor the no-gradient-checkpointing setting: the log did not
include `Gradient checkpointing enabled.` It failed on the first real training
step with CUDA OOM before logging a loss:

```text
rank12: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 876 MiB.
GPU 4 total capacity: 79.18 GiB
process memory in use: 78.63 GiB
allocated by PyTorch: 75.32 GiB
reserved but unallocated: 1.20 GiB
```

Conclusion: for the current 35B-A3B hpZ8, 32k-context, microbatch-1 recipe,
gradient checkpointing is not optional. It almost certainly costs extra
recompute, but disabling it exceeds 80GB H100 memory before a training step can
complete.

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

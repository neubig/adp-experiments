#!/usr/bin/env python3
"""Patch training libraries for ADP long-context Qwen3.5 runs.

Qwen3.5 long-context SFT runs with Liger avoid materializing full
batch x sequence x vocab logits during training. In eval, Hugging Face Trainer
sets the model to eval mode, so Liger's default Qwen3.5 forward path computes
full logits before loss. At 32k context this can OOM even when training fits.

This patch makes LLaMA-Factory pass ``skip_logits=True`` to model.forward only
for loss-only SFT eval/prediction steps. It should be paired with
``prediction_loss_only: true`` in the YAML config.

It also teaches LLaMA-Factory's Liger dispatch table about ``qwen3_5_moe``.
Recent Liger releases include Qwen3.5-MoE fused-linear-cross-entropy support,
but some LLaMA-Factory versions only dispatch Liger for the dense ``qwen3_5``
model type. Without this, Qwen3.5-MoE training materializes full logits and
OOMs before the first 32k step.

For kernel benchmarks, Liger's Qwen3.5-MoE SwiGLU patch rewrites instantiated
expert modules to ``LigerExperts``. LLaMA-Factory's v1 ``cuda_fused_moe`` matcher
must recognize that class name or the fused MoE kernel silently skips the model.

Finally, it lets a DeepSpeed ZeRO-3 run continue from a Hugging Face-format
model-only checkpoint when the model is already loaded from that checkpoint.
This preserves Trainer's data skipping via ``resume_from_checkpoint`` even when
the previous run used ``save_only_model: true`` and therefore did not write
DeepSpeed optimizer/scheduler state.

For MCA tensor/context-parallel runs, it also rounds the per-step padded
sequence length up to a safe TP/CP multiple. Megatron sequence parallel uses
reduce-scatter along the sequence dimension, and Megatron CP reshapes the input
into `2 * context_parallel_size` chunks before selecting the two local chunks.
Without this, TP/CP runs can fail on any gradient-accumulation group whose local
max sequence length is not divisible by the combined layout.

For Qwen3.5 full-attention layers with fewer KV groups than TP ranks, it mirrors
Megatron's query sub-slice onto the output gate. Without this, TP4+ runs keep a
full gathered gate tensor while the attention output is already rank-local.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys


PATCH_MARKER = "# ADP patch: force Liger loss-only eval to skip logits."
MOE_PATCH_MARKER = "# ADP patch: enable Liger for Qwen3.5-MoE."
DS_MODEL_ONLY_PATCH_MARKER = "# ADP patch: tolerate HF model-only checkpoint for DeepSpeed resume."
DS_MODEL_ONLY_SCHEDULER_PATCH_MARKER = (
    "# ADP patch: skip missing DeepSpeed scheduler state for HF model-only checkpoint."
)
SKIP_FINAL_SAVE_PATCH_MARKER = "# ADP patch: optionally skip benchmark final save."
SKIP_FINAL_PLOT_PATCH_MARKER = "# ADP patch: skip loss plotting when benchmark state is skipped."
FUSED_MOE_LIGER_EXPERTS_PATCH_MARKER = "# ADP patch: cuda_fused_moe recognizes LigerExperts."
MCA_SKIP_FINAL_SAVE_PATCH_MARKER = "# ADP patch: optionally skip MCA benchmark final save."
MCA_SKIP_FINAL_PLOT_PATCH_MARKER = "# ADP patch: skip MCA loss plotting when benchmark state is skipped."
MCA_QWEN35_LINEAR_CONFIG_PATCH_MARKER = "# ADP patch: accept Qwen3.5 linear-attention config fields."
MCA_TP_SEQ_LENGTH_PATCH_MARKER = "# ADP patch: round MCA step sequence length for tensor/context parallelism."
MCA_QWEN35_TP_OUTPUT_GATE_PATCH_MARKER = "# ADP patch: slice Qwen3.5 output gate for KV-heads < TP."
MCA_CONTIGUOUS_CP_BATCH_PATCH_MARKER = "# ADP patch: optional contiguous CP batch slicing for FLA GDN."
MCA_QWEN35_CONTIGUOUS_CP_IMPORT_PATCH_MARKER = "# ADP patch: import os for optional contiguous CP mode."
MCA_QWEN35_CONTIGUOUS_CP_RANGE_PATCH_MARKER = "# ADP patch: optional contiguous CP input ranges."
MCA_GDN_CP_IMPORT_PATCH_MARKER = "# ADP patch: import FLA context-parallel GDN helpers."
MCA_GDN_CP_INIT_PATCH_MARKER = "# ADP patch: record GDN context-parallel process group."
MCA_GDN_CP_FORWARD_PATCH_MARKER = "# ADP patch: pass FLA context-parallel metadata through GDN."
MCA_GDN_CP_CONV_PATCH_MARKER = "# ADP patch: use FLA context-parallel causal convolution in GDN."
MCA_GDN_CP_RULE_PATCH_MARKER = "# ADP patch: use FLA context-parallel gated-delta rule in GDN."
MCA_GDN_CP_CONFIG_PATCH_MARKER = "# ADP patch: allow experimental GDN context parallelism."
MCA_GDN_CP_DIST_IMPORT_PATCH_MARKER = "# ADP patch: import distributed helpers for GDN CP layout bridge."
MCA_GDN_CP_LAYOUT_HELPERS_PATCH_MARKER = "# ADP patch: load-balanced CP layout bridge for GDN."
MCA_GDN_CP_UNDO_LAYOUT_PATCH_MARKER = "# ADP patch: convert load-balanced CP layout to contiguous GDN chunks."
MCA_GDN_CP_REDO_LAYOUT_PATCH_MARKER = "# ADP patch: restore load-balanced CP layout after GDN."
MCA_GDN_CP_DETERMINISTIC_GUARD_PATCH_MARKER = "# ADP patch: disallow deterministic fallback for GDN CP."
OLD = """        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
"""
NEW = f"""        if prediction_loss_only and labels is not None and not self.args.predict_with_generate:
            inputs = dict(inputs)
            {PATCH_MARKER}
            inputs["skip_logits"] = True

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
"""
MOE_OLD = """    elif model_type == "qwen3_5":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as apply_liger_kernel
"""
MOE_NEW = f"""    elif model_type == "qwen3_5":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as apply_liger_kernel
    elif model_type in ["qwen3_5_moe", "qwen3_5_moe_text"]:
        {MOE_PATCH_MARKER}
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe as apply_liger_kernel
"""
DS_MODEL_ONLY_OLD = """    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")
"""
DS_MODEL_ONLY_NEW = f"""    else:
        hf_model_only_checkpoint = (
            glob.glob(f"{{checkpoint_path}}/model*.safetensors")
            or glob.glob(f"{{checkpoint_path}}/pytorch_model*.bin")
            or glob.glob(f"{{checkpoint_path}}/adapter_model*.safetensors")
        )
        if hf_model_only_checkpoint:
            {DS_MODEL_ONLY_PATCH_MARKER}
            logger.warning(
                "No DeepSpeed global_step* state found at %s, but Hugging Face-format "
                "model weights are present. Continuing without DeepSpeed optimizer/"
                "scheduler state; set model_name_or_path to this checkpoint so model "
                "weights are loaded before Trainer skips previously seen batches.",
                checkpoint_path,
            )
            return

        raise ValueError(f"Can't find a valid checkpoint at {{checkpoint_path}}")
"""
DS_MODEL_ONLY_SCHEDULER_OLD = """        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return
"""
DS_MODEL_ONLY_SCHEDULER_NEW = f"""        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                scheduler_path = os.path.join(checkpoint, SCHEDULER_NAME)
                if not os.path.isfile(scheduler_path):
                    hf_model_only_checkpoint = (
                        glob.glob(f"{{checkpoint}}/model*.safetensors")
                        or glob.glob(f"{{checkpoint}}/pytorch_model*.bin")
                        or glob.glob(f"{{checkpoint}}/adapter_model*.safetensors")
                    )
                    if hf_model_only_checkpoint:
                        {DS_MODEL_ONLY_SCHEDULER_PATCH_MARKER}
                        logger.warning(
                            "No DeepSpeed scheduler state found at %s. Continuing with the "
                            "new scheduler because this is a Hugging Face-format model-only checkpoint.",
                            scheduler_path,
                        )
                        return

                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True))
                reissue_pt_warnings(caught_warnings)
            return
"""
SKIP_FINAL_SAVE_OLD = """        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
"""
SKIP_FINAL_SAVE_NEW = f"""        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        adp_skip_final_save = __import__("os").environ.get("ADP_LF_SKIP_FINAL_SAVE", "").lower() in {{
            "1",
            "true",
            "yes",
        }}
        if adp_skip_final_save:
            {SKIP_FINAL_SAVE_PATCH_MARKER}
            logger.warning("ADP_LF_SKIP_FINAL_SAVE=1: skipping final save_model/save_state for benchmark run.")
        else:
            trainer.save_model()

        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if not adp_skip_final_save:
            trainer.save_state()
"""
SKIP_FINAL_PLOT_OLD = """        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
"""
SKIP_FINAL_PLOT_NEW = f"""        if (
            trainer.is_world_process_zero()
            and finetuning_args.plot_loss
            and not adp_skip_final_save  {SKIP_FINAL_PLOT_PATCH_MARKER}
        ):
"""
MCA_SKIP_FINAL_SAVE_OLD = """    train_result = trainer.train(training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
"""
MCA_SKIP_FINAL_SAVE_NEW = f"""    train_result = trainer.train(training_args.resume_from_checkpoint)
    adp_skip_final_save = __import__("os").environ.get("ADP_LF_SKIP_FINAL_SAVE", "").lower() in {{
        "1",
        "true",
        "yes",
    }}
    if adp_skip_final_save:
        {MCA_SKIP_FINAL_SAVE_PATCH_MARKER}
        logger.warning("ADP_LF_SKIP_FINAL_SAVE=1: skipping final MCA save_model/save_state for benchmark run.")
    else:
        trainer.save_model()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if not adp_skip_final_save:
        trainer.save_state()
"""
MCA_SKIP_FINAL_PLOT_OLD = """    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
"""
MCA_SKIP_FINAL_PLOT_NEW = f"""    if (
        trainer.is_world_process_zero()
        and finetuning_args.plot_loss
        and not adp_skip_final_save  {MCA_SKIP_FINAL_PLOT_PATCH_MARKER}
    ):
"""
MCA_QWEN35_LINEAR_CONFIG_OLD = """    # Gated Delta Net specific (for linear attention layers)
    layer_types: Optional[list[str]] = None

    # Vision specific
"""
MCA_QWEN35_LINEAR_CONFIG_NEW = f"""    # Gated Delta Net specific (for linear attention layers)
    layer_types: Optional[list[str]] = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_attention_freq: int = 4
    attention_output_gate: bool = True
    experimental_attention_variant: Optional[str] = None
    moe_shared_expert_gate: bool = True
    {MCA_QWEN35_LINEAR_CONFIG_PATCH_MARKER}

    # Vision specific
"""
FUSED_MOE_LIGER_EXPERTS_OLD = """    "Qwen3_5MoeForCausalLM": {
        "Qwen3_5MoeExperts": _triton_moe_experts_forward,
    },
    "Qwen3_5MoeForConditionalGeneration": {
        "Qwen3_5MoeExperts": _triton_moe_experts_forward,
    },
"""
FUSED_MOE_LIGER_EXPERTS_NEW = f"""    "Qwen3_5MoeForCausalLM": {{
        "Qwen3_5MoeExperts": _triton_moe_experts_forward,
        "LigerExperts": _triton_moe_experts_forward,  {FUSED_MOE_LIGER_EXPERTS_PATCH_MARKER}
    }},
    "Qwen3_5MoeForConditionalGeneration": {{
        "Qwen3_5MoeExperts": _triton_moe_experts_forward,
        "LigerExperts": _triton_moe_experts_forward,
    }},
"""
MCA_TP_SEQ_LENGTH_OLD = """        if len(step_inputs) < self.args.gradient_accumulation_steps:
            return None, 0, 0

        if not self.args.allow_variable_seq_lengths():
            step_inputs = [self._pad_batched_inputs(inputs, max_seq_length) for inputs in step_inputs]
"""
MCA_TP_SEQ_LENGTH_NEW = f"""        if len(step_inputs) < self.args.gradient_accumulation_steps:
            return None, 0, 0

        tensor_parallel_size = getattr(self.args, "tensor_model_parallel_size", 1) or 1
        context_parallel_size = getattr(self.args, "context_parallel_size", 1) or 1
        alignment = tensor_parallel_size
        if context_parallel_size > 1:
            cp_view_alignment = 2 * context_parallel_size
            sp_alignment = tensor_parallel_size * context_parallel_size
            alignment = math.lcm(alignment, cp_view_alignment, sp_alignment)
        if alignment > 1 and not self.args.allow_variable_seq_lengths():
            remainder = max_seq_length % alignment
            if remainder:
                {MCA_TP_SEQ_LENGTH_PATCH_MARKER}
                max_seq_length += alignment - remainder

        if not self.args.allow_variable_seq_lengths():
            step_inputs = [self._pad_batched_inputs(inputs, max_seq_length) for inputs in step_inputs]
"""
MCA_TP_SEQ_LENGTH_TP_ONLY_OLD = """        if len(step_inputs) < self.args.gradient_accumulation_steps:
            return None, 0, 0

        tensor_parallel_size = getattr(self.args, "tensor_model_parallel_size", 1) or 1
        if tensor_parallel_size > 1 and not self.args.allow_variable_seq_lengths():
            remainder = max_seq_length % tensor_parallel_size
            if remainder:
                # ADP patch: round MCA step sequence length for tensor parallelism.
                max_seq_length += tensor_parallel_size - remainder

        if not self.args.allow_variable_seq_lengths():
            step_inputs = [self._pad_batched_inputs(inputs, max_seq_length) for inputs in step_inputs]
"""
MCA_QWEN35_TP_OUTPUT_GATE_OLD = """        if output_gate:
            # Gate [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
            gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            return query, key, value, gate
"""
MCA_QWEN35_TP_OUTPUT_GATE_NEW = f"""        if output_gate:
            # Gate [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
            gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)
            if self.config.num_query_groups < self.world_size:
                # Mirror the query sub-slice above when TP exceeds the number
                # of KV groups. Without this, TP4+ Qwen3.5 attention keeps the
                # full gathered gate while core_attn_out is rank-local.
                idx = get_tensor_model_parallel_rank() % (
                    self.world_size // self.config.num_query_groups
                )
                size = self.num_attention_heads_per_partition // (
                    self.world_size // self.config.num_query_groups
                )
                gate = gate[:, :, idx * size : (idx + 1) * size, :]
                {MCA_QWEN35_TP_OUTPUT_GATE_PATCH_MARKER}
            return query, key, value, gate
"""
MCA_CONTIGUOUS_CP_BATCH_OLD = """            cp_rank = mpu.get_context_parallel_rank()
            for key, val in batch.items():
"""
MCA_CONTIGUOUS_CP_BATCH_NEW = f"""            cp_rank = mpu.get_context_parallel_rank()
            if os.environ.get("ADP_MCA_CONTIGUOUS_CP", "").lower() in {{"1", "true", "yes"}}:
                {MCA_CONTIGUOUS_CP_BATCH_PATCH_MARKER}
                for key, val in batch.items():
                    if val is not None and isinstance(val, torch.Tensor):
                        seq_dim = 2 if key in dim3_keys else 1
                        local_seq_len = val.shape[seq_dim] // cp_size
                        val = val.narrow(seq_dim, cp_rank * local_seq_len, local_seq_len).contiguous()
                        batch[key] = val
                return batch

            for key, val in batch.items():
"""
MCA_QWEN35_CONTIGUOUS_CP_IMPORT_OLD = """import heapq
import itertools
from typing import Optional
"""
MCA_QWEN35_CONTIGUOUS_CP_IMPORT_NEW = f"""import heapq
import itertools
import os
from typing import Optional

{MCA_QWEN35_CONTIGUOUS_CP_IMPORT_PATCH_MARKER}
"""
MCA_QWEN35_CONTIGUOUS_CP_RANGE_OLD = """        if self.config.context_parallel_size <= 1:
            return [list(get_sequence_range(0, total_seqlen, slice_rank, slice_size))]
        cp_rank = mpu.get_context_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        left_start = (total_seqlen // cp_size // 2) * cp_rank
"""
MCA_QWEN35_CONTIGUOUS_CP_RANGE_NEW = f"""        if self.config.context_parallel_size <= 1:
            return [list(get_sequence_range(0, total_seqlen, slice_rank, slice_size))]
        cp_rank = mpu.get_context_parallel_rank()
        cp_size = mpu.get_context_parallel_world_size()
        if os.environ.get("ADP_MCA_CONTIGUOUS_CP", "").lower() in {{"1", "true", "yes"}}:
            {MCA_QWEN35_CONTIGUOUS_CP_RANGE_PATCH_MARKER}
            cp_start, cp_end = get_sequence_range(0, total_seqlen, cp_rank, cp_size)
            return [list(get_sequence_range(cp_start, cp_end, slice_rank, slice_size))]
        left_start = (total_seqlen // cp_size // 2) * cp_rank
"""
MCA_GDN_CP_IMPORT_OLD = """try:
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None

    HAVE_FLA = False
"""
MCA_GDN_CP_IMPORT_NEW = f"""try:
    from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    {MCA_GDN_CP_IMPORT_PATCH_MARKER}
    HAVE_FLA = True
except ImportError:
    build_cp_context = None
    chunk_gated_delta_rule = None
    fla_causal_conv1d = None

    HAVE_FLA = False
"""
MCA_GDN_CP_INIT_OLD = """        # TODO: support CP

        self.reset_parameters()
"""
MCA_GDN_CP_INIT_NEW = f"""        self.cp_group = getattr(self.pg_collection, "cp", None)
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1
        {MCA_GDN_CP_INIT_PATCH_MARKER}

        self.reset_parameters()
"""
MCA_GDN_CP_FORWARD_OLD = """        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        if inference_context is not None:
"""
MCA_GDN_CP_FORWARD_NEW = f"""        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        cp_context = None
        if self.cp_size > 1:
            if batch != 1:
                raise NotImplementedError(
                    "ADP experimental GDN context parallelism currently expects microbatch size 1."
                )
            global_seq_len = seq_len * self.cp_size
            cu_seqlens_cpu = torch.tensor([0, global_seq_len], dtype=torch.long)
            cu_seqlens = cu_seqlens_cpu.to(device=hidden_states.device, non_blocking=True)
            {MCA_GDN_CP_FORWARD_PATCH_MARKER}
            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens,
                group=self.cp_group,
                conv1d_kernel_size=self.conv_kernel_dim,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )

        if inference_context is not None:
"""
MCA_GDN_CP_CONV_OLD = """        if (causal_conv1d_fn is None) or self.config.deterministic_mode:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"]
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
            )
"""
MCA_GDN_CP_CONV_NEW = """        if cp_context is not None:
            assert self.activation in ["silu", "swish"]
            # ADP patch: use FLA context-parallel causal convolution in GDN.
            qkv, _ = fla_causal_conv1d(
                x=qkv.transpose(1, 2).contiguous(),
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
                cp_context=cp_context,
            )
            qkv = qkv.transpose(1, 2).contiguous()
        elif (causal_conv1d_fn is None) or self.config.deterministic_mode:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"]
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
            )
"""
MCA_GDN_CP_RULE_OLD = """            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
"""
MCA_GDN_CP_RULE_NEW = """            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                # ADP patch: use FLA context-parallel gated-delta rule in GDN.
                cp_context=cp_context,
            )
"""
MCA_GDN_CP_CONFIG_OLD = """            # Do not support yet, but coming soon.
            assert self.context_parallel_size == 1, (
                f"Gated delta net does not support context parallel for now,"
                f" but got {self.context_parallel_size=}."
            )

        if self.fp8:
"""
MCA_GDN_CP_CONFIG_NEW = f"""            if self.context_parallel_size != 1:
                {MCA_GDN_CP_CONFIG_PATCH_MARKER}
                warnings.warn(
                    "ADP experimental patch enables GatedDeltaNet context parallelism via "
                    "FLA operator-level CP. This path has only been smoke-tested for "
                    "microbatch size 1."
                )

        if self.fp8:
"""
MCA_GDN_CP_DIST_IMPORT_OLD = """import torch
import torch.nn as nn
"""
MCA_GDN_CP_DIST_IMPORT_NEW = f"""import os

import torch
import torch.distributed as dist
import torch.nn as nn

{MCA_GDN_CP_DIST_IMPORT_PATCH_MARKER}
"""
MCA_GDN_CP_LAYOUT_HELPERS_OLD = """logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetSubmodules:
"""
MCA_GDN_CP_LAYOUT_HELPERS_NEW = f"""logger = logging.getLogger(__name__)


def _adp_env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {{"1", "true", "yes"}}


class _ADPCPAllGatherConcat(torch.autograd.Function):
    \"\"\"Autograd-safe all-gather/concat over a CP group.\"\"\"

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: dist.ProcessGroup, dim: int):
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        world_size = dist.get_world_size(group)
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor.contiguous(), group=group)
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.local_dim_size = local_tensor.size(dim)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_full = grad_output.contiguous()
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_dim_size
        grad_local = grad_full.narrow(ctx.dim, start, ctx.local_dim_size).contiguous()
        return grad_local, None, None


def _adp_cp_all_gather_concat(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    dim: int,
    differentiable: bool,
) -> torch.Tensor:
    if differentiable:
        return _ADPCPAllGatherConcat.apply(tensor, group, dim)

    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _adp_cp_local_positions(seq_len: int, group: dist.ProcessGroup, device: torch.device) -> torch.Tensor:
    cp_rank = dist.get_rank(group)
    cp_size = dist.get_world_size(group)

    if _adp_env_flag("ADP_MCA_CONTIGUOUS_CP"):
        start = cp_rank * seq_len
        return torch.arange(start, start + seq_len, device=device, dtype=torch.long)

    if seq_len % 2 != 0:
        raise RuntimeError(
            "Megatron load-balanced CP gives each rank two equal chunks, so the "
            f"rank-local sequence length must be even. Got {{seq_len}}."
        )

    chunk_len = seq_len // 2
    left_start = cp_rank * chunk_len
    right_start = (2 * cp_size - cp_rank - 1) * chunk_len
    left = torch.arange(left_start, left_start + chunk_len, device=device, dtype=torch.long)
    right = torch.arange(right_start, right_start + chunk_len, device=device, dtype=torch.long)
    return torch.cat((left, right), dim=0)


def _adp_cp_undo_load_balancing(
    tensor: torch.Tensor,
    local_positions: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cp_rank = dist.get_rank(group)
    local_seq_len = tensor.size(dim)
    cp_order_tensor = _adp_cp_all_gather_concat(tensor, group, dim=dim, differentiable=True)
    cp_order_positions = _adp_cp_all_gather_concat(
        local_positions, group, dim=0, differentiable=False
    )
    sort_order = torch.argsort(cp_order_positions)
    sorted_positions = cp_order_positions.index_select(0, sort_order)
    expected_positions = torch.arange(
        sorted_positions.numel(), device=sorted_positions.device, dtype=sorted_positions.dtype
    )
    if not torch.equal(sorted_positions, expected_positions):
        raise RuntimeError(
            "GDN CP layout bridge requires dense global token positions covering 0..S-1."
        )

    full_tensor = cp_order_tensor.index_select(dim, sort_order)
    start = cp_rank * local_seq_len
    return full_tensor.narrow(dim, start, local_seq_len).contiguous(), sorted_positions


def _adp_cp_redo_load_balancing(
    tensor: torch.Tensor,
    local_positions: torch.Tensor,
    sorted_positions: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    dim: int,
) -> torch.Tensor:
    full_tensor = _adp_cp_all_gather_concat(tensor, group, dim=dim, differentiable=True)
    restore_indices = torch.searchsorted(sorted_positions, local_positions)
    restored_positions = sorted_positions.index_select(0, restore_indices)
    if not torch.equal(restored_positions, local_positions):
        raise RuntimeError("Failed to restore GDN output to Megatron load-balanced CP layout.")
    return full_tensor.index_select(dim, restore_indices).contiguous()


{MCA_GDN_CP_LAYOUT_HELPERS_PATCH_MARKER}


@dataclass
class GatedDeltaNetSubmodules:
"""
MCA_GDN_CP_UNDO_LAYOUT_OLD = """        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
"""
MCA_GDN_CP_UNDO_LAYOUT_NEW = f"""        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        original_cp_positions = None
        sorted_cp_positions = None
        if cp_context is not None:
            if qkvzba.shape[1] != seq_len:
                raise RuntimeError(
                    "Unexpected GDN sequence shape after input projection: "
                    f"got {{qkvzba.shape[1]}}, expected {{seq_len}}."
                )
            original_cp_positions = _adp_cp_local_positions(seq_len, self.cp_group, qkvzba.device)
            qkvzba, sorted_cp_positions = _adp_cp_undo_load_balancing(
                qkvzba, original_cp_positions, self.cp_group, dim=1
            )
            {MCA_GDN_CP_UNDO_LAYOUT_PATCH_MARKER}

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
"""
MCA_GDN_CP_REDO_LAYOUT_OLD = """        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
"""
MCA_GDN_CP_REDO_LAYOUT_NEW = f"""        norm_out = norm_out.reshape(batch, seq_len, -1)
        if cp_context is not None:
            norm_out = _adp_cp_redo_load_balancing(
                norm_out, original_cp_positions, sorted_cp_positions, self.cp_group, dim=1
            )
            {MCA_GDN_CP_REDO_LAYOUT_PATCH_MARKER}
        norm_out = norm_out.transpose(0, 1).contiguous()
"""
MCA_GDN_CP_DETERMINISTIC_GUARD_OLD = """        if packed_seq_params is not None:
            # TODO: support packed sequence
            raise NotImplementedError("GDN does not support packed sequence for now.")

        # Input projection
"""
MCA_GDN_CP_DETERMINISTIC_GUARD_NEW = f"""        if packed_seq_params is not None:
            # TODO: support packed sequence
            raise NotImplementedError("GDN does not support packed sequence for now.")

        if cp_context is not None and self.config.deterministic_mode:
            {MCA_GDN_CP_DETERMINISTIC_GUARD_PATCH_MARKER}
            raise NotImplementedError(
                "GDN context parallelism requires FLA's CP recurrent kernel; "
                "the deterministic torch fallback does not synchronize recurrent state across CP ranks."
            )

        # Input projection
"""


def patch_file(
    module: str,
    old: str,
    new: str,
    marker: str,
    description: str,
    *,
    missing_ok: bool = False,
    old_missing_ok: bool = False,
) -> int:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError) as exc:
        if missing_ok:
            print(f"Skipping optional {description}: {module} import failed ({exc}).")
            return 0
        raise

    if spec is None or spec.origin is None:
        if missing_ok:
            print(f"Skipping optional {description}: could not find {module} on PYTHONPATH.")
            return 0
        print(f"Could not find {module} on PYTHONPATH", file=sys.stderr)
        return 1

    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if marker in text:
        print(f"Already patched {description}: {path}")
        return 0

    if old not in text:
        if old_missing_ok:
            print(f"Skipping {description}: expected prior block not found in {path}")
            return 0
        print(f"Expected {description} block not found in {path}", file=sys.stderr)
        return 1

    path.write_text(text.replace(old, new, 1))
    print(f"Patched {description}: {path}")
    return 0


def main() -> int:
    return max(
        patch_file(
            "llamafactory.train.sft.trainer",
            OLD,
            NEW,
            PATCH_MARKER,
            "SFT eval skip_logits",
        ),
        patch_file(
            "llamafactory.model.model_utils.liger_kernel",
            MOE_OLD,
            MOE_NEW,
            MOE_PATCH_MARKER,
            "Qwen3.5-MoE Liger dispatch",
        ),
        patch_file(
            "transformers.integrations.deepspeed",
            DS_MODEL_ONLY_OLD,
            DS_MODEL_ONLY_NEW,
            DS_MODEL_ONLY_PATCH_MARKER,
            "DeepSpeed HF model-only checkpoint continuation",
        ),
        patch_file(
            "transformers.trainer",
            DS_MODEL_ONLY_SCHEDULER_OLD,
            DS_MODEL_ONLY_SCHEDULER_NEW,
            DS_MODEL_ONLY_SCHEDULER_PATCH_MARKER,
            "DeepSpeed HF model-only scheduler continuation",
        ),
        patch_file(
            "llamafactory.train.sft.workflow",
            SKIP_FINAL_SAVE_OLD,
            SKIP_FINAL_SAVE_NEW,
            SKIP_FINAL_SAVE_PATCH_MARKER,
            "SFT benchmark final save skip",
        ),
        patch_file(
            "llamafactory.train.hyper_parallel.workflow",
            SKIP_FINAL_SAVE_OLD,
            SKIP_FINAL_SAVE_NEW,
            SKIP_FINAL_SAVE_PATCH_MARKER,
            "hyper-parallel SFT benchmark final save skip",
            missing_ok=True,
        ),
        patch_file(
            "llamafactory.train.sft.workflow",
            SKIP_FINAL_PLOT_OLD,
            SKIP_FINAL_PLOT_NEW,
            SKIP_FINAL_PLOT_PATCH_MARKER,
            "SFT benchmark plot_loss skip",
        ),
        patch_file(
            "llamafactory.train.hyper_parallel.workflow",
            SKIP_FINAL_PLOT_OLD,
            SKIP_FINAL_PLOT_NEW,
            SKIP_FINAL_PLOT_PATCH_MARKER,
            "hyper-parallel SFT benchmark plot_loss skip",
            missing_ok=True,
        ),
        patch_file(
            "llamafactory.train.mca.workflow",
            MCA_SKIP_FINAL_SAVE_OLD,
            MCA_SKIP_FINAL_SAVE_NEW,
            MCA_SKIP_FINAL_SAVE_PATCH_MARKER,
            "MCA SFT benchmark final save skip",
            missing_ok=True,
        ),
        patch_file(
            "llamafactory.train.mca.workflow",
            MCA_SKIP_FINAL_PLOT_OLD,
            MCA_SKIP_FINAL_PLOT_NEW,
            MCA_SKIP_FINAL_PLOT_PATCH_MARKER,
            "MCA SFT benchmark plot_loss skip",
            missing_ok=True,
        ),
        patch_file(
            "mcore_adapter.models.qwen3_5.config_qwen3_5",
            MCA_QWEN35_LINEAR_CONFIG_OLD,
            MCA_QWEN35_LINEAR_CONFIG_NEW,
            MCA_QWEN35_LINEAR_CONFIG_PATCH_MARKER,
            "MCA Qwen3.5 linear-attention config fields",
            missing_ok=True,
        ),
        patch_file(
            "llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.cuda_fused_moe",
            FUSED_MOE_LIGER_EXPERTS_OLD,
            FUSED_MOE_LIGER_EXPERTS_NEW,
            FUSED_MOE_LIGER_EXPERTS_PATCH_MARKER,
            "cuda_fused_moe LigerExperts compatibility",
        ),
        patch_file(
            "mcore_adapter.trainer.trainer",
            MCA_TP_SEQ_LENGTH_TP_ONLY_OLD,
            MCA_TP_SEQ_LENGTH_NEW,
            MCA_TP_SEQ_LENGTH_PATCH_MARKER,
            "MCA tensor/context-parallel sequence-length padding upgrade",
            missing_ok=True,
            old_missing_ok=True,
        ),
        patch_file(
            "mcore_adapter.trainer.trainer",
            MCA_TP_SEQ_LENGTH_OLD,
            MCA_TP_SEQ_LENGTH_NEW,
            MCA_TP_SEQ_LENGTH_PATCH_MARKER,
            "MCA tensor-parallel sequence-length padding",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.transformer.attention",
            MCA_QWEN35_TP_OUTPUT_GATE_OLD,
            MCA_QWEN35_TP_OUTPUT_GATE_NEW,
            MCA_QWEN35_TP_OUTPUT_GATE_PATCH_MARKER,
            "MCA Qwen3.5 TP output-gate slicing",
            missing_ok=True,
        ),
        patch_file(
            "mcore_adapter.models.model_factory",
            MCA_CONTIGUOUS_CP_BATCH_OLD,
            MCA_CONTIGUOUS_CP_BATCH_NEW,
            MCA_CONTIGUOUS_CP_BATCH_PATCH_MARKER,
            "MCA optional contiguous CP batch slicing",
            missing_ok=True,
        ),
        patch_file(
            "mcore_adapter.models.qwen3_5.modeling_qwen3_5",
            MCA_QWEN35_CONTIGUOUS_CP_IMPORT_OLD,
            MCA_QWEN35_CONTIGUOUS_CP_IMPORT_NEW,
            MCA_QWEN35_CONTIGUOUS_CP_IMPORT_PATCH_MARKER,
            "MCA Qwen3.5 optional contiguous CP import",
            missing_ok=True,
        ),
        patch_file(
            "mcore_adapter.models.qwen3_5.modeling_qwen3_5",
            MCA_QWEN35_CONTIGUOUS_CP_RANGE_OLD,
            MCA_QWEN35_CONTIGUOUS_CP_RANGE_NEW,
            MCA_QWEN35_CONTIGUOUS_CP_RANGE_PATCH_MARKER,
            "MCA Qwen3.5 optional contiguous CP input ranges",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_IMPORT_OLD,
            MCA_GDN_CP_IMPORT_NEW,
            MCA_GDN_CP_IMPORT_PATCH_MARKER,
            "MCA GDN context-parallel imports",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_DIST_IMPORT_OLD,
            MCA_GDN_CP_DIST_IMPORT_NEW,
            MCA_GDN_CP_DIST_IMPORT_PATCH_MARKER,
            "MCA GDN CP distributed imports",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_LAYOUT_HELPERS_OLD,
            MCA_GDN_CP_LAYOUT_HELPERS_NEW,
            MCA_GDN_CP_LAYOUT_HELPERS_PATCH_MARKER,
            "MCA GDN CP load-balanced layout bridge helpers",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_INIT_OLD,
            MCA_GDN_CP_INIT_NEW,
            MCA_GDN_CP_INIT_PATCH_MARKER,
            "MCA GDN context-parallel process group",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_FORWARD_OLD,
            MCA_GDN_CP_FORWARD_NEW,
            MCA_GDN_CP_FORWARD_PATCH_MARKER,
            "MCA GDN context-parallel metadata",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_UNDO_LAYOUT_OLD,
            MCA_GDN_CP_UNDO_LAYOUT_NEW,
            MCA_GDN_CP_UNDO_LAYOUT_PATCH_MARKER,
            "MCA GDN CP load-balanced to contiguous layout bridge",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_DETERMINISTIC_GUARD_OLD,
            MCA_GDN_CP_DETERMINISTIC_GUARD_NEW,
            MCA_GDN_CP_DETERMINISTIC_GUARD_PATCH_MARKER,
            "MCA GDN CP deterministic fallback guard",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_CONV_OLD,
            MCA_GDN_CP_CONV_NEW,
            MCA_GDN_CP_CONV_PATCH_MARKER,
            "MCA GDN context-parallel convolution",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_RULE_OLD,
            MCA_GDN_CP_RULE_NEW,
            MCA_GDN_CP_RULE_PATCH_MARKER,
            "MCA GDN context-parallel recurrent kernel",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.ssm.gated_delta_net",
            MCA_GDN_CP_REDO_LAYOUT_OLD,
            MCA_GDN_CP_REDO_LAYOUT_NEW,
            MCA_GDN_CP_REDO_LAYOUT_PATCH_MARKER,
            "MCA GDN CP contiguous to load-balanced layout bridge",
            missing_ok=True,
        ),
        patch_file(
            "megatron.core.transformer.transformer_config",
            MCA_GDN_CP_CONFIG_OLD,
            MCA_GDN_CP_CONFIG_NEW,
            MCA_GDN_CP_CONFIG_PATCH_MARKER,
            "MCA GDN context-parallel config assertion",
            missing_ok=True,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

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

Finally, it lets a DeepSpeed ZeRO-3 run continue from a Hugging Face-format
model-only checkpoint when the model is already loaded from that checkpoint.
This preserves Trainer's data skipping via ``resume_from_checkpoint`` even when
the previous run used ``save_only_model: true`` and therefore did not write
DeepSpeed optimizer/scheduler state.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


PATCH_MARKER = "# ADP patch: force Liger loss-only eval to skip logits."
MOE_PATCH_MARKER = "# ADP patch: enable Liger for Qwen3.5-MoE."
DS_MODEL_ONLY_PATCH_MARKER = "# ADP patch: tolerate HF model-only checkpoint for DeepSpeed resume."
DS_MODEL_ONLY_SCHEDULER_PATCH_MARKER = (
    "# ADP patch: skip missing DeepSpeed scheduler state for HF model-only checkpoint."
)
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


def patch_file(module: str, old: str, new: str, marker: str, description: str) -> int:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        print(f"Could not find {module} on PYTHONPATH", file=sys.stderr)
        return 1

    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if marker in text:
        print(f"Already patched {description}: {path}")
        return 0

    if old not in text:
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
    )


if __name__ == "__main__":
    raise SystemExit(main())

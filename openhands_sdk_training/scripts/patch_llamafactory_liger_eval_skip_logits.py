#!/usr/bin/env python3
"""Patch LLaMA-Factory SFT eval to use Liger's loss-only logits skip path.

Qwen3.5 long-context SFT runs with Liger avoid materializing full
batch x sequence x vocab logits during training. In eval, Hugging Face Trainer
sets the model to eval mode, so Liger's default Qwen3.5 forward path computes
full logits before loss. At 32k context this can OOM even when training fits.

This patch makes LLaMA-Factory pass ``skip_logits=True`` to model.forward only
for loss-only SFT eval/prediction steps. It should be paired with
``prediction_loss_only: true`` in the YAML config.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


PATCH_MARKER = "# ADP patch: force Liger loss-only eval to skip logits."
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


def main() -> int:
    spec = importlib.util.find_spec("llamafactory.train.sft.trainer")
    if spec is None or spec.origin is None:
        print("Could not find llamafactory.train.sft.trainer on PYTHONPATH", file=sys.stderr)
        return 1

    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if PATCH_MARKER in text:
        print(f"Already patched: {path}")
        return 0

    if OLD not in text:
        print(f"Expected prediction_step block not found in {path}", file=sys.stderr)
        return 1

    path.write_text(text.replace(OLD, NEW, 1))
    print(f"Patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

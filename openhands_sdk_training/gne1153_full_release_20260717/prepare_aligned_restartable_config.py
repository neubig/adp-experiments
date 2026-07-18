#!/usr/bin/env python3
"""Select only a complete four-rank ZeRO checkpoint for the aligned Babel run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def checkpoint_evidence(path: Path, world_size: int, minimum_age_seconds: int) -> dict[str, Any]:
    step = checkpoint_step(path)
    required_top = [
        "config.json",
        "model.safetensors",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "latest",
    ]
    missing = [name for name in required_top if not (path / name).is_file()]
    state = None
    try:
        state = json.loads((path / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        missing.append("readable_trainer_state")
    try:
        latest = (path / "latest").read_text().strip()
    except OSError:
        latest = None
    state_dir = path / f"global_step{step}"
    optimizer_files = [
        state_dir / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(world_size)
    ]
    model_state_files = [
        state_dir / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        for rank in range(world_size)
    ]
    rng_files = [path / f"rng_state_{rank}.pth" for rank in range(world_size)]
    all_shards = optimizer_files + model_state_files + rng_files
    missing.extend(str(item.relative_to(path)) for item in all_shards if not item.is_file())
    zero_size = [str(item.relative_to(path)) for item in all_shards if item.is_file() and item.stat().st_size == 0]
    mtimes = [item.stat().st_mtime for item in [path / name for name in required_top] + all_shards if item.is_file()]
    age_seconds = time.time() - max(mtimes) if mtimes else -1
    complete = (
        step >= 0
        and not missing
        and not zero_size
        and state is not None
        and state.get("global_step") == step
        and latest == f"global_step{step}"
        and age_seconds >= minimum_age_seconds
    )
    return {
        "path": str(path),
        "step": step,
        "complete": complete,
        "trainer_state_global_step": state.get("global_step") if state else None,
        "latest": latest,
        "world_size": world_size,
        "optimizer_shards": len([item for item in optimizer_files if item.is_file()]),
        "model_state_shards": len([item for item in model_state_files if item.is_file()]),
        "rng_state_files": len([item for item in rng_files if item.is_file()]),
        "missing": missing,
        "zero_size": zero_size,
        "youngest_file_age_seconds": age_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--minimum-age-seconds", type=int, default=120)
    args = parser.parse_args()

    config = yaml.safe_load(args.source.read_text())
    output_dir = Path(config["output_dir"])
    candidates = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step, reverse=True)
    checks = [checkpoint_evidence(path, args.world_size, args.minimum_age_seconds) for path in candidates]
    selected = next((item for item in checks if item["complete"]), None)
    config["resume_from_checkpoint"] = selected["path"] if selected else None
    config["overwrite_output_dir"] = False

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(args.destination)
    evidence = {
        "selected_checkpoint": selected,
        "candidate_checks": checks,
        "source_config": str(args.source.resolve()),
        "runtime_config": str(args.destination.resolve()),
    }
    evidence_tmp = args.evidence.with_suffix(args.evidence.suffix + f".tmp.{os.getpid()}")
    evidence_tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    evidence_tmp.replace(args.evidence)
    print(f"runtime_config={args.destination}")
    print(f"resume_from_checkpoint={selected['path'] if selected else None}")
    print(f"checkpoint_evidence={args.evidence}")


if __name__ == "__main__":
    main()

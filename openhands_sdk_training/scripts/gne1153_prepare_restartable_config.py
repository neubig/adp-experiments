#!/usr/bin/env python3
"""Create a run config that resumes only from a structurally complete checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def is_complete_checkpoint(path: Path) -> bool:
    state_path = path / "trainer_state.json"
    if not state_path.is_file() or not (path / "config.json").is_file():
        return False
    if not list(path.glob("*.safetensors")) and not (path / "model.safetensors.index.json").is_file():
        return False
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("global_step") == checkpoint_step(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.source.read_text())
    output_dir = Path(config["output_dir"])
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step, reverse=True)
    complete = next((path for path in checkpoints if is_complete_checkpoint(path)), None)
    config["resume_from_checkpoint"] = str(complete) if complete is not None else None
    config["overwrite_output_dir"] = False

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(args.destination)
    print(f"runtime_config={args.destination}")
    print(f"resume_from_checkpoint={complete}")


if __name__ == "__main__":
    main()

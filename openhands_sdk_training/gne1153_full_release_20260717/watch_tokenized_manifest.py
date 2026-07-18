#!/usr/bin/env python3
"""Record exact post-LLaMA-Factory tokenization lengths and artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenized-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait_seconds
    marker = args.tokenized_path / "dataset_dict.json"
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"tokenized dataset was not completed: {args.tokenized_path}")
        time.sleep(30)

    dataset = load_from_disk(str(args.tokenized_path))
    lengths = {name: len(split) for name, split in dataset.items()}
    manifest = json.loads(args.manifest.read_text())
    adapted_rows = int(manifest["adapted_train_rows"])
    train_rows = int(lengths["train"])
    filtered = adapted_rows - train_rows
    if filtered < 0:
        raise RuntimeError("tokenized train length exceeds adapted input length")

    files = []
    for path in sorted(p for p in args.tokenized_path.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(args.tokenized_path)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_or_tokenizer": "Qwen/Qwen3.5-4B-Base",
        "template": "qwen3_5_nothink",
        "cutoff_len": 32768,
        "tokenized_path": str(args.tokenized_path.resolve()),
        "splits": lengths,
        "adapter_train_rows": adapted_rows,
        "tokenizer_output_train_rows": train_rows,
        "tokenizer_skipped_train_rows": filtered,
        "tokenizer_skip_reasons": (
            {}
            if filtered == 0
            else {"filtered_by_llamafactory_preprocessing_or_empty_supervision": filtered}
        ),
        "files": files,
    }
    atomic_json(args.output, result)
    manifest["tokenization"] = result
    manifest["status"] = "tokenized_complete"
    atomic_json(args.manifest, manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

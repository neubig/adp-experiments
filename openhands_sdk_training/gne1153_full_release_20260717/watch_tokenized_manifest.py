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
    parser.add_argument("--training-log", type=Path, required=True)
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
    evaluation_rows = int(manifest["evaluation"]["rows"])
    evaluation_output_rows = int(lengths.get("validation", lengths.get("eval", 0)))
    train_filtered = adapted_rows - train_rows
    evaluation_filtered = evaluation_rows - evaluation_output_rows
    if train_filtered < 0 or evaluation_filtered < 0:
        raise RuntimeError("tokenized train length exceeds adapted input length")

    warning_phrase = b"Skipping this abnormal example"
    log_bytes = args.training_log.read_bytes()
    warning_count = log_bytes.count(warning_phrase)
    total_filtered = train_filtered + evaluation_filtered

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
        "tokenizer_skipped_train_rows": train_filtered,
        "tokenizer_skip_reasons": (
            {}
            if train_filtered == 0
            else {"llamafactory_abnormal_example_filter": train_filtered}
        ),
        "evaluation_input_rows": evaluation_rows,
        "evaluation_output_rows": evaluation_output_rows,
        "evaluation_skipped_rows": evaluation_filtered,
        "llamafactory_abnormal_warning": warning_phrase.decode(),
        "llamafactory_abnormal_warning_count": warning_count,
        "total_tokenizer_filtered_rows": total_filtered,
        "warning_count_matches_total_filtered": warning_count == total_filtered,
        "training_log": str(args.training_log.resolve()),
        "training_log_bytes_at_count": len(log_bytes),
        "files": files,
    }
    atomic_json(args.output, result)
    manifest["tokenization"] = result
    manifest["status"] = "tokenized_complete"
    atomic_json(args.manifest, manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

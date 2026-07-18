#!/usr/bin/env python3
"""Fail closed unless two tokenizer artifacts produce identical training IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
PROBES = (
    "plain assistant response",
    "<think>\nreasoning\n</think>\n\nanswer",
    "<tool_response>\ncommand output\n</tool_response>",
    "<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
    "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n",
    "<|im_start|>user\n<tool_response>\noutput\n</tool_response><|im_end|>\n",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifacts(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in FILES:
        path = root / name
        result[name] = (
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            if path.is_file()
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference / "tokenizer.json"
    candidate_path = args.candidate / "tokenizer.json"
    if not reference_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("both tokenizer.json files must exist")

    reference_json = json.loads(reference_path.read_text())
    candidate_json = json.loads(candidate_path.read_text())
    reference_added = {item["content"]: item["id"] for item in reference_json["added_tokens"]}
    candidate_added = {item["content"]: item["id"] for item in candidate_json["added_tokens"]}
    added_token_differences = {
        token: {"reference": reference_added.get(token), "candidate": candidate_added.get(token)}
        for token in sorted(reference_added.keys() | candidate_added.keys())
        if reference_added.get(token) != candidate_added.get(token)
    }

    reference_tokenizer = Tokenizer.from_file(str(reference_path))
    candidate_tokenizer = Tokenizer.from_file(str(candidate_path))
    probes = []
    for text in PROBES:
        reference_ids = reference_tokenizer.encode(text, add_special_tokens=False).ids
        candidate_ids = candidate_tokenizer.encode(text, add_special_tokens=False).ids
        probes.append(
            {
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "reference_ids": reference_ids,
                "candidate_ids": candidate_ids,
                "identical": reference_ids == candidate_ids,
            }
        )

    model_vocab_identical = reference_json["model"].get("vocab") == candidate_json["model"].get("vocab")
    merges_identical = reference_json["model"].get("merges") == candidate_json["model"].get("merges")
    probe_ids_identical = all(item["identical"] for item in probes)
    compatible = model_vocab_identical and merges_identical and not added_token_differences and probe_ids_identical
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": artifacts(args.reference),
        "candidate": artifacts(args.candidate),
        "model_vocab_identical": model_vocab_identical,
        "merges_identical": merges_identical,
        "added_token_differences": added_token_differences,
        "probes": probes,
        "probe_ids_identical": probe_ids_identical,
        "compatible_for_tokenized_artifact_reuse": compatible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if not compatible:
        raise SystemExit("tokenizers are not ID-compatible; retokenization is required")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a deterministic small real-loader/converter preflight on the aligned view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from datasets import load_dataset
from llamafactory.data.converter import SharegptDatasetConverter
from llamafactory.data.formatter import FunctionFormatter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


def first_last_lines(path: Path) -> list[bytes]:
    with path.open("rb") as handle:
        first = handle.readline()
        if not first:
            return []
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        offset = max(0, end - 1024 * 1024)
        while True:
            handle.seek(offset)
            tail = handle.read(end - offset)
            lines = [line + b"\n" for line in tail.rstrip(b"\r\n").splitlines()]
            if len(lines) >= 2 or offset == 0:
                break
            offset = max(0, offset - 1024 * 1024)
        last = lines[-1]
    return [first] if first.rstrip(b"\r\n") == last.rstrip(b"\r\n") else [first, last]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-manifest-name", default="alignment_manifest_v3.json")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    alignment = json.loads((root / args.alignment_manifest_name).read_text())
    dataset_info = json.loads((Path(alignment["dataset_view"]) / "dataset_info.json").read_text())
    assert manifest["observed_config_count"] == 52
    assert manifest["adapted_train_rows"] == 9_196_689
    assert alignment["status"] == "alignment_complete"

    samples: list[tuple[str, bytes]] = []
    empty = []
    for dataset in manifest["datasets"]:
        name = dataset["dataset_name"]
        rows = first_last_lines(Path(dataset_info[name]["file_name"]))
        if not rows:
            empty.append(name)
        samples.extend((name, raw) for raw in rows)
    eval_name = "adpv2_full24k_eval_500"
    samples.extend((eval_name, raw) for raw in first_last_lines(Path(dataset_info[eval_name]["file_name"])))
    assert empty == ["adpv2_full24k_37"]
    assert len(samples) == 104

    with tempfile.NamedTemporaryFile(prefix="adpv2-aligned-preflight-", suffix=".jsonl") as handle:
        for _, raw in samples:
            handle.write(raw)
        handle.flush()
        loaded = load_dataset("json", data_files=handle.name, split="train")
    assert len(loaded) == len(samples)
    assert set(loaded.column_names) == {"id", "messages", "tools"}

    attr = DatasetAttr(
        load_from="file",
        dataset_name="adpv2_aligned_preflight",
        formatting="openai",
        messages="messages",
        tools="tools",
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag="tool",
        function_tag="function_call",
        system_tag="system",
    )
    converter = SharegptDatasetConverter(attr, DataArguments(template="qwen3_5_nothink"))
    function_formatter = FunctionFormatter(slots=["{{content}}"], tool_format="qwen3_5")
    function_messages = 0
    for row in loaded:
        converted = converter(row)
        assert converted["_response"], row["id"]
        for message in converted["_response"]:
            if message["role"] == "function":
                function_formatter.apply(
                    content=message["content"], tool_call_words=("<tool_call>", "</tool_call>")
                )
                function_messages += 1

    evidence = {
        "validation": "passed",
        "sample_policy": "first and last row of every nonempty config and eval",
        "release_config_count": 52,
        "empty_config": "adpv2_full24k_37",
        "sample_rows": len(samples),
        "arrow_columns": sorted(loaded.column_names),
        "llamafactory_converter_rows": len(samples),
        "function_messages_formatted": function_messages,
        "samples": [
            {"dataset_name": name, "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in samples
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

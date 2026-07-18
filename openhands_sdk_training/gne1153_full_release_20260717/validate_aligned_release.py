#!/usr/bin/env python3
"""Validate the full aligned view before LLaMA-Factory tokenization."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROMPT_ROLES = {"user", "tool"}
RESPONSE_ROLES = {"assistant", "function_call"}
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_function_content(content: str) -> bool:
    """Validate direct JSON or one unambiguous thought/tool wrapper."""
    try:
        calls = json.loads(content)
        assert "<tool_call>" not in content and "</tool_call>" not in content
        wrapped = False
    except json.JSONDecodeError:
        assert content.count("<tool_call>") == 1
        assert content.count("</tool_call>") == 1
        match = TOOL_CALL_RE.search(content)
        assert match is not None
        calls = json.loads(match.group(1))
        wrapped = True
    if not isinstance(calls, list):
        calls = [calls]
    for call in calls:
        assert isinstance(call, dict)
        assert isinstance(call["name"], str) and call["name"]
        assert isinstance(call["arguments"], dict)
    return wrapped


def validate_file(item: tuple[str, str, int]) -> dict[str, Any]:
    dataset_name, path_text, expected_rows = item
    path = Path(path_text)
    rows = 0
    function_messages = 0
    wrapped_function_messages = 0
    digest = hashlib.sha256()
    for line_number, raw in enumerate(path.open("rb", buffering=16 * 1024 * 1024), 1):
        digest.update(raw)
        if not raw.strip():
            continue
        record = json.loads(raw)
        messages = record["messages"]
        assert isinstance(messages, list), (path, line_number)
        body = messages[1:] if messages and messages[0].get("role") == "system" else messages
        assert len(body) % 2 == 0, (path, line_number, len(body))
        for index, message in enumerate(body):
            role = message.get("role")
            expected = PROMPT_ROLES if index % 2 == 0 else RESPONSE_ROLES
            assert role in expected, (path, line_number, index, role)
            if role == "function_call":
                content = message.get("content")
                assert isinstance(content, str), (path, line_number, index)
                try:
                    wrapped = validate_function_content(content)
                except (AssertionError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid function content in {path}:{line_number} "
                        f"message {index}: {content[:500]!r}"
                    ) from exc
                function_messages += 1
                wrapped_function_messages += int(wrapped)
        rows += 1
    assert rows == expected_rows, (path, rows, expected_rows)
    return {
        "dataset_name": dataset_name,
        "path": str(path),
        "rows": rows,
        "sha256": digest.hexdigest(),
        "function_messages": function_messages,
        "wrapped_function_messages": wrapped_function_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alignment-manifest-name", default="alignment_manifest_v3.json")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    alignment_path = root / args.alignment_manifest_name
    alignment = json.loads(alignment_path.read_text())
    view = Path(alignment["dataset_view"])
    dataset_info = json.loads((view / "dataset_info.json").read_text())

    assert manifest["release"]["revision"] == "eb95b66ca30ef071d5e49495d5d780c29f9aef7b"
    assert manifest["observed_config_count"] == 52
    assert manifest["source_rows"] == 9_640_563
    assert manifest["adapted_train_rows"] == 9_196_689
    assert len(manifest["datasets"]) == 52
    assert alignment["status"] == "alignment_complete"
    assert alignment["source_config_count"] == 52
    assert alignment["source_adapted_rows"] == 9_196_689
    assert alignment["nonempty_config_count"] == 51
    assert alignment["empty_dataset_names"] == ["adpv2_full24k_37"]

    items = []
    for dataset in manifest["datasets"]:
        name = dataset["dataset_name"]
        entry = dataset_info[name]
        items.append((name, entry["file_name"], dataset["adapted_rows"]))
    assert len(items) == 52
    eval_entry = dataset_info["adpv2_full24k_eval_500"]
    items.append(("adpv2_full24k_eval_500", eval_entry["file_name"], 500))
    # Start the largest files first so validation cannot finish with one worker
    # alone on a late 100+ GB shard.
    items.sort(key=lambda item: Path(item[1]).stat().st_size, reverse=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(validate_file, items))
    training_results = [item for item in results if item["dataset_name"] != "adpv2_full24k_eval_500"]
    evaluation_result = next(
        item for item in results if item["dataset_name"] == "adpv2_full24k_eval_500"
    )
    assert sum(item["rows"] for item in training_results) == 9_196_689

    alignment_files = {item["destination"]: item for item in alignment["files"]}
    for result in results:
        aligned_record = alignment_files.get(result["path"])
        if aligned_record:
            assert result["rows"] == aligned_record["rows"]
            assert result["sha256"] == aligned_record["destination_sha256"]

    output = {
        "validation": "passed",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "alignment_manifest_sha256": sha256_file(alignment_path),
        "dataset_info_sha256": sha256_file(view / "dataset_info.json"),
        "config_count": 52,
        "nonempty_config_count": 51,
        "training_rows": sum(item["rows"] for item in training_results),
        "evaluation_rows": evaluation_result["rows"],
        "function_messages": sum(item["function_messages"] for item in results),
        "wrapped_function_messages": sum(item["wrapped_function_messages"] for item in results),
        "files": results,
        "normalization": {
            "normalized_rows": alignment["normalized_rows"],
            "merged_messages": alignment["merged_messages"],
            "source_manifest_sha256": alignment["source_manifest_sha256"],
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

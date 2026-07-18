#!/usr/bin/env python3
"""Create an aligned LLaMA-Factory view without dropping adapted records."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROMPT_ROLES = {"user", "tool"}
RESPONSE_ROLES = {"assistant", "function_call"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def combine_response_group(group: list[dict[str, str]]) -> dict[str, str]:
    if len(group) == 1:
        return dict(group[0])
    saw_function = False
    thoughts: list[str] = []
    functions: list[object] = []
    for message in group:
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            if saw_function:
                raise ValueError("assistant message follows a function_call in one response group")
            if content:
                thoughts.append(content)
        elif role == "function_call":
            saw_function = True
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                parsed = [parsed]
            functions.extend(parsed)
        else:
            raise ValueError(f"unexpected response role: {role}")
    if functions:
        function_json = json.dumps(functions, ensure_ascii=False)
        thought = "\n\n".join(thoughts)
        prefix = f"{thought}\n" if thought else ""
        return {
            "role": "function_call",
            "content": f"{prefix}<tool_call>{function_json}</tool_call>",
        }
    return {"role": "assistant", "content": "\n\n".join(thoughts)}


def normalize_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    prefix: list[dict[str, str]] = []
    body = messages
    if messages and messages[0].get("role") == "system":
        prefix = [dict(messages[0])]
        body = messages[1:]
    output = list(prefix)
    merged_messages = 0
    index = 0
    while index < len(body):
        message = body[index]
        role = message.get("role")
        if role in PROMPT_ROLES:
            output.append(dict(message))
            index += 1
            continue
        if role not in RESPONSE_ROLES:
            raise ValueError(f"unexpected role: {role}")
        end = index + 1
        while end < len(body) and body[end].get("role") in RESPONSE_ROLES:
            end += 1
        group = body[index:end]
        output.append(combine_response_group(group))
        merged_messages += len(group) - 1
        index = end
    normalized_body = output[len(prefix) :]
    for turn_index, message in enumerate(normalized_body):
        expected = PROMPT_ROLES if turn_index % 2 == 0 else RESPONSE_ROLES
        if message.get("role") not in expected:
            raise ValueError(
                f"unaligned normalized role at index {turn_index}: {message.get('role')}"
            )
    if len(normalized_body) % 2:
        raise ValueError("normalized record does not end in a response")
    return output, merged_messages


def normalize_file(source: Path, destination: Path) -> dict:
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    rows = 0
    changed_rows = 0
    merged_messages = 0
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        for line_number, raw in enumerate(input_handle, 1):
            source_digest.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            normalized, merged = normalize_messages(record["messages"])
            if merged:
                record["messages"] = normalized
                changed_rows += 1
                merged_messages += merged
            encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
            output_handle.write(encoded)
            destination_digest.update(encoded)
            rows += 1
    temporary.replace(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "rows": rows,
        "changed_rows": changed_rows,
        "merged_messages": merged_messages,
        "source_sha256": source_digest.hexdigest(),
        "destination_sha256": destination_digest.hexdigest(),
        "destination_bytes": destination.stat().st_size,
    }


def normalize_pair(pair: tuple[Path, Path]) -> dict:
    return normalize_file(*pair)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    projection = root / "training_projection"
    aligned = root / "training_projection_aligned"
    aligned.mkdir(exist_ok=False)
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["adapted_train_rows"] != 9_196_689:
        raise ValueError("unexpected adapted row count")
    dataset_info = json.loads((root / "dataset_info.json").read_text())
    targets = set(args.targets)
    files_by_name = {path.name: path for path in projection.glob("*.jsonl")}
    missing = sorted(targets - files_by_name.keys())
    if missing:
        raise FileNotFoundError(f"missing targets: {missing}")
    pairs = []
    for name in sorted(targets):
        pairs.append((files_by_name[name], aligned / name))
    eval_source = projection / "eval_500.llamafactory.jsonl"
    if eval_source.name not in targets:
        pairs.append((eval_source, aligned / eval_source.name))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(normalize_pair, pairs))

    nonempty_names = []
    for dataset in manifest["datasets"]:
        dataset_name = dataset["dataset_name"]
        basename = Path(dataset["adapted_path"]).name
        if basename in targets:
            file_path = aligned / basename
        else:
            file_path = projection / basename
        dataset_info[dataset_name]["file_name"] = str(file_path)
        if dataset["adapted_rows"]:
            nonempty_names.append(dataset_name)
    dataset_info["adpv2_full24k_eval_500"]["file_name"] = str(
        aligned / "eval_500.llamafactory.jsonl"
    )
    view = root.parent / "dataset_aligned"
    view.mkdir(exist_ok=False)
    atomic_json(view / "dataset_info.json", dataset_info)
    result = {
        "schema_version": 1,
        "status": "alignment_complete",
        "created_at": now(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_adapted_rows": manifest["adapted_train_rows"],
        "source_config_count": manifest["observed_config_count"],
        "nonempty_config_count": len(nonempty_names),
        "empty_dataset_names": [
            item["dataset_name"] for item in manifest["datasets"] if not item["adapted_rows"]
        ],
        "train_dataset_names": nonempty_names,
        "dataset_view": str(view),
        "files": results,
        "normalized_rows": sum(item["changed_rows"] for item in results),
        "merged_messages": sum(item["merged_messages"] for item in results),
    }
    atomic_json(root / "alignment_manifest.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

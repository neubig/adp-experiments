#!/usr/bin/env python3
"""Make literal tool-call markers unambiguous inside function JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


OPEN = "<tool_call>"
CLOSE = "</tool_call>"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False)
    return encoded.replace(OPEN, r"\u003ctool_call>").replace(CLOSE, r"\u003c/tool_call>")


def validate_calls(value: object) -> None:
    calls = value if isinstance(value, list) else [value]
    for call in calls:
        if not isinstance(call, dict):
            raise TypeError("function call is not an object")
        if not isinstance(call.get("name"), str) or not call["name"]:
            raise ValueError("function call has no name")
        if not isinstance(call.get("arguments"), dict):
            raise TypeError("function arguments are not an object")


def sanitize_content(content: str) -> tuple[str, bool]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        if not content.endswith(CLOSE):
            raise
        outer_close = len(content) - len(CLOSE)
        positions = []
        start = 0
        while (position := content.find(OPEN, start)) >= 0:
            positions.append(position)
            start = position + len(OPEN)
        selected = None
        for position in positions:
            candidate = content[position + len(OPEN) : outer_close]
            try:
                value = json.loads(candidate)
                validate_calls(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            selected = (position, value)
            break
        if selected is None:
            raise ValueError("cannot identify the outer tool-call wrapper")
        position, parsed = selected
        thought = content[:position].replace(OPEN, "&lt;tool_call&gt;").replace(
            CLOSE, "&lt;/tool_call&gt;"
        )
        sanitized = f"{thought}{OPEN}{safe_json(parsed)}{CLOSE}"
    else:
        validate_calls(parsed)
        sanitized = safe_json(parsed)
    validate_calls(parsed)
    assert sanitized.count(OPEN) <= 1
    assert sanitized.count(CLOSE) <= 1
    return sanitized, sanitized != content


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    path = args.path.resolve()
    backup = path.with_name(path.name + ".pre_function_marker_sanitization")
    if backup.exists():
        raise FileExistsError(backup)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    source_sha256 = digest(path)
    rows = 0
    changed_rows = 0
    changed_messages = 0
    with path.open() as input_handle, temporary.open("w") as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row_changed = False
            for message in record.get("messages") or []:
                if message.get("role") != "function_call":
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    raise TypeError("function content is not a string")
                sanitized, changed = sanitize_content(content)
                if changed:
                    message["content"] = sanitized
                    changed_messages += 1
                    row_changed = True
            changed_rows += int(row_changed)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
    os.link(path, backup)
    temporary.replace(path)
    destination_sha256 = digest(path)
    alignment = json.loads(args.alignment_manifest.read_text())
    matches = [item for item in alignment["files"] if Path(item["destination"]).resolve() == path]
    if len(matches) != 1:
        raise ValueError(f"expected one alignment file entry, found {len(matches)}")
    matches[0]["destination_sha256"] = destination_sha256
    matches[0]["destination_bytes"] = path.stat().st_size
    result = {
        "status": "function_marker_sanitization_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": str(path),
        "backup": str(backup),
        "rows": rows,
        "changed_rows": changed_rows,
        "changed_messages": changed_messages,
        "source_sha256": source_sha256,
        "destination_sha256": destination_sha256,
        "semantic_policy": (
            "JSON unicode-escape literal <tool_call> markers inside function arguments; "
            "after nested JSON decoding the function name and arguments are unchanged"
        ),
    }
    alignment["function_marker_sanitization"] = result
    atomic_json(args.alignment_manifest, alignment)
    atomic_json(args.output_manifest, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

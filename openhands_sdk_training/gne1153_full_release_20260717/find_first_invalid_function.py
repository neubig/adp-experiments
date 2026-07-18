#!/usr/bin/env python3
"""Locate invalid function content without recording conversation text."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import re
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def scan(item: tuple[str, object]) -> dict:
    path_text, stop = item
    path = Path(path_text)
    for line_number, raw in enumerate(path.open("rb"), 1):
        if line_number % 1000 == 0 and stop.is_set():
            return {"path": str(path), "cancelled_after_peer_found_invalid": True}
        if not raw.strip():
            continue
        record = json.loads(raw)
        messages = record.get("messages") or []
        for message_index, message in enumerate(messages):
            if message.get("role") != "function_call":
                continue
            content = message.get("content")
            try:
                if not isinstance(content, str):
                    raise TypeError(type(content).__name__)
                match = TOOL_CALL_RE.search(content)
                parsed = json.loads(match.group(1) if match else content)
                calls = parsed if isinstance(parsed, list) else [parsed]
                assert all(
                    isinstance(call, dict)
                    and isinstance(call.get("name"), str)
                    and call["name"]
                    and isinstance(call.get("arguments"), dict)
                    for call in calls
                )
            except (AssertionError, json.JSONDecodeError, TypeError, ValueError) as exc:
                stop.set()
                roles = [item.get("role") for item in messages if isinstance(item, dict)]
                return {
                    "path": str(path),
                    "line": line_number,
                    "record_id_sha256": hashlib.sha256(str(record.get("id")).encode()).hexdigest(),
                    "message_index": message_index,
                    "role_sequence": roles,
                    "content_sha256": hashlib.sha256(
                        content.encode() if isinstance(content, str) else repr(content).encode()
                    ).hexdigest(),
                    "content_bytes": len(content.encode()) if isinstance(content, str) else None,
                    "starts_with_machine_wrapper": isinstance(content, str) and content.startswith("<tool_call>[") ,
                    "first_tool_call_offset": content.find("<tool_call>") if isinstance(content, str) else None,
                    "selected_span_bytes": len(match.group(1).encode()) if isinstance(content, str) and match else None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return {"path": str(path), "invalid": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    args = parser.parse_args()
    alignment = json.loads(args.alignment_manifest.read_text())
    paths = [item["destination"] for item in alignment["files"]]
    with multiprocessing.Manager() as manager:
        stop = manager.Event()
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(paths)) as pool:
            results = list(pool.map(scan, [(path, stop) for path in paths]))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

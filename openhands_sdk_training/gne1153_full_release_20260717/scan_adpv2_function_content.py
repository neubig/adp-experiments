#!/usr/bin/env python3
"""Find function-call messages that LLaMA-Factory cannot parse."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def scan(item: tuple[str, str]) -> dict:
    dataset_name, path_text = item
    path = Path(path_text)
    rows = 0
    function_messages = 0
    invalid_messages = 0
    samples = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            for message_index, message in enumerate(record.get("messages") or []):
                if message.get("role") != "function_call":
                    continue
                function_messages += 1
                content = message.get("content")
                try:
                    if not isinstance(content, str):
                        raise TypeError(type(content).__name__)
                    match = TOOL_CALL_RE.search(content)
                    parsed = json.loads(match.group(1) if match else content)
                    calls = parsed if isinstance(parsed, list) else [parsed]
                    for call in calls:
                        if not isinstance(call, dict):
                            raise TypeError("call is not an object")
                        if not isinstance(call.get("name"), str) or not call["name"]:
                            raise ValueError("missing function name")
                        if not isinstance(call.get("arguments"), dict):
                            raise TypeError("arguments are not an object")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    invalid_messages += 1
                    if len(samples) < 20:
                        samples.append(
                            {
                                "line": line_number,
                                "id": record.get("id"),
                                "message_index": message_index,
                                "error": f"{type(exc).__name__}: {exc}",
                                "content_prefix": repr(content)[:500],
                            }
                        )
    return {
        "dataset_name": dataset_name,
        "path": str(path),
        "rows": rows,
        "function_messages": function_messages,
        "invalid_messages": invalid_messages,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    info = json.loads(args.dataset_info.read_text())
    items = [(name, entry["file_name"]) for name, entry in info.items()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(scan, items))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

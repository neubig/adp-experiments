#!/usr/bin/env python3
"""Find invalid or marker-ambiguous function-call messages.

LLaMA-Factory accepts either bare function JSON or thought text followed by a
single ``<tool_call>JSON</tool_call>`` wrapper.  Literal marker strings inside
otherwise-valid bare JSON are dangerous because its formatter searches for a
wrapper before attempting the bare JSON parse.
"""

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
    issue_counts: dict[str, int] = {}
    samples = []

    def issue(
        kind: str,
        *,
        line_number: int,
        record: dict,
        message_index: int,
        content: object,
        error: str | None = None,
    ) -> None:
        issue_counts[kind] = issue_counts.get(kind, 0) + 1
        if len(samples) < 50:
            sample = {
                "kind": kind,
                "line": line_number,
                "id": record.get("id"),
                "message_index": message_index,
                "open_markers": content.count("<tool_call>") if isinstance(content, str) else None,
                "close_markers": content.count("</tool_call>") if isinstance(content, str) else None,
                "content_prefix": repr(content)[:500],
            }
            if error:
                sample["error"] = error
            samples.append(sample)

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
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        open_markers = content.count("<tool_call>")
                        close_markers = content.count("</tool_call>")
                        if open_markers != 1 or close_markers != 1:
                            issue(
                                "wrapped_ambiguous_markers",
                                line_number=line_number,
                                record=record,
                                message_index=message_index,
                                content=content,
                            )
                        match = TOOL_CALL_RE.search(content)
                        if match is None:
                            raise ValueError("neither bare JSON nor a tool-call wrapper")
                        parsed = json.loads(match.group(1))
                    else:
                        if "<tool_call>" in content or "</tool_call>" in content:
                            issue(
                                "bare_json_with_literal_markers",
                                line_number=line_number,
                                record=record,
                                message_index=message_index,
                                content=content,
                            )
                    calls = parsed if isinstance(parsed, list) else [parsed]
                    for call in calls:
                        if not isinstance(call, dict):
                            raise TypeError("call is not an object")
                        if not isinstance(call.get("name"), str) or not call["name"]:
                            raise ValueError("missing function name")
                        if not isinstance(call.get("arguments"), dict):
                            raise TypeError("arguments are not an object")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    issue(
                        "invalid_function_content",
                        line_number=line_number,
                        record=record,
                        message_index=message_index,
                        content=content,
                        error=f"{type(exc).__name__}: {exc}",
                    )
    return {
        "dataset_name": dataset_name,
        "path": str(path),
        "rows": rows,
        "function_messages": function_messages,
        "issue_messages": sum(issue_counts.values()),
        "issue_counts": issue_counts,
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

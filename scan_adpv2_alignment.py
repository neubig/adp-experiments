#!/usr/bin/env python3
"""Scan projected JSONL files for LLaMA-Factory alternating-role violations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path


PROMPT = {"user", "tool"}
RESPONSE = {"assistant", "function_call"}


def scan(path: Path) -> dict:
    rows = 0
    invalid_rows = 0
    adjacent: Counter[tuple[str, str]] = Counter()
    reasons: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            messages = json.loads(line).get("messages") or []
            body = messages[1:] if messages and messages[0].get("role") == "system" else messages
            bad = False
            for index, message in enumerate(body):
                role = message.get("role")
                expected = PROMPT if index % 2 == 0 else RESPONSE
                if role not in expected:
                    reasons[f"index_{index}_{role}"] += 1
                    bad = True
                    break
            if len(body) % 2:
                reasons["odd_message_count"] += 1
                bad = True
            if bad:
                invalid_rows += 1
            for left, right in zip(body, body[1:]):
                pair = (str(left.get("role")), str(right.get("role")))
                if (pair[0] in PROMPT and pair[1] in PROMPT) or (
                    pair[0] in RESPONSE and pair[1] in RESPONSE
                ):
                    adjacent[pair] += 1
    return {
        "path": str(path),
        "rows": rows,
        "invalid_rows": invalid_rows,
        "adjacent_same_side": adjacent.most_common(),
        "reasons": reasons.most_common(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(scan, args.paths))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

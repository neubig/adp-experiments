#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from openhands.sdk import LLM
from pydantic import SecretStr


def token_count(llm: LLM, messages: list[dict[str, Any]]) -> int:
    try:
        return llm.get_token_count(messages)
    except Exception:
        return sum(len(json.dumps(message, ensure_ascii=False)) for message in messages) // 4


def turn_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get('role') != 'system')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl', type=Path)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--model', default='gpt-4o-mini')
    args = parser.parse_args()

    llm = LLM(model=args.model, api_key=SecretStr('not-used'))
    trajectories: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    with args.jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = record.get('metadata', {}).get('source_trajectory_id')
            if not source_id:
                continue
            if source_id not in trajectories:
                if len(trajectories) >= args.limit:
                    break
                trajectories[source_id] = []
            trajectories[source_id].append(record)

    header = [
        'source_trajectory_id',
        'records',
        'trajectory_segments',
        'condensations',
        'token_lengths',
        'message_counts',
        'turn_counts',
        'record_ids',
    ]
    print('\t'.join(header))

    for source_id, records in trajectories.items():
        trajectory_segments = 0
        condensations = 0
        lengths: list[str] = []
        message_counts: list[str] = []
        turn_counts: list[str] = []
        record_ids: list[str] = []
        for record in records:
            metadata = record.get('metadata', {})
            record_ids.append(record.get('id', ''))
            length = token_count(llm, record.get('messages', []))
            if metadata.get('generation') == 'openhands_sdk_condensation_prompt':
                condensations += 1
                label = f'condensation_{metadata.get("condensation_index")}'
            elif metadata.get('record_type') == 'trajectory':
                trajectory_segments += 1
                label = f'trajectory_{metadata.get("trajectory_segment_index")}'
            else:
                label = metadata.get('generation', 'unknown')
            lengths.append(f'{label}:{length}')
            messages = record.get('messages', [])
            message_counts.append(f'{label}:{len(messages)}')
            turn_counts.append(f'{label}:{turn_count(messages)}')

        print(
            '\t'.join(
                [
                    source_id,
                    str(len(records)),
                    str(trajectory_segments),
                    str(condensations),
                    ','.join(lengths),
                    ','.join(message_counts),
                    ','.join(turn_counts),
                    ','.join(record_ids),
                ]
            )
        )


if __name__ == '__main__':
    main()

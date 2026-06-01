#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agents.openhands_sdk import condensation_sft
from schema.dataset_metadata import DatasetMetadata

ORIGINAL_LOAD_METADATA = condensation_sft.load_dataset_metadata


def load_cached_metadata(
    dataset_name: str | None,
    *,
    required: bool = False,
    dataset_root: Path | None = None,
) -> DatasetMetadata:
    metadata_dir = os.getenv('ADP_CONDENSER_METADATA_DIR')
    if not dataset_name or not metadata_dir:
        return ORIGINAL_LOAD_METADATA(
            dataset_name,
            required=required,
            dataset_root=dataset_root,
        )
    metadata_path = Path(metadata_dir) / f'{dataset_name}.metadata.json'
    if metadata_path.exists():
        return DatasetMetadata(**json.loads(metadata_path.read_text()))
    return ORIGINAL_LOAD_METADATA(
        dataset_name,
        required=required,
        dataset_root=dataset_root,
    )


condensation_sft.load_dataset_metadata = load_cached_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--errors', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=12_000)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    total = 0
    failed = 0
    emitted = 0
    with_condensation = 0
    max_condensations = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open() as input_file, args.output.open('w') as output_file, args.errors.open('w') as error_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                records = process_row(
                    line,
                    max_tokens=args.max_tokens,
                    model=args.model,
                    dataset_name=args.dataset_name,
                    include_trajectories=False,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                row_id = None
                try:
                    row_id = json.loads(line).get('id')
                except Exception:  # noqa: BLE001
                    pass
                print(
                    json.dumps(
                        {
                            'line_number': line_number,
                            'id': row_id,
                            'error_type': type(exc).__name__,
                            'error': str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=error_file,
                    flush=True,
                )
                continue

            count = len(records)
            emitted += count
            if count:
                with_condensation += 1
            max_condensations = max(max_condensations, count)
            for record in records:
                print(json.dumps(record, ensure_ascii=False), file=output_file)

            if total % 25 == 0:
                print(
                    json.dumps(
                        {
                            'input': str(args.input),
                            'total': total,
                            'failed': failed,
                            'emitted': emitted,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    processed = total - failed
    stats = {
        'dataset_name': args.dataset_name,
        'input': str(args.input),
        'output': str(args.output),
        'errors': str(args.errors),
        'total_trajectories': total,
        'failed_trajectories': failed,
        'processed_trajectories': processed,
        'condensation_records': emitted,
        'trajectories_with_condensation': with_condensation,
        'max_condensations_per_trajectory': max_condensations,
        'avg_condensations_per_input_trajectory': emitted / total if total else 0.0,
        'avg_condensations_per_processed_trajectory': emitted / processed if processed else 0.0,
    }
    args.stats.write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()

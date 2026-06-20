#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def has_converter(dataset_dir: Path) -> bool:
    if not (dataset_dir / "extract_raw.py").is_file():
        return False
    if (dataset_dir / "raw_to_standardized.py").is_file():
        return True
    return (dataset_dir / "raw_to_atif.py").is_file() and (
        dataset_dir / "atif_to_std.py"
    ).is_file()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List ADP dataset adapters runnable by the experiment scripts."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="agent-data-protocol repository root.",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "json", "bash"),
        default="plain",
        help="Output format.",
    )
    args = parser.parse_args()

    datasets_root = args.repo / "datasets"
    datasets = sorted(
        path.name
        for path in datasets_root.iterdir()
        if path.is_dir() and has_converter(path)
    )

    if args.format == "json":
        print(json.dumps(datasets, indent=2))
    elif args.format == "bash":
        print("DATASETS=(")
        for dataset in datasets:
            print(f'  "{dataset}"')
        print(")")
    else:
        for dataset in datasets:
            print(dataset)


if __name__ == "__main__":
    main()

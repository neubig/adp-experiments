#!/usr/bin/env python3
"""Validate and adapt the pinned ADP-v2 full 24k OpenHands SFT release."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ID = "neulab/adp-v2"
REVISION = "eb95b66ca30ef071d5e49495d5d780c29f9aef7b"
SPLIT = "sft_openhands_24k"
EXPECTED_ROWS = {
    "AlienKevin_SWE-ZERO-12M-trajectories": 1001563,
    "CharlieDreemur_OpenManus-RL": 44647,
    "SALT-NLP_SWE-chat": 40427,
    "agenttuning_alfworld": 336,
    "agenttuning_db": 538,
    "agenttuning_kg": 324,
    "agenttuning_mind2web": 122,
    "agenttuning_os": 195,
    "agenttuning_webshop": 351,
    "allenai_Sera-4.6-Lite-T2": 120051,
    "android_in_the_wild": 5,
    "androidcontrol": 3,
    "code_feedback": 66383,
    "codeactinstruct": 7139,
    "coderforge_preview": 709580,
    "codescout": 62107,
    "cognitivekernel_pro_sft": 47271,
    "dolci_instruct_sft_tool_use": 227579,
    "eto": 6425,
    "finch_collection": 242234,
    "gair_davinci_dev": 326010,
    "go-browse-wa": 22164,
    "hybrid-gym": 8076,
    "jupyter-agent-dataset": 51429,
    "kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k": 36637,
    "litecoder-terminal-sft": 32053,
    "llava_plus": 117039,
    "logicstar_swe-star": 389530,
    "mind2web": 16125,
    "mini-coder": 82406,
    "miroverse_v0_1": 216207,
    "nebius_SWE-agent-trajectories": 137154,
    "nebius_SWE-rebench-openhands-trajectories": 181359,
    "nemotron_terminal_corpus": 45106,
    "nnetnav-live": 48333,
    "nnetnav-wa": 45277,
    "nvidia_SWE-Zero-openhands-trajectories": 956294,
    "omniact": 6789,
    "openhands": 857,
    "openresearcher": 655476,
    "openthoughts_agent_sft": 126566,
    "openthoughts_tb_dev": 36,
    "orca_agentinstruct": 1046772,
    "scale_swe_distilled": 394399,
    "screenagent": 2000,
    "swe-gym_openhands_sampled_trajectories": 12186,
    "swe-play-trajectories": 2818,
    "swe-smith": 74658,
    "synatra": 99924,
    "toolmind": 163560,
    "toucan_1_5m": 1765409,
    "webarena_successful": 634,
}
EXPECTED_TOTAL_ROWS = 9_640_563


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("adp_sft_to_llamafactory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonical adapter at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_digest(digest: "hashlib._Hash", raw: bytes) -> None:
    digest.update(raw)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "formatting": "openai",
        "columns": {"messages": "messages", "tools": "tools"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function_call",
            "system_tag": "system",
        },
        "file_name": file_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument(
        "--projection-workers",
        type=int,
        default=8,
        help="parallel workers for the byte-preserving metadata-free training projection",
    )
    args = parser.parse_args()

    if len(EXPECTED_ROWS) != 52 or sum(EXPECTED_ROWS.values()) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError("embedded pinned release table is internally inconsistent")
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing release directory: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    adapted_root = args.output_root / "adapted"
    adapted_root.mkdir()
    skips_path = args.output_root / "adapter_skips.jsonl"
    adapter = load_adapter(args.adapter)
    adapter_sha256 = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    adapter_repo = args.adapter.parents[2]

    local_dirs = {p.name for p in args.source_root.iterdir() if p.is_dir()}
    expected_names = set(EXPECTED_ROWS)
    missing = sorted(expected_names - local_dirs)
    extra = sorted(local_dirs - expected_names)
    if missing:
        raise RuntimeError(f"missing pinned release configs: {missing}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "building",
        "started_at": now(),
        "release": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "split": SPLIT,
            "expected_config_count": 52,
            "expected_source_rows": EXPECTED_TOTAL_ROWS,
            "source_of_expected_counts": (
                f"https://huggingface.co/datasets/{REPO_ID}/blob/{REVISION}/README.md"
            ),
        },
        "source_root": str(args.source_root.resolve()),
        "extra_local_directories_excluded": extra,
        "adapter": {
            "path": str(args.adapter.resolve()),
            "sha256": adapter_sha256,
            "repository": str(adapter_repo),
            "repository_commit": git_revision(adapter_repo),
            "options": ["trim_to_trainable", "skip_untrainable"],
        },
        "evaluation": {
            "policy": (
                "500 adapted records with smallest sha256(config_name + NUL + record id); "
                "evaluation records remain present in every training input"
            ),
            "requested_rows": args.eval_size,
            "removed_from_training": 0,
        },
        "datasets": [],
    }
    atomic_json(args.output_root / "manifest.json", manifest)

    total_source = 0
    total_adapted = 0
    total_skipped = 0
    eval_heap: list[tuple[int, int, str, dict[str, Any]]] = []
    eval_ordinal = 0
    dataset_info: dict[str, Any] = {}

    with skips_path.open("w", encoding="utf-8") as skips_handle:
        for index, (config_name, expected_rows) in enumerate(EXPECTED_ROWS.items()):
            source = (
                args.source_root
                / config_name
                / "24k/full_sft/full_sft_openhands_sdk_condensed_24k.jsonl"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            output = adapted_root / f"{index:02d}_{config_name}.llamafactory.jsonl"
            source_digest = hashlib.sha256()
            output_digest = hashlib.sha256()
            source_rows = 0
            adapted_rows = 0
            skip_reasons: Counter[str] = Counter()
            with source.open("rb") as input_handle, output.open("wb") as output_handle:
                for line_number, raw in enumerate(input_handle, 1):
                    update_digest(source_digest, raw)
                    if not raw.strip():
                        continue
                    source_rows += 1
                    record = json.loads(raw)
                    try:
                        converted = adapter.adapt_record(record, trim_to_trainable=True)
                    except adapter.UntrainableRecordError as exc:
                        reason = str(exc)
                        skip_reasons[reason] += 1
                        skips_handle.write(
                            json.dumps(
                                {
                                    "config": config_name,
                                    "line": line_number,
                                    "id": record.get("id"),
                                    "reason": reason,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    encoded = (json.dumps(converted, ensure_ascii=False) + "\n").encode()
                    output_handle.write(encoded)
                    update_digest(output_digest, encoded)
                    adapted_rows += 1
                    record_id = str(converted.get("id"))
                    key = int.from_bytes(
                        hashlib.sha256(
                            config_name.encode() + b"\0" + record_id.encode()
                        ).digest(),
                        "big",
                    )
                    item = (-key, -eval_ordinal, config_name, converted)
                    eval_ordinal += 1
                    if len(eval_heap) < args.eval_size:
                        heapq.heappush(eval_heap, item)
                    elif item > eval_heap[0]:
                        heapq.heapreplace(eval_heap, item)
            if source_rows != expected_rows:
                raise RuntimeError(
                    f"{config_name}: local rows {source_rows} != pinned rows {expected_rows}"
                )
            dataset_name = f"adpv2_full24k_{index:02d}"
            dataset_info[dataset_name] = dataset_info_entry(
                str(output.relative_to(args.output_root))
            )
            row = {
                "index": index,
                "config_name": config_name,
                "dataset_name": dataset_name,
                "source_path": str(source.resolve()),
                "source_bytes": source.stat().st_size,
                "source_rows_expected": expected_rows,
                "source_rows_observed": source_rows,
                "source_sha256": source_digest.hexdigest(),
                "adapted_path": str(output.resolve()),
                "adapted_bytes": output.stat().st_size,
                "adapted_rows": adapted_rows,
                "adapted_sha256": output_digest.hexdigest(),
                "skipped_rows": sum(skip_reasons.values()),
                "skip_reasons": dict(skip_reasons),
            }
            manifest["datasets"].append(row)
            total_source += source_rows
            total_adapted += adapted_rows
            total_skipped += row["skipped_rows"]
            atomic_json(args.output_root / "manifest.json", manifest)
            print(json.dumps(row), flush=True)

    if total_source != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"total local rows {total_source} != pinned {EXPECTED_TOTAL_ROWS}")
    if total_adapted + total_skipped != total_source:
        raise RuntimeError("adapted plus skipped does not equal source total")

    eval_path = args.output_root / "eval_500.llamafactory.jsonl"
    eval_digest = hashlib.sha256()
    selected = sorted(
        [
            (-negative_key, config_name, record)
            for negative_key, _, config_name, record in eval_heap
        ]
    )
    with eval_path.open("wb") as eval_handle:
        for _, _, record in selected:
            raw = (json.dumps(record, ensure_ascii=False) + "\n").encode()
            eval_handle.write(raw)
            eval_digest.update(raw)
    dataset_info["adpv2_full24k_eval_500"] = dataset_info_entry(eval_path.name)
    atomic_json(args.output_root / "dataset_info.json", dataset_info)

    manifest.update(
        {
            "status": "adapted_complete",
            "finished_at": now(),
            "observed_config_count": len(manifest["datasets"]),
            "source_rows": total_source,
            "adapted_train_rows": total_adapted,
            "adapter_skipped_rows": total_skipped,
            "adapter_skips_path": str(skips_path.resolve()),
            "dataset_info_path": str((args.output_root / "dataset_info.json").resolve()),
            "train_dataset_names": [d["dataset_name"] for d in manifest["datasets"]],
        }
    )
    manifest["evaluation"].update(
        {
            "observed_rows": len(selected),
            "path": str(eval_path.resolve()),
            "sha256": eval_digest.hexdigest(),
        }
    )
    atomic_json(args.output_root / "manifest.json", manifest)
    print(json.dumps({"manifest": str(args.output_root / "manifest.json"), **manifest}, indent=2))

    # Preserve the canonical adapted files above, then create the distinct
    # id/messages/tools-only view consumed by LLaMA-Factory. Import locally so
    # this release builder and the standalone repair command share one exact
    # invariant checker and temp+rename implementation.
    from project_training_schema import build_projection

    build_projection(args.output_root, args.projection_workers)


if __name__ == "__main__":
    main()

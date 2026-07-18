#!/usr/bin/env python3
"""Create schema-stable training JSONL without reserializing training payloads.

The canonical adapter emits top-level fields in this order: id, messages, tools,
and optional metadata.  Metadata is not consumed by LLaMA-Factory and has a
heterogeneous nested schema.  This script strips only that final metadata field
while preserving the original id/messages/tools bytes exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METADATA_MARKERS = (b', "metadata": ', b',"metadata":')
MESSAGES_MARKERS = (b', "messages": ', b',"messages":')
TOOLS_MARKERS = (b', "tools": ', b',"tools":')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def locate_unique(body: bytes, markers: tuple[bytes, ...], label: str) -> tuple[int, bytes]:
    matches: list[tuple[int, bytes]] = []
    for marker in markers:
        count = body.count(marker)
        if count > 1:
            raise RuntimeError(f"multiple unescaped {label} markers")
        if count == 1:
            matches.append((body.find(marker), marker))
    if len(matches) != 1:
        raise RuntimeError(f"expected one unescaped {label} marker, found {len(matches)}")
    return matches[0]


def project_file(source_text: str, destination_text: str, expected_sha256: str) -> dict[str, Any]:
    source = Path(source_text)
    destination = Path(destination_text)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    rows = 0
    metadata_rows = 0
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    try:
        with source.open("rb", buffering=16 * 1024 * 1024) as input_handle, temporary.open(
            "wb", buffering=16 * 1024 * 1024
        ) as output_handle:
            for line_number, raw in enumerate(input_handle, 1):
                if not raw.strip():
                    continue
                input_digest.update(raw)
                body = raw.rstrip(b"\r\n")
                if not body.startswith(b'{"id":') or not body.endswith(b"}"):
                    raise RuntimeError(f"{source}:{line_number}: unexpected canonical JSON shape")
                messages_pos, _ = locate_unique(body, MESSAGES_MARKERS, "messages")
                tools_pos, _ = locate_unique(body, TOOLS_MARKERS, "tools")
                if messages_pos >= tools_pos:
                    raise RuntimeError(f"{source}:{line_number}: unexpected field order")

                metadata_matches: list[tuple[int, bytes]] = []
                for marker in METADATA_MARKERS:
                    count = body.count(marker)
                    if count > 1:
                        raise RuntimeError(f"{source}:{line_number}: multiple metadata fields")
                    if count == 1:
                        metadata_matches.append((body.find(marker), marker))
                if len(metadata_matches) > 1:
                    raise RuntimeError(f"{source}:{line_number}: ambiguous metadata marker")
                if metadata_matches:
                    metadata_pos, marker = metadata_matches[0]
                    if metadata_pos <= tools_pos:
                        raise RuntimeError(f"{source}:{line_number}: metadata is not the final field")
                    value = body[metadata_pos + len(marker) : -1].lstrip()
                    if not value or value[:1] not in (b"{", b"[", b'"', b"n"):
                        raise RuntimeError(f"{source}:{line_number}: unexpected metadata value")
                    encoded = body[:metadata_pos] + b"}\n"
                    metadata_rows += 1
                else:
                    encoded = body + b"\n"
                output_handle.write(encoded)
                output_digest.update(encoded)
                rows += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        observed_input_sha256 = input_digest.hexdigest()
        if observed_input_sha256 != expected_sha256:
            raise RuntimeError(
                f"{source}: input sha256 {observed_input_sha256} != manifest {expected_sha256}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "rows": rows,
        "bytes": destination.stat().st_size,
        "sha256": output_digest.hexdigest(),
        "source_adapted_sha256_verified": expected_sha256,
        "metadata_rows_stripped": metadata_rows,
        "rows_without_metadata_copied": rows - metadata_rows,
        "projection_method": "byte-preserving final top-level metadata strip",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    manifest_path = root / "manifest.json"
    dataset_info_path = root / "dataset_info.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "adapted_complete":
        raise RuntimeError(f"expected adapted_complete, got {manifest['status']}")
    if len(manifest["datasets"]) != 52 or manifest["adapted_train_rows"] != 9_196_689:
        raise RuntimeError("adapted manifest identity check failed")

    projection_root = root / "training_projection"
    if projection_root.exists():
        raise FileExistsError(f"refusing to overwrite {projection_root}")
    projection_root.mkdir()
    progress_path = root / "training_projection_manifest.inprogress.json"
    projection_manifest: dict[str, Any] = {
        "status": "building",
        "started_at": now(),
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_dataset_info_sha256": sha256_file(dataset_info_path),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "workers": args.workers,
        "rule": (
            "retain every canonical-adapter row and preserve id/messages/tools bytes exactly; "
            "strip only a final top-level metadata field"
        ),
        "reason": (
            "heterogeneous metadata structs caused datasets Arrow schema inference to fail; "
            "LLaMA-Factory consumes only messages and tools"
        ),
        "datasets": [],
    }
    atomic_json(progress_path, projection_manifest)

    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for dataset in manifest["datasets"]:
            source = Path(dataset["adapted_path"])
            destination = projection_root / source.name
            future = executor.submit(
                project_file,
                str(source),
                str(destination),
                dataset["adapted_sha256"],
            )
            futures[future] = (dataset, destination)
        completed: dict[str, dict[str, Any]] = {}
        for future in as_completed(futures):
            dataset, destination = futures[future]
            result = future.result()
            if result["rows"] != dataset["adapted_rows"]:
                raise RuntimeError(f"row mismatch for {dataset['config_name']}")
            result.update(
                {
                    "config_name": dataset["config_name"],
                    "dataset_name": dataset["dataset_name"],
                    "source_adapted_path": dataset["adapted_path"],
                    "path": str(destination),
                }
            )
            completed[dataset["dataset_name"]] = result
            projection_manifest["datasets"] = [
                completed[name]
                for name in sorted(completed)
            ]
            atomic_json(progress_path, projection_manifest)
            print(json.dumps(result), flush=True)

    ordered_results = [completed[d["dataset_name"]] for d in manifest["datasets"]]
    total_rows = sum(result["rows"] for result in ordered_results)
    if total_rows != manifest["adapted_train_rows"]:
        raise RuntimeError("total projected rows do not match adapted training rows")
    dataset_info = {
        result["dataset_name"]: dataset_info_entry(
            str(Path(result["path"]).relative_to(root))
        )
        for result in ordered_results
    }

    eval_source = Path(manifest["evaluation"]["path"])
    eval_destination = projection_root / eval_source.name
    eval_result = project_file(
        str(eval_source),
        str(eval_destination),
        manifest["evaluation"]["sha256"],
    )
    if eval_result["rows"] != manifest["evaluation"]["observed_rows"]:
        raise RuntimeError("evaluation row mismatch")
    dataset_info["adpv2_full24k_eval_500"] = dataset_info_entry(
        str(eval_destination.relative_to(root))
    )

    original_dataset_info = root / "dataset_info.adapter_complete.json"
    if original_dataset_info.exists():
        raise FileExistsError(original_dataset_info)
    original_dataset_info.write_bytes(dataset_info_path.read_bytes())
    atomic_json(dataset_info_path, dataset_info)
    projection_manifest.update(
        {
            "status": "complete",
            "finished_at": now(),
            "config_count": len(ordered_results),
            "train_rows": total_rows,
            "datasets": ordered_results,
            "evaluation": {"path": str(eval_destination), **eval_result},
            "dataset_info_path": str(dataset_info_path),
            "dataset_info_sha256": sha256_file(dataset_info_path),
            "preserved_original_dataset_info_path": str(original_dataset_info),
            "preserved_original_dataset_info_sha256": sha256_file(original_dataset_info),
        }
    )
    final_projection_manifest = root / "training_projection_manifest.json"
    atomic_json(final_projection_manifest, projection_manifest)
    progress_path.unlink()

    manifest["status"] = "training_projection_complete"
    manifest["training_projection"] = {
        **projection_manifest,
        "manifest_path": str(final_projection_manifest),
        "manifest_sha256": sha256_file(final_projection_manifest),
    }
    manifest["dataset_info_path"] = str(dataset_info_path)
    manifest["dataset_info_sha256"] = sha256_file(dataset_info_path)
    atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "train_rows": total_rows}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extend a hash-proven aligned release without copying its existing files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from normalize_adpv2_projection import atomic_json, normalize_pair, now


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-alignment-manifest", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--patch-dir-name", default="training_projection_aligned_v4_patch")
    parser.add_argument("--view-dir-name", default="dataset_aligned_v4")
    parser.add_argument("--manifest-name", default="alignment_manifest_v4.json")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    release_manifest_bytes = (root / "manifest.json").read_bytes()
    release_manifest = json.loads(release_manifest_bytes)
    assert release_manifest["adapted_train_rows"] == 9_196_689
    assert release_manifest["observed_config_count"] == 52

    base_path = args.base_alignment_manifest.resolve()
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    assert base["status"] == "alignment_complete"
    assert base["schema_version"] == 3
    assert base["source_manifest_sha256"] == hashlib.sha256(release_manifest_bytes).hexdigest()
    assert base["source_adapted_rows"] == 9_196_689
    assert base["source_config_count"] == 52

    # Reuse only after every base aligned file matches its immutable v3 hash.
    for item in base["files"]:
        path = Path(item["destination"])
        observed = sha256_file(path)
        if observed != item["destination_sha256"]:
            raise ValueError(
                f"base aligned hash mismatch: {path} expected={item['destination_sha256']} "
                f"observed={observed}"
            )

    base_view = Path(base["dataset_view"])
    base_info_path = base_view / "dataset_info.json"
    base_info_bytes = base_info_path.read_bytes()
    dataset_info = json.loads(base_info_bytes)
    projection = root / "training_projection"
    files_by_name = {path.name: path for path in projection.glob("*.jsonl")}
    targets = sorted(set(args.targets))
    missing = sorted(set(targets) - files_by_name.keys())
    if missing:
        raise FileNotFoundError(f"missing targets: {missing}")
    already_aligned = {Path(item["destination"]).name for item in base["files"]}
    overlap = sorted(set(targets) & already_aligned)
    if overlap:
        raise ValueError(f"targets already present in base alignment: {overlap}")

    patch_dir = root / args.patch_dir_name
    patch_dir.mkdir(exist_ok=False)
    pairs = [(files_by_name[name], patch_dir / name) for name in targets]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        patch_results = list(pool.map(normalize_pair, pairs))

    target_set = set(targets)
    for dataset in release_manifest["datasets"]:
        basename = Path(dataset["adapted_path"]).name
        if basename in target_set:
            dataset_info[dataset["dataset_name"]]["file_name"] = str(patch_dir / basename)

    view = root.parent / args.view_dir_name
    view.mkdir(exist_ok=False)
    atomic_json(view / "dataset_info.json", dataset_info)
    output = {
        "schema_version": 4,
        "status": "alignment_complete",
        "created_at": now(),
        "source_manifest": str(root / "manifest.json"),
        "source_manifest_sha256": hashlib.sha256(release_manifest_bytes).hexdigest(),
        "source_adapted_rows": release_manifest["adapted_train_rows"],
        "source_config_count": release_manifest["observed_config_count"],
        "nonempty_config_count": base["nonempty_config_count"],
        "empty_dataset_names": base["empty_dataset_names"],
        "train_dataset_names": base["train_dataset_names"],
        "dataset_view": str(view),
        "files": base["files"] + patch_results,
        "normalized_rows": base["normalized_rows"]
        + sum(item["changed_rows"] for item in patch_results),
        "merged_messages": base["merged_messages"]
        + sum(item["merged_messages"] for item in patch_results),
        "wrapped_function_groups": base["wrapped_function_groups"]
        + sum(item["wrapped_function_groups"] for item in patch_results),
        "response_merge_encoding": base["response_merge_encoding"],
        "incremental_reuse": {
            "base_alignment_manifest": str(base_path),
            "base_alignment_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "base_dataset_info": str(base_info_path),
            "base_dataset_info_sha256": hashlib.sha256(base_info_bytes).hexdigest(),
            "base_files_reused": len(base["files"]),
            "base_file_hashes_reverified": True,
            "new_targets": targets,
        },
    }
    atomic_json(root / args.manifest_name, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

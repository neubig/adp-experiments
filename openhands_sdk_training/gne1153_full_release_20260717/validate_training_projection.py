#!/usr/bin/env python3
"""Independently validate the closed full-release training projection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REVISION = "eb95b66ca30ef071d5e49495d5d780c29f9aef7b"
SOURCE_ROWS = 9_640_563
ADAPTED_ROWS = 9_196_689


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    manifest_path = root / "manifest.json"
    projection_path = root / "training_projection_manifest.json"
    dataset_info_path = root / "dataset_info.json"
    manifest = json.loads(manifest_path.read_bytes())
    projection = json.loads(projection_path.read_bytes())
    dataset_info = json.loads(dataset_info_path.read_bytes())

    assert manifest["status"] == "training_projection_complete"
    assert manifest["release"]["revision"] == REVISION
    assert manifest["release"]["expected_config_count"] == 52
    assert manifest["observed_config_count"] == 52
    assert manifest["source_rows"] == SOURCE_ROWS
    assert manifest["adapted_train_rows"] == ADAPTED_ROWS
    assert manifest["adapter_skipped_rows"] == SOURCE_ROWS - ADAPTED_ROWS
    assert manifest["evaluation"]["observed_rows"] == 500
    assert manifest["evaluation"]["removed_from_training"] == 0
    assert len(manifest["datasets"]) == 52
    assert all(d["source_rows_expected"] == d["source_rows_observed"] for d in manifest["datasets"])
    assert sum(d["source_rows_observed"] for d in manifest["datasets"]) == SOURCE_ROWS
    assert sum(d["adapted_rows"] for d in manifest["datasets"]) == ADAPTED_ROWS
    assert sum(d["skipped_rows"] for d in manifest["datasets"]) == SOURCE_ROWS - ADAPTED_ROWS

    embedded = manifest["training_projection"]
    assert embedded["status"] == projection["status"] == "complete"
    assert embedded["manifest_sha256"] == sha256_file(projection_path)
    assert projection["config_count"] == 52
    assert projection["train_rows"] == ADAPTED_ROWS
    assert len(projection["datasets"]) == 52
    assert sum(d["rows"] for d in projection["datasets"]) == ADAPTED_ROWS
    assert projection["evaluation"]["rows"] == 500
    assert projection["dataset_info_sha256"] == sha256_file(dataset_info_path)
    assert embedded["dataset_info_sha256"] == sha256_file(dataset_info_path)

    adapted_by_name = {d["dataset_name"]: d for d in manifest["datasets"]}
    projected_by_name = {d["dataset_name"]: d for d in projection["datasets"]}
    expected_names = set(adapted_by_name)
    assert len(expected_names) == len(projected_by_name) == 52
    assert set(projected_by_name) == expected_names
    assert set(dataset_info) == expected_names | {"adpv2_full24k_eval_500"}
    for name in sorted(expected_names):
        adapted = adapted_by_name[name]
        projected = projected_by_name[name]
        path = Path(projected["path"])
        assert projected["rows"] == adapted["adapted_rows"]
        assert projected["source_adapted_sha256_verified"] == adapted["adapted_sha256"]
        assert projected["rows"] == (
            projected["metadata_rows_stripped"] + projected["rows_without_metadata_copied"]
        )
        assert path.is_file() and path.stat().st_size == projected["bytes"]
        info_path = root / dataset_info[name]["file_name"]
        assert info_path == path
        assert dataset_info[name]["columns"] == {"messages": "messages", "tools": "tools"}

    eval_path = Path(projection["evaluation"]["path"])
    assert eval_path.is_file() and eval_path.stat().st_size == projection["evaluation"]["bytes"]
    assert root / dataset_info["adpv2_full24k_eval_500"]["file_name"] == eval_path
    partials = sorted(root.glob("training_projection/**/*.tmp.*"))
    assert not partials, partials
    assert not (root / "training_projection_manifest.inprogress.json").exists()

    result = {
        "validation": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "projection_manifest_path": str(projection_path),
        "projection_manifest_sha256": sha256_file(projection_path),
        "dataset_info_path": str(dataset_info_path),
        "dataset_info_sha256": sha256_file(dataset_info_path),
        "config_count": 52,
        "source_rows": SOURCE_ROWS,
        "adapted_and_projected_train_rows": ADAPTED_ROWS,
        "adapter_skipped_rows": SOURCE_ROWS - ADAPTED_ROWS,
        "projected_metadata_rows_stripped": sum(
            d["metadata_rows_stripped"] for d in projection["datasets"]
        ),
        "eval_rows": 500,
        "eval_removed_from_training": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

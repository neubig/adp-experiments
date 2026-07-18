#!/usr/bin/env python3
"""Record validated aligned-view and loader-preflight identities in the release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--loader-preflight", type=Path, required=True)
    parser.add_argument("--builder-job-id", type=int, required=True)
    parser.add_argument("--validation-job-id", type=int, required=True)
    parser.add_argument("--loader-preflight-job-id", type=int, required=True)
    args = parser.parse_args()

    alignment = json.loads(args.alignment_manifest.read_text())
    validation = json.loads(args.validation.read_text())
    preflight = json.loads(args.loader_preflight.read_text())
    assert alignment["status"] == "alignment_complete"
    assert alignment["schema_version"] == 3
    assert alignment["source_config_count"] == 52
    assert alignment["source_adapted_rows"] == 9_196_689
    assert alignment["nonempty_config_count"] == 51
    assert alignment["empty_dataset_names"] == ["adpv2_full24k_37"]
    assert validation["validation"] == "passed"
    assert validation["config_count"] == 52
    assert validation["training_rows"] == 9_196_689
    assert validation["evaluation_rows"] == 500
    assert validation["alignment_manifest_sha256"] == sha256(args.alignment_manifest)
    assert preflight["validation"] == "passed"
    assert preflight["release_config_count"] == 52
    assert preflight["sample_rows"] == 104
    assert preflight["llamafactory_converter_rows"] == 104
    assert preflight["arrow_columns"] == ["id", "messages", "tools"]

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["observed_config_count"] == 52
    assert manifest["source_rows"] == 9_640_563
    assert manifest["adapted_train_rows"] == 9_196_689
    if "training_alignment" in manifest:
        raise RuntimeError("training alignment already recorded; refusing to overwrite")
    prior_sha = hashlib.sha256(manifest_bytes).hexdigest()
    backup = args.manifest.parent.parent / "evidence" / f"manifest.before_training_alignment.{prior_sha}.json"
    with backup.open("xb") as handle:
        handle.write(manifest_bytes)

    manifest["training_alignment"] = {
        "status": "validated",
        "alignment_schema_version": 3,
        "builder_job_id": args.builder_job_id,
        "validation_job_id": args.validation_job_id,
        "loader_preflight_job_id": args.loader_preflight_job_id,
        "alignment_manifest_path": str(args.alignment_manifest.resolve()),
        "alignment_manifest_sha256": sha256(args.alignment_manifest),
        "validation_path": str(args.validation.resolve()),
        "validation_sha256": sha256(args.validation),
        "loader_preflight_path": str(args.loader_preflight.resolve()),
        "loader_preflight_sha256": sha256(args.loader_preflight),
        "dataset_view": alignment["dataset_view"],
        "source_config_count": 52,
        "nonempty_config_count": 51,
        "training_rows": 9_196_689,
        "evaluation_rows": 500,
        "normalized_rows": alignment["normalized_rows"],
        "merged_messages": alignment["merged_messages"],
        "function_messages_validated": validation["function_messages"],
        "wrapped_function_messages_validated": validation["wrapped_function_messages"],
        "loader_preflight_rows": 104,
        "prior_manifest_sha256": prior_sha,
        "prior_manifest_backup": str(backup.resolve()),
    }
    atomic_json(args.manifest, manifest)
    print(f"prior_manifest_sha256={prior_sha}")
    print(f"updated_manifest_sha256={sha256(args.manifest)}")


if __name__ == "__main__":
    main()

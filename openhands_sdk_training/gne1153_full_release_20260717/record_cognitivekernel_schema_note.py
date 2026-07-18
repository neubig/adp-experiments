#!/usr/bin/env python3
"""Record concise cognitivekernel schema facts from the immutable validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_EVIDENCE_SHA256 = "74d86c23ded124ed603ccc7c15a82e9fa5037a8ba6c2fc4a00b3368d875f426c"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    evidence_bytes = args.evidence.read_bytes()
    assert sha256(evidence_bytes) == EXPECTED_EVIDENCE_SHA256
    evidence = json.loads(evidence_bytes)
    assert evidence["validation"] == "passed"
    assert evidence["counts"] == {
        "adapted_rows": 87,
        "adapted_without_independent_response_indicator": 0,
        "skipped_untrainable_rows": 47184,
        "skipped_with_independent_response_indicator": 0,
        "source_rows": 47271,
    }
    distribution = evidence["source_distribution"]
    assert distribution["role_sequences"] == [
        {"rows": 47184, "value": ["system", "user"]},
        {"rows": 87, "value": ["system", "user", "assistant"]},
    ]
    assert distribution["top_level_schemas"] == [
        {"rows": 47271, "value": ["id", "messages", "metadata", "tools"]}
    ]
    assert distribution["message_schemas"] == [
        {"rows": 47271, "value": ["system", ["content", "role"], "str", False]},
        {"rows": 47271, "value": ["user", ["content", "role"], "str", False]},
        {"rows": 87, "value": ["assistant", ["content", "role"], "str", False]},
    ]

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    note = manifest["validations"]["cognitivekernel_pro_sft"]
    assert note["validation"] == "passed"
    assert note["evidence_sha256"] == EXPECTED_EVIDENCE_SHA256
    if "schema_evidence" in note:
        raise RuntimeError("schema evidence already recorded; refusing to overwrite")

    prior_sha = sha256(manifest_bytes)
    backup = args.evidence.parent / f"manifest.before_cognitivekernel_schema_note.{prior_sha}.json"
    with backup.open("xb") as handle:
        handle.write(manifest_bytes)
    note["schema_evidence"] = {
        "source_rows": 47271,
        "all_rows_have_roles": ["system", "user"],
        "rows_also_having_assistant": 87,
        "rows_having_function_response": 0,
        "rows_having_tool_calls": 0,
        "trainable_flags_present": False,
        "top_level_keys": ["id", "messages", "metadata", "tools"],
        "message_keys": ["content", "role"],
        "adapter_rule": (
            "trainable_prefix raises UntrainableRecordError when no assistant/function "
            "response exists"
        ),
        "expected_outcome": {"retained": 87, "skipped": 47184},
        "evidence_sha256": EXPECTED_EVIDENCE_SHA256,
        "prior_manifest_sha256": prior_sha,
        "prior_manifest_backup": str(backup.resolve()),
    }
    atomic_json(args.manifest, manifest)
    print(f"prior_manifest_sha256={prior_sha}")
    print(f"updated_manifest_sha256={sha256(args.manifest.read_bytes())}")


if __name__ == "__main__":
    main()

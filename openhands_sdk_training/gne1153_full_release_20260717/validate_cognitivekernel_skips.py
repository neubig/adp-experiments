#!/usr/bin/env python3
"""Validate the unusually high cognitivekernel canonical-adapter skip rate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_NAME = "cognitivekernel_pro_sft"
EXPECTED_SOURCE_ROWS = 47_271
EXPECTED_ADAPTED_ROWS = 87
EXPECTED_SKIPPED_ROWS = 47_184
EXPECTED_REASON = "record contains no trainable assistant/function response"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("cognitivekernel_adapter_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def role_sequence(messages: Any) -> tuple[str, ...]:
    if not isinstance(messages, list):
        return ("<non-list>",)
    return tuple(
        str(message.get("role", "<missing>")) if isinstance(message, dict) else "<non-object>"
        for message in messages
    )


def response_indicators(messages: Any) -> dict[str, int]:
    assistant_roles = 0
    function_call_roles = 0
    tool_call_messages = 0
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            assistant_roles += message.get("role") == "assistant"
            function_call_roles += message.get("role") == "function_call"
            tool_call_messages += bool(message.get("tool_calls"))
    return {
        "assistant_role_messages": assistant_roles,
        "function_call_role_messages": function_call_roles,
        "messages_with_tool_calls": tool_call_messages,
    }


def safe_sample(
    record: dict[str, Any], raw: bytes, line_number: int, outcome: str, reason: str | None
) -> dict[str, Any]:
    messages = record.get("messages")
    indicators = response_indicators(messages)
    message_schemas = []
    content_lengths = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                message_schemas.append(sorted(message))
                content = message.get("content")
                content_lengths.append(len(content) if isinstance(content, str) else None)
            else:
                message_schemas.append(["<non-object>"])
                content_lengths.append(None)
    return {
        "line": line_number,
        "id": str(record.get("id")),
        "source_record_sha256": sha256_bytes(raw),
        "top_level_keys": sorted(record),
        "role_sequence": role_sequence(messages),
        "message_schemas": message_schemas,
        "content_lengths": content_lengths,
        "tools_type": type(record.get("tools")).__name__,
        "metadata_present": "metadata" in record,
        **indicators,
        "adapter_outcome": outcome,
        "adapter_reason": reason,
    }


def deterministic_samples(items: list[tuple[str, dict[str, Any]]], count: int = 5) -> list[dict[str, Any]]:
    return [sample for _, sample in sorted(items)[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--record-manifest-note", action="store_true")
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    dataset = next(d for d in manifest["datasets"] if d["config_name"] == CONFIG_NAME)
    assert dataset["source_rows_observed"] == EXPECTED_SOURCE_ROWS
    assert dataset["adapted_rows"] == EXPECTED_ADAPTED_ROWS
    assert dataset["skipped_rows"] == EXPECTED_SKIPPED_ROWS
    assert dataset["skip_reasons"] == {EXPECTED_REASON: EXPECTED_SKIPPED_ROWS}
    source_path = Path(dataset["source_path"])
    adapter_path = Path(manifest["adapter"]["path"])
    assert sha256_file(source_path) == dataset["source_sha256"]
    assert sha256_file(adapter_path) == manifest["adapter"]["sha256"]
    adapter = load_adapter(adapter_path)

    source_rows = 0
    adapted_rows = 0
    skipped_rows = 0
    errors: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    top_level_schemas: Counter[tuple[str, ...]] = Counter()
    role_sequences: Counter[tuple[str, ...]] = Counter()
    terminal_roles: Counter[str] = Counter()
    message_schemas: Counter[tuple[str, tuple[str, ...], str, bool]] = Counter()
    response_presence: Counter[str] = Counter()
    skipped_with_response_indicator = 0
    kept_without_response_indicator = 0
    skipped_samples: list[tuple[str, dict[str, Any]]] = []
    kept_samples: list[tuple[str, dict[str, Any]]] = []

    with source_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            source_rows += 1
            record = json.loads(raw)
            messages = record.get("messages")
            sequence = role_sequence(messages)
            top_level_schemas[tuple(sorted(record))] += 1
            role_sequences[sequence] += 1
            terminal_roles[sequence[-1] if sequence else "<empty>"] += 1
            indicators = response_indicators(messages)
            has_response_indicator = any(indicators.values())
            response_presence["has_response_indicator" if has_response_indicator else "no_response_indicator"] += 1
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        message_schemas[("<non-object>", (), type(message).__name__, False)] += 1
                        continue
                    content = message.get("content")
                    message_schemas[
                        (
                            str(message.get("role", "<missing>")),
                            tuple(sorted(message)),
                            type(content).__name__,
                            bool(message.get("tool_calls")),
                        )
                    ] += 1
            try:
                adapter.adapt_record(record, trim_to_trainable=True)
            except adapter.UntrainableRecordError as exc:
                outcome = "skipped_untrainable"
                reason = str(exc)
                skipped_rows += 1
                skip_reasons[reason] += 1
                skipped_with_response_indicator += has_response_indicator
                bucket = skipped_samples
            except Exception as exc:  # Builder would have failed on any such row.
                errors[f"{type(exc).__name__}: {exc}"] += 1
                continue
            else:
                outcome = "adapted"
                reason = None
                adapted_rows += 1
                kept_without_response_indicator += not has_response_indicator
                bucket = kept_samples
            sample = safe_sample(record, raw, line_number, outcome, reason)
            key = hashlib.sha256(
                CONFIG_NAME.encode() + b"\0" + str(record.get("id")).encode() + b"\0" + str(line_number).encode()
            ).hexdigest()
            bucket.append((key, sample))

    assert not errors, errors
    assert source_rows == EXPECTED_SOURCE_ROWS
    assert adapted_rows == EXPECTED_ADAPTED_ROWS
    assert skipped_rows == EXPECTED_SKIPPED_ROWS
    assert skip_reasons == Counter({EXPECTED_REASON: EXPECTED_SKIPPED_ROWS})
    assert skipped_with_response_indicator == 0
    assert kept_without_response_indicator == 0
    assert response_presence == Counter(
        {"no_response_indicator": EXPECTED_SKIPPED_ROWS, "has_response_indicator": EXPECTED_ADAPTED_ROWS}
    )

    def common(counter: Counter[Any], limit: int = 20) -> list[dict[str, Any]]:
        return [{"value": value, "rows": rows} for value, rows in counter.most_common(limit)]

    result = {
        "validation": "passed",
        "validated_at": now(),
        "config_name": CONFIG_NAME,
        "source_path": str(source_path),
        "source_sha256": dataset["source_sha256"],
        "canonical_adapter_path": str(adapter_path),
        "canonical_adapter_sha256": manifest["adapter"]["sha256"],
        "canonical_adapter_repository_commit": manifest["adapter"]["repository_commit"],
        "adapter_options": manifest["adapter"]["options"],
        "counts": {
            "source_rows": source_rows,
            "adapted_rows": adapted_rows,
            "skipped_untrainable_rows": skipped_rows,
            "skipped_with_independent_response_indicator": skipped_with_response_indicator,
            "adapted_without_independent_response_indicator": kept_without_response_indicator,
        },
        "adapter_skip_reasons": dict(skip_reasons),
        "independent_response_test": (
            "response indicator is any source assistant/function_call role or any message with tool_calls"
        ),
        "source_distribution": {
            "response_presence": dict(response_presence),
            "terminal_roles": dict(terminal_roles),
            "top_level_schemas": common(top_level_schemas),
            "role_sequences": common(role_sequences),
            "message_schemas": common(message_schemas),
        },
        "deterministic_sample_policy": (
            "five smallest sha256(config + NUL + record id + NUL + source line) per adapter outcome; "
            "conversation content omitted"
        ),
        "deterministic_samples": {
            "skipped_untrainable": deterministic_samples(skipped_samples),
            "adapted": deterministic_samples(kept_samples),
        },
        "conclusion": (
            "Canonical adapter behavior is correct: all 47,184 skipped rows lack any source "
            "assistant/function response indicator, while all 87 retained rows contain one; "
            "zero skipped rows are response-bearing bug candidates."
        ),
        "input_manifest_sha256": sha256_bytes(manifest_bytes),
    }

    if args.evidence_output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.evidence_output}")
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.evidence_output, result)
    evidence_sha256 = sha256_file(args.evidence_output)
    print(json.dumps({**result, "evidence_path": str(args.evidence_output), "evidence_sha256": evidence_sha256}, indent=2))

    if args.record_manifest_note:
        prior_sha256 = sha256_bytes(manifest_bytes)
        backup = args.evidence_output.parent / f"manifest.before_cognitivekernel_validation.{prior_sha256}.json"
        if not backup.exists():
            with backup.open("xb") as handle:
                handle.write(manifest_bytes)
        elif backup.read_bytes() != manifest_bytes:
            raise RuntimeError(f"existing manifest backup differs: {backup}")
        validations = manifest.setdefault("validations", {})
        if CONFIG_NAME in validations:
            raise RuntimeError(f"manifest validation note already exists for {CONFIG_NAME}")
        validations[CONFIG_NAME] = {
            "validation": "passed",
            "validated_at": result["validated_at"],
            "evidence_path": str(args.evidence_output.resolve()),
            "evidence_sha256": evidence_sha256,
            "source_rows": source_rows,
            "adapted_rows": adapted_rows,
            "skipped_untrainable_rows": skipped_rows,
            "skip_reason": EXPECTED_REASON,
            "skipped_with_independent_response_indicator": skipped_with_response_indicator,
            "conclusion": result["conclusion"],
            "prior_manifest_sha256": prior_sha256,
            "prior_manifest_backup": str(backup.resolve()),
        }
        atomic_json(args.manifest, manifest)
        print(f"updated_manifest_sha256={sha256_file(args.manifest)}")


if __name__ == "__main__":
    main()

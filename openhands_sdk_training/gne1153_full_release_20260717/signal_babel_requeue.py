#!/usr/bin/env python3
"""Deliberately issue one guarded USR1 after independently proven Babel gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing a second signal; evidence already exists: {args.output}")
    snapshot_bytes = args.snapshot.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    assert snapshot["job_id"] == args.job_id
    assert snapshot["flags"]["job_running"] is True
    assert snapshot["pre_requeue_gate"] is True
    assert snapshot["slurm_restart_count"] == 0
    complete = [item for item in snapshot["checkpoints"] if item["complete"]]
    assert complete
    selected = max(complete, key=lambda item: item["step"])
    assert selected["trainer_state_global_step"] == selected["step"]
    assert not selected["missing"] and not selected["zero_size"]

    command = ["scancel", "--batch", "--signal=USR1", args.job_id]
    record = {
        "requested_at": now(),
        "job_id": args.job_id,
        "signal": "USR1",
        "target": "batch shell",
        "selected_complete_checkpoint": selected,
        "snapshot_path": str(args.snapshot.resolve()),
        "pre_signal_snapshot": snapshot,
        "command": command,
        "status": "requesting",
    }
    atomic_json(args.output, record)
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    record.update(
        {
            "completed_at": now(),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "status": "sent" if result.returncode == 0 else "failed",
        }
    )
    atomic_json(args.output, record)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

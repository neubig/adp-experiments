#!/usr/bin/env python3
"""Record replacement of a dependency-cancelled training monitor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str) -> str:
    return subprocess.run(args, check=False, text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--cancelled-monitor-job-id", required=True)
    parser.add_argument("--replacement-monitor-job-id", required=True)
    args = parser.parse_args()
    value = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "training_job_id": int(args.training_job_id),
        "cancelled_monitor_job_id": int(args.cancelled_monitor_job_id),
        "replacement_monitor_job_id": int(args.replacement_monitor_job_id),
        "training_scontrol": command("scontrol", "show", "job", args.training_job_id, "-o"),
        "cancelled_monitor_sacct": command(
            "sacct", "-j", args.cancelled_monitor_job_id, "-X", "-n", "-P",
            "--format=JobIDRaw,State,ExitCode,Elapsed"
        ),
        "replacement_monitor_scontrol": command(
            "scontrol", "show", "job", args.replacement_monitor_job_id, "-o"
        ),
        "reason": "the original after:<training-id> monitor was cancelled by Slurm before execution; replacement was submitted only after training was RUNNING",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

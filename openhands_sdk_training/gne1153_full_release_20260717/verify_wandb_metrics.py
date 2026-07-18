#!/usr/bin/env python3
"""Capture authoritative W&B run identity and synced loss metrics without secrets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = wandb.Api(timeout=60).run(f"{args.entity}/{args.project}/{args.run_id}")
    latest_loss = None
    rows_with_loss = 0
    for loss_key in ("train/loss", "loss"):
        for row in run.scan_history(keys=["_step", loss_key]):
            if row.get(loss_key) is None:
                continue
            rows_with_loss += 1
            candidate = {key: json_safe(value) for key, value in row.items()}
            if latest_loss is None or candidate.get("_step", -1) >= latest_loss.get(
                "_step", -1
            ):
                latest_loss = candidate
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
        "state": run.state,
        "rows_with_synced_loss": rows_with_loss,
        "latest_synced_loss_row": latest_loss,
        "summary_step": run.summary.get("_step"),
        "summary_runtime": run.summary.get("_runtime"),
    }
    assert run.entity == args.entity
    assert run.project == args.project
    assert run.id == args.run_id
    assert rows_with_loss > 0 and latest_loss is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

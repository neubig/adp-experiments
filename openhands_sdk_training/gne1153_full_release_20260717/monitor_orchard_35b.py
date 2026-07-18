#!/usr/bin/env python3
"""Observe the Orchard 35B job until all full-release acceptance gates pass."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOSS_RE = re.compile(r"(?:['\"]loss['\"]\s*:|\bloss\s*=)\s*['\"]?([0-9.eE+-]+)")
WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s]+")
JOB_NAME = "adpv2full24k-alignedv4-q35-35b-orchard-1ep"
WANDB_ENTITY = "gneubig"
WANDB_PROJECT = "adp-experiments"
WANDB_RUN_ID = "adpv2-full24k-aligned-v4-qwen35-35b-orchard-1ep-eb95b66-20260717"
MODEL_DIR = "/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(*args: str) -> tuple[int, str, str]:
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def tail_text(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace").replace("\r", "\n")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_summary(evidence_dir: Path, job_id: str) -> dict[str, Any]:
    max_util = 0.0
    max_memory = 0.0
    samples = 0
    files = sorted(evidence_dir.glob(f"gpu_orchard_{job_id}_restart_*.log"))
    for path in files:
        for row in csv.reader(path.read_text(errors="replace").splitlines()):
            if len(row) != 6 or not row[0].strip().isdigit():
                continue
            try:
                max_memory = max(max_memory, float(row[2]))
                max_util = max(max_util, float(row[4]))
                samples += 1
            except ValueError:
                continue
    return {
        "files": [str(path) for path in files],
        "samples": samples,
        "max_memory_used_mib": max_memory,
        "max_utilization_percent": max_util,
    }


def valid_tokenization(value: dict[str, Any] | None, tokenized_path: Path) -> bool:
    return bool(
        value is not None
        and value.get("model_or_tokenizer") == MODEL_DIR
        and value.get("splits") == {"train": 9_196_689, "validation": 500}
        and value.get("adapter_train_rows") == 9_196_689
        and value.get("tokenizer_output_train_rows") == 9_196_689
        and value.get("tokenizer_skipped_train_rows") == 0
        and value.get("evaluation_input_rows") == 500
        and value.get("evaluation_output_rows") == 500
        and value.get("evaluation_skipped_rows") == 0
        and value.get("total_tokenizer_filtered_rows") == 0
        and value.get("llamafactory_abnormal_warning_count") == 0
        and value.get("warning_count_matches_total_filtered") is True
        and value.get("tokenized_path") == str(tokenized_path.resolve())
    )


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    combined = tail_text(args.slurm_log_dir / f"{JOB_NAME}-{args.job_id}.out") + "\n" + tail_text(
        args.slurm_log_dir / f"{JOB_NAME}-{args.job_id}.err"
    )
    loss_lines = [line[-2000:] for line in combined.splitlines() if LOSS_RE.search(line)]
    wandb_urls = sorted(set(WANDB_URL_RE.findall(combined)))
    tokenization = json.loads(args.tokenized_manifest.read_text()) if args.tokenized_manifest.is_file() else None
    wandb = json.loads(args.wandb_evidence.read_text()) if args.wandb_evidence.is_file() else None
    gpu = gpu_summary(args.evidence_dir, args.job_id)
    _, squeue, _ = command("squeue", "-h", "-j", args.job_id, "-o", "%T|%M|%N|%R")
    _, sacct, _ = command(
        "sacct", "-j", args.job_id, "-X", "--format=State,ExitCode,Elapsed,NodeList", "-n", "-P"
    )
    flags = {
        "job_running": squeue.startswith("RUNNING|"),
        "tokenized_complete": valid_tokenization(tokenization, args.tokenized_path),
        "real_loss_seen": bool(loss_lines),
        "gpu_active_seen": gpu["max_utilization_percent"] > 0 and gpu["max_memory_used_mib"] > 1000,
        "wandb_url_seen": bool(wandb_urls),
        "wandb_server_synced_loss": bool(
            wandb
            and wandb.get("entity") == WANDB_ENTITY
            and wandb.get("project") == WANDB_PROJECT
            and wandb.get("run_id") == WANDB_RUN_ID
            and wandb.get("rows_with_synced_loss", 0) > 0
        ),
    }
    return {
        "observed_at": now(),
        "job_id": args.job_id,
        "squeue": squeue,
        "sacct": sacct,
        "flags": flags,
        "accepted": all(flags.values()),
        "loss_lines_tail": loss_lines[-20:],
        "wandb_urls": wandb_urls,
        "wandb": wandb,
        "gpu": gpu,
        "tokenization": tokenization,
    }


def verify_wandb(args: argparse.Namespace) -> dict[str, Any]:
    rc, stdout, stderr = command(
        sys.executable,
        str(args.wandb_verifier),
        "--entity",
        WANDB_ENTITY,
        "--project",
        WANDB_PROJECT,
        "--run-id",
        WANDB_RUN_ID,
        "--output",
        str(args.wandb_evidence),
    )
    return {
        "checked_at": now(),
        "returncode": rc,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--slurm-log-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--tokenized-manifest", type=Path, required=True)
    parser.add_argument("--tokenized-path", type=Path, required=True)
    parser.add_argument("--wandb-verifier", type=Path, required=True)
    parser.add_argument("--wandb-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=715)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = args.output.with_suffix(".history.jsonl")
    verifier_history = args.output.with_suffix(".wandb_verifier.history.jsonl")
    last_wandb_attempt = 0.0
    for _ in range(args.iterations):
        value = snapshot(args)
        if (
            value["flags"]["real_loss_seen"]
            and value["flags"]["wandb_url_seen"]
            and not value["flags"]["wandb_server_synced_loss"]
            and time.monotonic() - last_wandb_attempt >= 300
        ):
            last_wandb_attempt = time.monotonic()
            verification = verify_wandb(args)
            with verifier_history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(verification, sort_keys=True) + "\n")
            if verification["returncode"] == 0:
                value = snapshot(args)
        atomic_json(args.output, value)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
        print(json.dumps({"observed_at": value["observed_at"], "flags": value["flags"]}), flush=True)
        if value["accepted"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

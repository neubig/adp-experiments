#!/usr/bin/env python3
"""Persist read-only Babel launch evidence independently of Agent Canvas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOSS_RE = re.compile(r"(?:['\"]loss['\"]\s*:|\bloss\s*=)\s*['\"]?([0-9.eE+-]+)")
STEP_RE = re.compile(r"(?:['\"](?:step|global_step)['\"]\s*:|\bstep\s*=)\s*([0-9]+)")
WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s]+")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(*args: str) -> str:
    return subprocess.run(args, check=False, text=True, capture_output=True).stdout.strip()


def tail_text(path: Path, limit: int = 16 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace").replace("\r", "\n")


def complete_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(output_dir.glob("checkpoint-*")):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
            state = json.loads((path / "trainer_state.json").read_text())
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        model_present = (path / "model.safetensors.index.json").is_file() or bool(
            list(path.glob("*.safetensors"))
        )
        complete = (
            state.get("global_step") == step
            and (path / "config.json").is_file()
            and model_present
        )
        result.append(
            {
                "path": str(path),
                "step": step,
                "trainer_state_global_step": state.get("global_step"),
                "complete": complete,
            }
        )
    return result


def gpu_summary(evidence_dir: Path, job_id: str) -> dict[str, Any]:
    max_util = 0.0
    max_memory = 0.0
    samples = 0
    files = sorted(evidence_dir.glob(f"gpu_{job_id}_restart_*.csv"))
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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def snapshot(job_id: str, run_root: Path) -> dict[str, Any]:
    evidence = run_root / "evidence"
    output_log = evidence / f"adpv2full24k-q35-4b-babel-1ep-{job_id}.out"
    error_log = evidence / f"adpv2full24k-q35-4b-babel-1ep-{job_id}.err"
    combined = tail_text(output_log) + "\n" + tail_text(error_log)
    loss_lines = [line[-2000:] for line in combined.splitlines() if LOSS_RE.search(line)]
    step_values = [int(match.group(1)) for match in STEP_RE.finditer(combined)]
    wandb_urls = sorted(set(WANDB_URL_RE.findall(combined)))
    tokenized_manifest_path = run_root / "dataset/tokenized_manifest.json"
    tokenized = None
    if tokenized_manifest_path.is_file():
        tokenized = json.loads(tokenized_manifest_path.read_text())
    release_manifest = json.loads((run_root / "dataset/manifest.json").read_text())
    checkpoints = complete_checkpoints(run_root / "output_qwen35_4b_base_full_sft_one_epoch")
    requeue_markers = [
        line for line in combined.splitlines() if "requeue_signal_received_at=" in line
    ]
    restart_markers = [line for line in combined.splitlines() if line.startswith("restart_count=")]
    gpu = gpu_summary(evidence, job_id)
    squeue = command("squeue", "-h", "-j", job_id, "-o", "%T|%M|%N|%R")
    sacct = command(
        "sacct", "-j", job_id, "-X", "--format=State,ExitCode,Elapsed,NodeList", "-n", "-P"
    )
    flags = {
        "job_running": squeue.startswith("RUNNING|"),
        "release_manifest_full52": (
            release_manifest.get("observed_config_count") == 52
            and release_manifest.get("source_rows") == 9_640_563
        ),
        "tokenized_manifest_complete": tokenized is not None,
        "real_loss_seen": bool(loss_lines),
        "gpu_active_seen": gpu["max_utilization_percent"] > 0 and gpu["max_memory_used_mib"] > 1000,
        "wandb_url_seen": bool(wandb_urls),
        "requeue_seen": bool(requeue_markers),
        "restart_count_seen": len(set(restart_markers)) > 1,
        "complete_checkpoint_seen": any(item["complete"] for item in checkpoints),
    }
    return {
        "observed_at": now(),
        "job_id": job_id,
        "squeue": squeue,
        "sacct": sacct,
        "release_manifest": {
            "status": release_manifest.get("status"),
            "revision": release_manifest.get("release", {}).get("revision"),
            "config_count": release_manifest.get("observed_config_count"),
            "source_rows": release_manifest.get("source_rows"),
            "adapted_train_rows": release_manifest.get("adapted_train_rows"),
        },
        "tokenization": tokenized,
        "loss_lines_tail": loss_lines[-20:],
        "max_step_in_log_tail": max(step_values, default=None),
        "wandb_urls": wandb_urls,
        "expected_wandb_run_id": "adpv2-full24k-qwen35-4b-babel-1ep-eb95b66-20260717",
        "gpu": gpu,
        "checkpoints": checkpoints,
        "requeue_markers": requeue_markers[-10:],
        "restart_markers": restart_markers[-10:],
        "flags": flags,
        "stage1_pre_requeue_gate": all(
            flags[key]
            for key in (
                "job_running",
                "release_manifest_full52",
                "tokenized_manifest_complete",
                "real_loss_seen",
                "gpu_active_seen",
                "wandb_url_seen",
            )
        ),
        "restart_verification_gate": all(
            flags[key]
            for key in (
                "requeue_seen",
                "restart_count_seen",
                "complete_checkpoint_seen",
                "real_loss_seen",
                "wandb_url_seen",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=715)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = args.output.with_suffix(".history.jsonl")
    for _ in range(args.iterations):
        value = snapshot(args.job_id, args.run_root)
        atomic_json(args.output, value)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
        print(json.dumps({"observed_at": value["observed_at"], "flags": value["flags"]}), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

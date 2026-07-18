#!/usr/bin/env python3
"""Persist aligned Babel acceptance evidence without changing the training job."""

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

from prepare_aligned_restartable_config import checkpoint_evidence, checkpoint_step


LOSS_RE = re.compile(r"(?:['\"]loss['\"]\s*:|\bloss\s*=)\s*['\"]?([0-9.eE+-]+)")
PROGRESS_RE = re.compile(r"(?<!\d)(\d{1,9})/(\d{1,9})(?!\d)")
WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s]+")
EXPECTED_WANDB_ID = "adpv2-full24k-aligned-v4-qwen35-4b-babel-1ep-eb95b66-20260717"
EXPECTED_WANDB_RUN_URL = (
    "https://wandb.ai/gneubig/adp-experiments/runs/" + EXPECTED_WANDB_ID
)


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


def checkpoint_checks(output_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step, reverse=True)
    return [checkpoint_evidence(path, 4, 180) for path in paths]


def restart_count(job_id: str) -> int | None:
    rc, stdout, _ = command("scontrol", "show", "job", job_id, "-o")
    if rc:
        return None
    match = re.search(r"\bRestarts=(\d+)", stdout)
    return int(match.group(1)) if match else None


def snapshot(job_id: str, run_root: Path) -> dict[str, Any]:
    evidence_dir = run_root / "evidence"
    prefix = f"adpv2full24k-alignedv4-q35-4b-babel-1ep-{job_id}"
    combined = tail_text(evidence_dir / f"{prefix}.out") + "\n" + tail_text(evidence_dir / f"{prefix}.err")
    loss_lines = [line[-2000:] for line in combined.splitlines() if LOSS_RE.search(line)]
    wandb_urls = sorted(set(WANDB_URL_RE.findall(combined)))
    wandb_run_urls = sorted(url for url in wandb_urls if "/runs/" in url)
    progress = [(int(a), int(b)) for a, b in PROGRESS_RE.findall(combined)]
    gpu = gpu_summary(evidence_dir, job_id)
    checkpoints = checkpoint_checks(run_root / "output_aligned_v4_qwen35_4b_base_full_sft_one_epoch")
    token_path = run_root / "dataset/tokenized_aligned_v4_manifest.json"
    tokenization = json.loads(token_path.read_text()) if token_path.is_file() else None
    release = json.loads((run_root / "dataset/manifest.json").read_text())
    rc, squeue, _ = command("squeue", "-h", "-j", job_id, "-o", "%T|%M|%N|%R")
    rc, sacct, _ = command(
        "sacct", "-j", job_id, "-X", "--format=State,ExitCode,Elapsed,NodeList", "-n", "-P"
    )
    restart_markers = [line for line in combined.splitlines() if line.startswith("restart_count=")]
    resume_markers = [line for line in combined.splitlines() if line.startswith("resume_from_checkpoint=")]
    run_id_markers = [line for line in combined.splitlines() if line.startswith("wandb_run_id=")]
    run_id_values = [line.split("=", 1)[1] for line in run_id_markers]
    restart_segment = combined.rsplit("restart_count=", 1)[-1] if "restart_count=" in combined else ""
    post_restart_losses = [line[-2000:] for line in restart_segment.splitlines() if LOSS_RE.search(line)]
    post_restart_progress = [(int(a), int(b)) for a, b in PROGRESS_RE.findall(restart_segment)]
    selection_path = evidence_dir / f"checkpoint_selection.{job_id}.restart_1.json"
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else None
    selected_step = (
        selection.get("selected_checkpoint", {}).get("step")
        if selection and selection.get("selected_checkpoint")
        else None
    )
    wandb_dirs = sorted(str(path) for path in (run_root / "wandb").glob(f"**/*{EXPECTED_WANDB_ID}*"))
    flags = {
        "job_running": squeue.startswith("RUNNING|"),
        "full_release_validated": (
            release.get("observed_config_count") == 52
            and release.get("source_rows") == 9_640_563
            and release.get("adapted_train_rows") == 9_196_689
            and release.get("training_alignment", {}).get("status") == "validated"
            and release.get("training_alignment", {}).get("alignment_schema_version") == 4
        ),
        "tokenized_complete": (
            tokenization is not None
            and tokenization.get("splits") == {"train": 9_196_689, "validation": 500}
            and tokenization.get("tokenizer_skipped_train_rows") == 0
            and tokenization.get("evaluation_skipped_rows") == 0
            and tokenization.get("total_tokenizer_filtered_rows") == 0
            and tokenization.get("llamafactory_abnormal_warning_count") == 0
            and tokenization.get("warning_count_matches_total_filtered") is True
        ),
        "real_loss_seen": bool(loss_lines),
        "gpu_active_seen": gpu["max_utilization_percent"] > 0 and gpu["max_memory_used_mib"] > 1000,
        "wandb_url_seen": bool(wandb_urls),
        "wandb_local_run_seen": bool(wandb_dirs),
        "complete_checkpoint_seen": any(item["complete"] for item in checkpoints),
    }
    return {
        "observed_at": now(),
        "job_id": job_id,
        "squeue": squeue,
        "sacct": sacct,
        "slurm_restart_count": restart_count(job_id),
        "flags": flags,
        "loss_lines_tail": loss_lines[-20:],
        "progress_max_current": max((item[0] for item in progress), default=None),
        "progress_totals": sorted(set(item[1] for item in progress))[-10:],
        "wandb_urls": wandb_urls,
        "wandb_run_urls": wandb_run_urls,
        "expected_wandb_run_id": EXPECTED_WANDB_ID,
        "wandb_dirs": wandb_dirs,
        "gpu": gpu,
        "checkpoints": checkpoints,
        "restart_markers": restart_markers[-10:],
        "resume_markers": resume_markers[-10:],
        "run_id_markers": run_id_markers[-10:],
        "run_id_values": run_id_values[-10:],
        "restart_1_checkpoint_selection": selection,
        "post_restart_loss_lines_tail": post_restart_losses[-20:],
        "post_restart_progress_max_current": max((item[0] for item in post_restart_progress), default=None),
        "tokenization": tokenization,
        "pre_requeue_gate": all(
            flags[key]
            for key in (
                "job_running",
                "full_release_validated",
                "tokenized_complete",
                "real_loss_seen",
                "gpu_active_seen",
                "wandb_url_seen",
                "wandb_local_run_seen",
                "complete_checkpoint_seen",
            )
        ),
        "restart_verification_gate": (
            (restart_count(job_id) or 0) >= 1
            and selection is not None
            and selected_step is not None
            and bool(post_restart_losses)
            and max((item[0] for item in post_restart_progress), default=-1) > selected_step
            and len(set(run_id_values)) == 1
            and set(run_id_values) == {EXPECTED_WANDB_ID}
            and set(wandb_run_urls) == {EXPECTED_WANDB_RUN_URL}
            and flags["job_running"]
            and flags["gpu_active_seen"]
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

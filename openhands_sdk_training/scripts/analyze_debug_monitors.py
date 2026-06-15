#!/usr/bin/env python3
"""Summarize lightweight Slurm debug monitor logs.

The debug Slurm launchers write one directory per job with nvidia-smi dmon,
sar, pidstat, mpstat, and /proc/net/dev snapshots. This script produces a
compact summary that is useful for comparing short training-loop burn-in runs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def elapsed_to_seconds(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    parts = value.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours = 0
        minutes, seconds = numbers
    else:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def fmt(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def parse_trainer_log(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []

    events = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
        if isinstance(record, dict):
            events.append(record)
    return events


def summarize_trainer(events: list[dict]) -> list[str]:
    rows = []
    steps = [event for event in events if "step" in event or "current_steps" in event]
    if not steps:
        return ["trainer_log: no parsed step records"]

    last = steps[-1]
    timed_steps: list[tuple[int, float]] = []
    for event in steps:
        step = event.get("current_steps", event.get("step"))
        seconds = elapsed_to_seconds(event.get("elapsed_time"))
        if isinstance(step, int) and seconds is not None:
            timed_steps.append((step, seconds))

    rows.append(
        "trainer_log: "
        f"records={len(events)} last_step={last.get('step', last.get('current_steps', 'n/a'))} "
        f"last_loss={last.get('loss', 'n/a')} "
        f"runtime={last.get('train_runtime', 'n/a')}"
    )
    if timed_steps:
        deltas = []
        prev_step = 0
        prev_seconds = 0.0
        for step, seconds in timed_steps:
            step_delta = step - prev_step
            if step_delta > 0:
                deltas.append((seconds - prev_seconds) / step_delta)
            prev_step = step
            prev_seconds = seconds
        rows.append(
            "trainer_steps: "
            f"first_step_s={fmt(deltas[0], 1)} "
            f"median_step_s={fmt(percentile(deltas, 50), 1)} "
            f"last_step_s={fmt(deltas[-1], 1)} "
            f"steps_per_hour={fmt(3600.0 / percentile(deltas, 50), 1)}"
        )
    return rows


def parse_dmon_file(path: Path) -> list[dict]:
    samples = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 21 or not parts[0].isdigit():
            continue
        try:
            samples.append(
                {
                    "file": path.name,
                    "date": parts[0],
                    "time": parts[1],
                    "gpu": int(parts[2]),
                    "pwr": float(parts[3]),
                    "sm": float(parts[6]),
                    "mem": float(parts[7]),
                    "fb_mb": float(parts[16]),
                    "rxpci_mb_s": float(parts[19]),
                    "txpci_mb_s": float(parts[20]),
                }
            )
        except ValueError:
            continue
    return samples


def summarize_dmon(debug_dir: Path) -> list[str]:
    rows = []
    for path in sorted(debug_dir.glob("nvidia_dmon_*.log")):
        samples = parse_dmon_file(path)
        if not samples:
            rows.append(f"{path.name}: no nvidia-smi dmon samples")
            continue

        by_gpu: dict[int, list[dict]] = defaultdict(list)
        for sample in samples:
            by_gpu[sample["gpu"]].append(sample)

        sm_values = [sample["sm"] for sample in samples]
        mem_values = [sample["mem"] for sample in samples]
        rx_values = [sample["rxpci_mb_s"] for sample in samples if not math.isnan(sample["rxpci_mb_s"])]
        tx_values = [sample["txpci_mb_s"] for sample in samples if not math.isnan(sample["txpci_mb_s"])]
        active = sum(1 for value in sm_values if value >= 50.0)

        rows.append(
            f"{path.name}: samples={len(samples)} gpus={len(by_gpu)} "
            f"sm_avg={fmt(mean(sm_values))}% sm_p50={fmt(percentile(sm_values, 50))}% "
            f"sm_p90={fmt(percentile(sm_values, 90))}% active_ge50={fmt(100 * active / len(sm_values))}% "
            f"mem_avg={fmt(mean(mem_values))}% fb_max={fmt(max(s['fb_mb'] for s in samples), 0)}MB "
            f"pcie_rx_max={fmt(max(rx_values), 0)}MB/s pcie_tx_max={fmt(max(tx_values), 0)}MB/s"
        )
    return rows or ["nvidia_dmon: no files"]


def parse_sar_file(path: Path) -> list[dict]:
    samples = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or "IFACE" in line or line.startswith("Linux"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        if parts[0] == "Average:":
            parts = parts[1:]
        if len(parts) < 10:
            continue
        iface = parts[1]
        if iface == "IFACE":
            continue
        try:
            samples.append(
                {
                    "file": path.name,
                    "time": parts[0],
                    "iface": iface,
                    "rx_mb_s": float(parts[4]) / 1024.0,
                    "tx_mb_s": float(parts[5]) / 1024.0,
                    "ifutil": float(parts[9]),
                }
            )
        except ValueError:
            continue
    return samples


def summarize_sar(debug_dir: Path, interfaces: set[str] | None) -> list[str]:
    rows = []
    for path in sorted(debug_dir.glob("sar_net_*.log")):
        samples = parse_sar_file(path)
        if interfaces:
            samples = [sample for sample in samples if sample["iface"] in interfaces]
        if not samples:
            rows.append(f"{path.name}: no sar network samples")
            continue

        by_iface: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            by_iface[sample["iface"]].append(sample)

        iface_summaries = []
        for iface, iface_samples in sorted(by_iface.items()):
            rx = [sample["rx_mb_s"] for sample in iface_samples]
            tx = [sample["tx_mb_s"] for sample in iface_samples]
            util = [sample["ifutil"] for sample in iface_samples]
            if max(rx + tx) < 1.0 and max(util) < 0.1:
                continue
            iface_summaries.append(
                f"{iface}:rx_avg={fmt(mean(rx))} rx_max={fmt(max(rx))} "
                f"tx_avg={fmt(mean(tx))} tx_max={fmt(max(tx))} util_max={fmt(max(util))}%"
            )

        if iface_summaries:
            rows.append(f"{path.name}: " + "; ".join(iface_summaries))
        else:
            rows.append(f"{path.name}: no active interfaces above threshold")
    return rows or ["sar_net: no files"]


def parse_pidstat_file(path: Path) -> list[dict]:
    samples = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "UID" in line or line.startswith("Linux"):
            continue
        parts = line.split()
        if len(parts) < 17:
            continue
        try:
            samples.append(
                {
                    "file": path.name,
                    "time": parts[0],
                    "pid": parts[2],
                    "usr": float(parts[3]),
                    "system": float(parts[4]),
                    "wait": float(parts[6]),
                    "cpu": float(parts[7]),
                    "read_kb_s": float(parts[13]),
                    "write_kb_s": float(parts[14]),
                    "command": parts[-1],
                }
            )
        except ValueError:
            continue
    return samples


def summarize_pidstat(debug_dir: Path) -> list[str]:
    rows = []
    for path in sorted(debug_dir.glob("pidstat_*.log")):
        samples = parse_pidstat_file(path)
        samples = [sample for sample in samples if "python" in sample["command"]]
        if not samples:
            rows.append(f"{path.name}: no pidstat python samples")
            continue

        cpu_values = [sample["cpu"] for sample in samples]
        wait_values = [sample["wait"] for sample in samples]
        read_values = [sample["read_kb_s"] for sample in samples]
        write_values = [sample["write_kb_s"] for sample in samples]
        rows.append(
            f"{path.name}: samples={len(samples)} "
            f"rank_cpu_avg={fmt(mean(cpu_values))}% rank_cpu_p90={fmt(percentile(cpu_values, 90))}% "
            f"rank_cpu_max={fmt(max(cpu_values))}% wait_max={fmt(max(wait_values))}% "
            f"read_max={fmt(max(read_values) / 1024.0)}MB/s write_max={fmt(max(write_values) / 1024.0)}MB/s"
        )
    return rows or ["pidstat: no files"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_dir", type=Path, help="Directory with debug_*/ monitor logs")
    parser.add_argument("--trainer-log", type=Path, default=None, help="Optional trainer_log.jsonl path")
    parser.add_argument(
        "--interfaces",
        default="enp6s0f0,enp7s0f0,enp13s0f0,enp14s0f0,enp134s0f0,enp135s0f0,enp141s0f0,enp142s0f0",
        help="Comma-separated network interfaces to include in sar summaries",
    )
    args = parser.parse_args()

    debug_dir = args.debug_dir
    if not debug_dir.is_dir():
        raise SystemExit(f"debug_dir does not exist: {debug_dir}")

    interfaces = {item for item in args.interfaces.split(",") if item}

    print(f"debug_dir: {debug_dir}")
    for row in summarize_trainer(parse_trainer_log(args.trainer_log)):
        print(row)

    print("\nGPU")
    for row in summarize_dmon(debug_dir):
        print(row)

    print("\nNetwork")
    for row in summarize_sar(debug_dir, interfaces):
        print(row)

    print("\nCPU/IO")
    for row in summarize_pidstat(debug_dir):
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

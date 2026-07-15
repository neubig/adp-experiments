#!/usr/bin/env python3
"""Fail fast if the job's four-rank CUDA/NCCL mapping or transport is broken."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=datetime.timedelta(seconds=90),
    )
    dist.barrier(device_ids=[local_rank])
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    if value.item() != 10.0:
        raise RuntimeError(f"unexpected all-reduce result on rank {rank}: {value.item()}")
    print(
        f"rank={rank} local_rank={local_rank} device={torch.cuda.current_device()} "
        f"barrier=ok all_reduce={value.item()}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

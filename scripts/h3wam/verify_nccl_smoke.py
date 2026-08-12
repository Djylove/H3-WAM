#!/usr/bin/env python3
"""Minimal multi-GPU NCCL smoke test for the H3-WAM training host."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))

    value = torch.tensor(float(dist.get_rank()), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() - 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce returned {value.item()}, expected {expected}")

    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"NCCL_OK world_size={dist.get_world_size()} "
            f"all_reduce_sum={value.item():.0f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

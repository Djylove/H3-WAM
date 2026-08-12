#!/usr/bin/env python3
"""Minimal eight-rank CUDA/NCCL/FSDP acceptance test for H3-WAM clusters."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    model = FSDP(
        torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 16),
        ).to(device),
        device_id=device,
        use_orig_params=True,
    )
    inputs = torch.randn(32, 128, device=device)
    loss = model(inputs).float().square().mean()
    loss.backward()

    allreduce = torch.ones((), device=device)
    dist.all_reduce(allreduce)
    record = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "loss": float(loss.detach()),
        "allreduce": float(allreduce),
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    if int(allreduce.item()) != dist.get_world_size():
        raise RuntimeError("NCCL all-reduce returned the wrong world size")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
